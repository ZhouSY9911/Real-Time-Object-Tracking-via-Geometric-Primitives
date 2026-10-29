import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


# =========================
# Mouse box selection state
# =========================
drawing = False
ix, iy = -1, -1
roi_box = None
roi_done = False


@dataclass
class SAMConfig:
    checkpoint: str
    model_type: str = "vit_b"
    device: str = "cuda"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def prepare_output_dir(save_dir):
    """Clear previous results and reuse the configured output directory."""
    ensure_dir(save_dir)
    with os.scandir(save_dir) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
    return save_dir


def write_image(path, image):
    """Report failed writes before claiming that an object was saved."""
    try:
        success = cv2.imwrite(str(path), image)
    except cv2.error as exc:
        raise OSError(f"Failed to save image: {path}") from exc
    if not success:
        raise OSError(f"Failed to save image: {path}")


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, roi_box, roi_done

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi_box = None
        roi_done = False

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_box = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_box = (ix, iy, x, y)
        roi_done = True


def build_sam_predictor(cfg: SAMConfig):
    if not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {cfg.checkpoint}"
        )

    sam = sam_model_registry[cfg.model_type](
        checkpoint=cfg.checkpoint
    )
    sam.to(device=cfg.device)

    return SamPredictor(sam)


def overlay_masks_on_image(image_bgr, masks_u8):
    """Overlay multiple masks with colors and contours."""
    vis = image_bgr.copy()

    if len(masks_u8) == 0:
        return vis

    colors = [
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
    ]

    for i, mask_u8 in enumerate(masks_u8):
        color = colors[i % len(colors)]
        mask_bool = mask_u8 > 0

        color_layer = np.zeros_like(vis, dtype=np.uint8)
        color_layer[mask_bool] = color

        vis = cv2.addWeighted(
            vis, 1.0,
            color_layer, 0.35,
            0,
        )

        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis, contours, -1, color, 2)

    return vis


def mask_to_rgba(image_bgr, mask_u8):
    rgba = np.zeros(
        (image_bgr.shape[0], image_bgr.shape[1], 4),
        dtype=np.uint8,
    )
    rgba[..., :3] = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )
    rgba[..., 3] = mask_u8

    return rgba


def save_single_object(
    save_dir,
    obj_id,
    frame_bgr,
    mask_u8,
):
    obj_dir = ensure_dir(
        os.path.join(save_dir, f"obj_{obj_id:03d}")
    )

    write_image(
        os.path.join(obj_dir, "mask.png"),
        mask_u8,
    )

    # Save the overlay image.
    overlay = frame_bgr.copy()
    color_mask = np.zeros_like(frame_bgr)
    color_mask[:, :, 1] = mask_u8

    overlay = cv2.addWeighted(
        overlay, 0.7,
        color_mask, 0.3,
        0,
    )
    write_image(
        os.path.join(obj_dir, "overlay.png"),
        overlay,
    )

    # Save the image with a transparent background.
    rgba = mask_to_rgba(frame_bgr, mask_u8)

    write_image(
        os.path.join(obj_dir, "cutout.png"),
        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
    )


def save_summary(save_dir, frame_bgr, all_masks_u8):
    write_image(
        os.path.join(save_dir, "frame0.png"),
        frame_bgr,
    )

    merged = np.zeros(
        frame_bgr.shape[:2],
        dtype=np.uint8,
    )

    for mask in all_masks_u8:
        merged = np.maximum(merged, mask)

    write_image(
        os.path.join(save_dir, "all_masks_union.png"),
        merged,
    )

    vis = overlay_masks_on_image(frame_bgr, all_masks_u8)

    write_image(
        os.path.join(save_dir, "all_masks_overlay.png"),
        vis,
    )


def run_interactive_segmentation(
    frame0,
    sam_cfg,
    save_dir,
    window_width=1280,
    window_height=960,
    min_box_size=5,
    single_object=False,
    *,
    predictor=None,
    window_name=None,
    save_results=True,
):
    """Segment boxes; single_object mode replaces the previous selection on redraw.

    A supplied predictor reuses model weights between stages. Its image is reset
    on every call. With save_results=False this function performs no file writes
    or directory cleanup; the caller owns saving the confirmed selection.
    """
    global drawing, ix, iy, roi_box, roi_done

    # Reset the mouse state on each call.
    drawing = False
    ix, iy = -1, -1
    roi_box = None
    roi_done = False

    if save_results:
        save_dir = prepare_output_dir(str(save_dir))
        print(f"[INFO] Output directory: {save_dir}")

    if predictor is None:
        print("[INFO] Building SAM predictor...")
        predictor = build_sam_predictor(sam_cfg)

    rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb0)

    all_masks_u8 = []
    all_boxes = []
    all_scores = []

    if window_name is None:
        window_name = ("SAM: draw target box, redraw to replace, ESC to confirm" if single_object
                       else "Multi-object SAM on first frame (ESC to exit)")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        window_width,
        window_height,
    )
    cv2.setMouseCallback(window_name, mouse_callback)

    if single_object:
        print("[INFO] Draw a box around the target. Redraw to replace the mask; ESC confirms.")
    else:
        print("[INFO] Draw boxes repeatedly on the same image to segment different objects.")
    print("[INFO] Release the mouse button to segment and preview the mask.")
    print("[INFO] Press ESC to finish this segmentation stage.")

    try:
        while True:
            vis = overlay_masks_on_image(frame0, all_masks_u8)

            if roi_box is not None and not roi_done:
                x1, y1, x2, y2 = roi_box

                cv2.rectangle(
                    vis,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] ESC pressed, exiting...")
                break

            if roi_done and roi_box is not None:
                x1, y1, x2, y2 = roi_box

                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                roi_done = False
                roi_box = None

                if (
                    (x2 - x1) < min_box_size
                    or (y2 - y1) < min_box_size
                ):
                    print("[WARN] ROI too small, skip.")
                    continue

                box_np = np.array(
                    [[x1, y1, x2, y2]],
                    dtype=np.float32,
                )

                masks, scores, _ = predictor.predict(
                    box=box_np,
                    multimask_output=False,
                )

                mask_u8 = masks[0].astype(np.uint8) * 255
                score = float(scores[0])
                if single_object:
                    all_masks_u8.clear()
                    all_boxes.clear()
                    all_scores.clear()
                obj_id = len(all_masks_u8)

                if save_results:
                    save_single_object(
                        save_dir=save_dir,
                        obj_id=obj_id,
                        frame_bgr=frame0,
                        mask_u8=mask_u8,
                    )

                all_masks_u8.append(mask_u8)
                all_boxes.append((x1, y1, x2, y2))
                all_scores.append(score)

                if save_results:
                    save_summary(
                        save_dir,
                        frame0,
                        all_masks_u8,
                    )

                print(
                    f"[OK] object {obj_id} segmented, "
                    f"score={score:.4f}"
                )

        # Save the summary once more before exiting normally.
        if save_results:
            save_summary(save_dir, frame0, all_masks_u8)

    finally:
        cv2.destroyAllWindows()

    print(
        f"[INFO] Total objects segmented: {len(all_masks_u8)}"
    )
    if save_results:
        print(f"[INFO] Results saved to: {save_dir}")

    return all_masks_u8, all_boxes, all_scores
