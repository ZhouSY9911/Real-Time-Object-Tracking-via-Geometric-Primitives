from pathlib import Path
import yaml

from extract_first_frame import read_first_frame_from_bag, save_rgbd_frame
from get_table import get_table
from pcd_generation import generate_object_pointcloud
from run_dsqnet import (
    DSQNetInference,
    clear_dsqnet_outputs,
    predict_and_save,
)
from sam_segment import (
    SAMConfig,
    build_sam_predictor,
    run_interactive_segmentation,
)


# This file is located in the source directory under MA.
# parent is the source directory; parent.parent is the MA project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "video_sam.yaml"


def resolve_path(path):
    """Resolve relative paths in the configuration against the MA project root."""
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_cfg(path):
    path = resolve_path(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    for section, key in (("realsense", "bag_path"), ("sam", "checkpoint")):
        values = cfg.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Missing or invalid configuration section: {section}")
        if not isinstance(values.get(key), str) or not values[key].strip():
            raise ValueError(f"Missing or invalid configuration field: {section}.{key}")
    cfg.setdefault("runtime", {})
    if not isinstance(cfg["runtime"], dict):
        raise ValueError("Configuration section runtime must be a mapping")
    cfg.setdefault("dsqnet", {})
    if not isinstance(cfg["dsqnet"], dict):
        raise ValueError("Configuration section dsqnet must be a mapping")
    cfg.setdefault("table", {})
    if not isinstance(cfg["table"], dict):
        raise ValueError("Configuration section table must be a mapping")
    return cfg


def main(cfg_path=DEFAULT_CONFIG_PATH):
    cfg = load_cfg(cfg_path)

    bag_path = resolve_path(cfg["realsense"]["bag_path"])

    save_dir = resolve_path(
        cfg["runtime"].get(
            "save_dir",
            "outputs/first_frame_multiobj",
        )
    )
    table_dir = resolve_path("outputs/table")
    # Object segmentation clears its output directory. Neither stage may own
    # the other's directory, including through a configured parent or symlink.
    if (save_dir == table_dir or save_dir in table_dir.parents
            or table_dir in save_dir.parents):
        raise ValueError("runtime.save_dir must be separate from outputs/table (no nesting)")
    save_dir.mkdir(parents=True, exist_ok=True)

    sam_cfg = SAMConfig(
        checkpoint=str(
            resolve_path(cfg["sam"]["checkpoint"])
        ),
        model_type=cfg["sam"].get("model_type", "vit_b"),
        device=cfg["sam"].get("device", "cuda"),
    )

    print("[INFO] Loading frame from bag...")

    frame0, depth0, intr, depth_scale = read_first_frame_from_bag(
        str(bag_path),
        skip_frames=cfg["realsense"].get("skip_frames", 0),
    )
    window_options = {
        "window_width": int(cfg["runtime"].get("window_width", 640)),
        "window_height": int(cfg["runtime"].get("window_height", 480)),
        "min_box_size": int(cfg["runtime"].get("min_box_size", 5)),
    }
    print("[INFO] Building SAM predictor for table and object segmentation...")
    predictor = build_sam_predictor(sam_cfg)
    table_cfg = cfg.get("table", {})
    try:
        table_info = get_table(
            frame0, depth0, intr, depth_scale, sam_cfg, output_dir=table_dir,
            predictor=predictor, **window_options,
            distance_threshold=table_cfg.get("distance_threshold", 0.005),
            num_iterations=table_cfg.get("num_iterations", 1000),
            min_points=table_cfg.get("min_points", 100),
            min_inlier_ratio=table_cfg.get("min_inlier_ratio", 0.5),
        )
        save_rgbd_frame(table_dir, frame0, depth0, intr, depth_scale)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[ERROR] Table estimation failed: {exc}. Object processing was not started.")
        return
    print(f"[OK] Table saved to {table_dir}: "
          f"{table_info['inlier_count']}/{table_info['point_count']} inliers, "
          f"RMSE={table_info['rmse_m'] * 1000:.2f} mm")

    print("[INFO] Now select the objects on the same first frame.")
    all_masks_u8, _, _ = run_interactive_segmentation(
        frame0=frame0,
        sam_cfg=sam_cfg,
        save_dir=save_dir,
        predictor=predictor,
        **window_options,
    )
    del predictor  # Release SAM weights before loading DSQNet.
    # Segmentation clears save_dir, so persist RGB-D after it finishes.
    save_rgbd_frame(save_dir, frame0, depth0, intr, depth_scale)

    print("[INFO] Generating object point clouds and fitting DSQNet...")
    dsqnet = None  # Load once, when the first nonempty cloud is available.
    for obj_id, mask_u8 in enumerate(all_masks_u8):
        obj_dir = save_dir / f"obj_{obj_id:03d}"
        clear_dsqnet_outputs(obj_dir)
        output_path = obj_dir / "pointcloud.ply"
        point_count = generate_object_pointcloud(
            mask_u8=mask_u8,
            depth_u16=depth0,
            image_bgr=frame0,
            intr=intr,
            depth_scale=depth_scale,
            output_path=output_path,
        )
        if point_count == 0:
            print(f"[WARN] object {obj_id}: no points remain after depth/filter checks; no point cloud saved.")
            continue
        print(f"[OK] object {obj_id}: {point_count} points saved to {output_path}")

        if dsqnet is None:
            dsqnet = DSQNetInference(device=cfg["dsqnet"].get("device", "cuda:0"))
        try:
            predict_and_save(
                dsqnet, output_path, output_dir=obj_dir,
                T_camera_table=table_info["T_camera_table"],
                grasp_num_samples=cfg["dsqnet"].get("grasp_num_samples", 200),
            )
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"[WARN] object {obj_id}: DSQNet failed: {exc}")
            continue
        print(f"[OK] object {obj_id}: DSQNet pose, parameters, mesh and visualization saved to {obj_dir}")


if __name__ == "__main__":
    main()
