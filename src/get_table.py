"""Estimate a table plane from a SAM mask and aligned RGB-D, in meters.

The camera must view the upper side of the table: the normal is oriented toward
the camera, not inferred from gravity. T_camera_table maps table coordinates to
camera coordinates (X right, Y down, Z forward). Table Z points off the surface;
the origin is the projected inlier centroid and X is the projected camera X axis
(camera Y is used near degeneracy). A plane does not define a persistent physical
origin or table boundary. Re-estimate after moving the camera or table.
"""

import json
from pathlib import Path

import numpy as np

from pcd_generation import mask_to_pointcloud, save_ply


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "table"
OUTPUT_FILES = (
    "table_mask.png", "table_overlay.png", "table_pointcloud.ply",
    "table_inliers.ply", "plane_camera.txt", "T_camera_table.txt",
    "T_table_camera.txt", "table_info.json",
)


def clear_table_outputs(output_dir):
    """Remove only generated table results; retain unrelated files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        (output_dir / filename).unlink(missing_ok=True)


def fit_table_plane(
    pcd, *, distance_threshold=0.005, num_iterations=1000,
    min_points=100, min_inlier_ratio=0.5,
):
    """Fit and validate a dominant plane; return metadata and point indices.

    Signed height in meters is points_camera @ normal_camera + offset_m.
    The thresholds reject sparse/poor selections; they cannot tell a tabletop
    from another planar surface, so the SAM preview must identify the table.
    """
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive and finite")
    for name, value, minimum in (("num_iterations", num_iterations, 1),
                                 ("min_points", min_points, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(min_inlier_ratio) or not 0 < min_inlier_ratio <= 1:
        raise ValueError("min_inlier_ratio must be in (0, 1]")
    points = np.asarray(pcd.points)
    if len(points) < min_points or not np.isfinite(points).all():
        raise ValueError(f"Table requires at least {min_points} finite points")
    _, indices = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3,
        num_iterations=num_iterations,
    )
    if len(indices) < min_points or len(indices) / len(points) < min_inlier_ratio:
        raise ValueError("No dominant table plane; select a larger clear tabletop region")

    # Least-squares refinement on the RANSAC consensus, with a degeneracy check.
    selected = points[indices]
    center = selected.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(selected - center, full_matrices=False)
    if singular_values[1] <= max(1e-6, singular_values[0] * 1e-3):
        raise ValueError("Table points are collinear or cover too narrow a strip")
    normal = vh[-1]
    offset = -float(normal @ center)
    if offset < 0:  # The camera origin must lie on the positive side.
        normal = -normal
        offset = -offset
    if offset <= distance_threshold:
        raise ValueError("Table plane passes too close to the camera to orient reliably")

    residuals = points @ normal + offset
    indices = np.flatnonzero(np.abs(residuals) <= distance_threshold)
    if len(indices) < min_points or len(indices) / len(points) < min_inlier_ratio:
        raise ValueError("Refined table plane has insufficient support")
    center = points[indices].mean(axis=0)
    origin = center - (normal @ center + offset) * normal
    reference = np.array([1., 0., 0.])
    if abs(reference @ normal) > 0.95:
        reference = np.array([0., 1., 0.])
    x_axis = reference - (reference @ normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    T_camera_table = np.eye(4)
    T_camera_table[:3, :3] = np.column_stack((x_axis, y_axis, normal))
    T_camera_table[:3, 3] = origin
    T_table_camera = np.linalg.inv(T_camera_table)

    info = {
        "units": "meters",
        "camera_axes": "X right, Y down, Z forward",
        "plane_equation": "normal_camera dot point_camera + offset_m = 0",
        "normal_camera": normal.tolist(),
        "offset_m": offset,
        "normal_convention": "toward camera; assumes camera above tabletop",
        "table_origin_camera": origin.tolist(),
        "table_axes": "Z=normal; X=projected camera X (Y fallback); Y=Z cross X",
        "transform_convention": "T_A_B maps coordinates in B to coordinates in A",
        "T_camera_table": T_camera_table.tolist(),
        "T_table_camera": T_table_camera.tolist(),
        "point_count": len(points),
        "inlier_count": len(indices),
        "inlier_ratio": len(indices) / len(points),
        "rmse_m": float(np.sqrt(np.mean(residuals[indices] ** 2))),
        "distance_threshold_m": float(distance_threshold),
        "num_iterations": num_iterations,
        "min_points": min_points,
        "min_inlier_ratio": float(min_inlier_ratio),
        "scope": "plane only, no table boundary; valid for this camera/table configuration",
    }
    return info, indices


def estimate_and_save_table(
    mask_u8, depth_u16, image_bgr, intr, depth_scale,
    output_dir=DEFAULT_OUTPUT_DIR, *, distance_threshold=0.005,
    num_iterations=1000, min_points=100, min_inlier_ratio=0.5,
):
    """Fit a confirmed table mask and save geometry plus inspectable diagnostics.

    table_info.json is written last and marks a complete successful estimate.
    Failed reruns remove old estimates instead of silently reusing calibration.
    """
    # Keep geometric functions usable without importing/loading SAM.
    import cv2

    output_dir = Path(output_dir)
    clear_table_outputs(output_dir)
    pcd = mask_to_pointcloud(
        mask_u8, depth_u16, image_bgr, intr, depth_scale,
        depth_band=None, erode_mask=True,
    )
    if pcd is None:
        raise ValueError("Table mask contains no valid depth points")
    info, indices = fit_table_plane(
        pcd, distance_threshold=distance_threshold, num_iterations=num_iterations,
        min_points=min_points, min_inlier_ratio=min_inlier_ratio,
    )
    mask = (np.asarray(mask_u8) > 0).astype(np.uint8) * 255
    overlay = image_bgr.copy()
    selected = mask > 0
    overlay[selected] = (
        overlay[selected].astype(float) * 0.65 + np.array([0, 255, 0]) * 0.35
    ).astype(np.uint8)
    for name, image in (("table_mask.png", mask), ("table_overlay.png", overlay)):
        if not cv2.imwrite(str(output_dir / name), image):
            raise OSError(f"Failed to save {name}")
    save_ply(output_dir / "table_pointcloud.ply", pcd)
    save_ply(output_dir / "table_inliers.ply", pcd.select_by_index(indices.tolist()))
    np.savetxt(output_dir / "plane_camera.txt",
               [info["normal_camera"] + [info["offset_m"]]], fmt="%.12g")
    for name in ("T_camera_table", "T_table_camera"):
        np.savetxt(output_dir / f"{name}.txt", info[name], fmt="%.12g")
    (output_dir / "table_info.json").write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return info


def get_table(
    frame0, depth0, intr, depth_scale, sam_cfg, output_dir=DEFAULT_OUTPUT_DIR,
    *, predictor=None, window_width=640, window_height=480, min_box_size=5,
    distance_threshold=0.005, num_iterations=1000, min_points=100,
    min_inlier_ratio=0.5,
):
    """Interactively select one tabletop with SAM, then estimate and save it.

    Redrawing replaces the selection. ESC confirms; leaving without a mask is
    an error. Outputs are separate from the subsequent object segmentation.
    """
    from sam_segment import run_interactive_segmentation

    clear_table_outputs(output_dir)
    print("[INFO] Select the tabletop with SAM. Check the mask; redraw if needed.")
    print("[INFO] ESC confirms the table and continues to object segmentation.")
    masks, _, _ = run_interactive_segmentation(
        frame0, sam_cfg, output_dir, window_width=window_width,
        window_height=window_height, min_box_size=min_box_size,
        single_object=True, predictor=predictor, save_results=False,
        window_name="SAM table: draw tabletop box, redraw to replace, ESC to confirm",
    )
    if len(masks) != 1:
        raise ValueError("No table selected; draw a tabletop box before confirming")
    return estimate_and_save_table(
        masks[0], depth0, frame0, intr, depth_scale, output_dir,
        distance_threshold=distance_threshold, num_iterations=num_iterations,
        min_points=min_points, min_inlier_ratio=min_inlier_ratio,
    )
