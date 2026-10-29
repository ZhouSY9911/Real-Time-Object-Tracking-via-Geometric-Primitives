"""Generate and clean colored object point clouds from aligned RGB-D and masks.

Use the color image, aligned depth, intrinsics and depth scale returned by
read_first_frame_from_bag. Masks must refer to that same frame and resolution.
Coordinates are in meters: X right, Y down, Z forward in the camera frame.
Requires Open3D. Depth is always multiplied by depth_scale; use 1.0 for
depth images that are already in meters.
"""

import copy
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def mask_to_pointcloud(
    mask_u8, depth_u16, image_bgr, intr, depth_scale, save_path=None,
    *, min_range=0.05, max_range=3.0, depth_band=0.15, voxel=0.003,
    erode_mask=False, erode_iter=1, remove_outlier=True,
    estimate_normals=False, normal_radius=0.02, normal_max_nn=30,
):
    """Return an Open3D PointCloud, or None if no points survive filtering.

    The first five arguments match the existing RealSense pipeline:
      mask_u8: HxW boolean, 0/1 or 0/255 mask.
      depth_u16: HxW raw depth; floating-point depth is also accepted.
      image_bgr: HxWx3 uint8 color image from the same frame.
      intr: RealSense intrinsics or a NumPy 3x3 K matrix for rectified images.
      depth_scale: multiplier converting the supplied depth values to meters.

    Processing: optional 3x3 mask erosion, depth range, median depth band,
    back-projection, voxel downsampling, statistical outlier removal, and
    optional normal estimation. RGB colors are normalized to [0, 1].

    All distance options are in meters. depth_band is the half-width around
    the median; set it to None to retain objects with a large depth extent.
    Set voxel to None or 0 to disable downsampling. Outlier removal is skipped
    for clouds with at most 25 points; normals require at least 3 points.
    If save_path is supplied, save the final cloud as PLY.
    """
    if not np.isfinite(min_range) or min_range < 0:
        raise ValueError("min_range must be finite and non-negative")
    if np.isnan(max_range) or max_range <= min_range:
        raise ValueError("max_range must be greater than min_range")
    if depth_band is not None and (not np.isfinite(depth_band) or depth_band <= 0):
        raise ValueError("depth_band must be positive or None")
    if voxel is not None and (not np.isfinite(voxel) or voxel < 0):
        raise ValueError("voxel must be non-negative or None")
    if erode_mask and (
        isinstance(erode_iter, bool) or not isinstance(erode_iter, int) or erode_iter < 0
    ):
        raise ValueError("erode_iter must be a non-negative integer")
    if estimate_normals:
        if not np.isfinite(normal_radius) or normal_radius <= 0:
            raise ValueError("normal_radius must be finite and positive")
        if isinstance(normal_max_nn, bool) or not isinstance(normal_max_nn, int) or normal_max_nn < 3:
            raise ValueError("normal_max_nn must be an integer of at least 3")

    mask = np.asarray(mask_u8)
    depth = np.asarray(depth_u16)
    image = np.asarray(image_bgr)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Mask and aligned depth must have the same (H, W) shape")
    if image.shape != (*depth.shape, 3) or image.dtype != np.uint8:
        raise ValueError("Color image must be uint8 BGR with shape (H, W, 3)")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    if isinstance(intr, np.ndarray):
        K = np.asarray(intr, dtype=np.float64)
        if (K.shape != (3, 3) or not np.isfinite(K).all()
                or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1])
                or np.linalg.det(K) == 0):
            raise ValueError("Camera matrix must be a valid invertible 3x3 K matrix")
    else:
        if (intr.height, intr.width) != depth.shape:
            raise ValueError("Camera intrinsics resolution must match aligned depth")
        if not np.all(np.isfinite([intr.fx, intr.fy, intr.ppx, intr.ppy])) or min(intr.fx, intr.fy) <= 0:
            raise ValueError("Camera intrinsics must be finite with positive focal lengths")

    selected = np.isfinite(mask) & (mask > 0)
    if erode_mask and erode_iter > 0:
        selected = cv2.erode(
            selected.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
            iterations=erode_iter, borderType=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)

    # Do not guess units from the magnitude of a depth value.
    depth_m = depth.astype(np.float64) * depth_scale
    valid = selected & np.isfinite(depth_m) & (depth_m > min_range) & (depth_m < max_range)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return None

    z = depth_m[ys, xs]
    if depth_band is not None:
        keep = np.abs(z - np.median(z)) <= depth_band
        xs, ys, z = xs[keep], ys[keep], z[keep]
        if z.size == 0:
            return None

    if isinstance(intr, np.ndarray):
        pixels = np.column_stack((xs, ys, np.ones(len(xs))))
        points = (pixels @ np.linalg.inv(K).T) * z[:, None]
    else:
        # Only the sensor-specific branch needs the RealSense SDK.
        import pyrealsense2 as rs

        brown_models = (
            rs.distortion.brown_conrady,
            rs.distortion.inverse_brown_conrady,
            rs.distortion.modified_brown_conrady,
        )
        pinhole = intr.model == rs.distortion.none or (
            intr.model in brown_models and not np.any(intr.coeffs)
        )
        if pinhole:
            x = (xs - intr.ppx) * z / intr.fx
            y = (ys - intr.ppy) * z / intr.fy
            points = np.column_stack((x, y, z)).astype(np.float32)
        else:
            if intr.model == rs.distortion.modified_brown_conrady:
                raise ValueError(
                    "Nonzero modified Brown-Conrady distortion is unsupported; "
                    "rectify the color/depth images and use matching intrinsics first"
                )
            points = np.asarray([
                rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(d))
                for u, v, d in zip(xs, ys, z)
            ], dtype=np.float32)

    colors = image[ys, xs, ::-1].astype(np.float64) / 255.0  # BGR -> RGB
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[finite])
    pcd.colors = o3d.utility.Vector3dVector(colors[finite])

    if voxel is not None and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
    if remove_outlier and len(pcd.points) > 25:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    if len(pcd.points) == 0:
        return None
    if estimate_normals and len(pcd.points) >= 3:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn,
        ))
    if save_path is not None:
        save_ply(save_path, pcd)
    return pcd


def save_ply(path, pcd):
    """Save an Open3D cloud with colors and any normals as binary PLY."""
    points = np.asarray(pcd.points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Cannot save an empty point cloud")
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates must be finite")
    path = Path(path)
    if path.suffix.lower() != ".ply":
        raise ValueError("Point cloud output path must end in .ply")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd, write_ascii=False):
        raise OSError(f"Failed to save point cloud: {path}")
    return path


def generate_object_pointcloud(
    mask_u8, depth_u16, image_bgr, intr, depth_scale, output_path,
    **options,
):
    """Generate and save one object's point cloud as PLY; return its point count.

    Keeps main.py's existing interface. Filtering options are forwarded to
    mask_to_pointcloud. Return 0 without saving if no points survive.
    """
    pcd = mask_to_pointcloud(
        mask_u8, depth_u16, image_bgr, intr, depth_scale,
        save_path=output_path, **options,
    )
    return 0 if pcd is None else len(pcd.points)


def pcd_to_numpy(pcd):
    """Return an independent Nx3 array of point coordinates in meters."""
    return np.asarray(pcd.points).copy()


def transform_pcd(pcd, T):
    """Apply a finite 4x4 homogeneous transform to a copy of the cloud."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("T must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("T must have homogeneous last row [0, 0, 0, 1]")
    transformed = copy.deepcopy(pcd)
    transformed.transform(T)
    return transformed
