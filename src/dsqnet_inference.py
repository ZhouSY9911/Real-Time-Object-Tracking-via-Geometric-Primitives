import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


# ============================================================
# DSQNet path
# ============================================================

DSQNET_ROOT = Path.home() / "MA" / "third_party" / "DSQNet-public"

if str(DSQNET_ROOT) not in sys.path:
    sys.path.insert(0, str(DSQNET_ROOT))


# ============================================================
# DSQNet official modules
# ============================================================

from models import load_pretrained # type: ignore

from functions.pc_preprocess import ( # type: ignore
    normalize_numpy_pc,
    transform_to_pc_frame,
    upsampling,
    voxel_downsample_npoints,
)

from functions.utils_numpy import ( # type: ignore
    quaternions_to_rotation_matrices,
    define_SE3
)


def _validate_geometry(points):
    """Reject clouds for which a stable 3D object frame cannot be computed."""
    if len(points) < 20:
        raise ValueError("Point cloud requires at least 20 distinct valid points.")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates.")

    centered = points - points.mean(axis=0)
    scale = np.linalg.norm(np.ptp(points, axis=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Point cloud has an invalid or zero spatial extent.")
    singular_values = np.linalg.svd(centered / scale, compute_uv=False)
    if singular_values[-1] <= singular_values[0] * 1e-6:
        raise ValueError("Point cloud is degenerate: planar, collinear, or too thin.")


def _preprocess_point_cloud(pcd, num_points=300):
    """Use official sampling/frame routines with safe sampling fallbacks."""
    # Work on an independent cloud; only positions are needed for inference.
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pcd.points)))
    pcd.remove_non_finite_points()
    pcd.remove_duplicated_points()
    points = np.asarray(pcd.points).copy()
    _validate_geometry(points)
    if num_points < 20:
        raise ValueError("DSQNet requires at least 20 sampled points.")

    if len(points) > num_points:
        try:
            # Call directly to avoid the upstream resampler's undefined floor.
            sampled = np.asarray(voxel_downsample_npoints(pcd, num_points).points).copy()
            if sampled.shape != (num_points, 3):
                raise ValueError("Voxel downsampling returned the wrong point count.")
            _validate_geometry(sampled)
        except (RuntimeError, ValueError):
            indices = np.random.default_rng(0).choice(len(points), num_points, replace=False)
            sampled = points[indices]
    elif len(points) < num_points:
        # Center and scale before the upstream float32 SVD to reduce roundoff.
        center = points.mean(axis=0)
        scale = np.linalg.norm(np.ptp(points, axis=0))
        try:
            sampled = upsampling((points - center) / scale, num_points)
            if sampled.shape != (num_points, 3) or not np.isfinite(sampled).all():
                raise ValueError("Upsampling returned invalid points.")
            sampled = sampled * scale + center
            _validate_geometry(sampled)
        except (RuntimeError, ValueError):
            # Retain every original point without inventing noisy geometry.
            sampled = points[np.arange(num_points) % len(points)]
    else:
        sampled = points

    _validate_geometry(sampled)
    pcd.points = o3d.utility.Vector3dVector(sampled)
    pcd_processed, pcd_frame = transform_to_pc_frame(pcd)
    if pcd_processed is None or pcd_frame is None:
        raise ValueError("Cannot compute a 3D bounding box for this point cloud.")
    if not np.isfinite(pcd_frame).all():
        raise ValueError("Point cloud preprocessing produced an invalid frame.")
    return pcd_processed, pcd_frame


# ============================================================
# Convert DSQNet output -> geometric parameters
# Based on official evaluation.py
# ============================================================

def pred_to_parameters(pred_fit, normalizer, pcd_frame):

    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    quaternion = pred_fit[3:7]

    rotation = quaternions_to_rotation_matrices(
        np.expand_dims(quaternion, axis=0)
    )

    rotation = np.squeeze(rotation)

    translation = normalizer * pred_fit[:3]

    SE3 = define_SE3(
        rotation,
        translation
    )

    SE3 = np.dot(pcd_frame, SE3)

    # --------------------------------------------------------
    # Shape parameters
    # --------------------------------------------------------

    parameters = {
        "a1": float(pred_fit[7] * normalizer),
        "a2": float(pred_fit[8] * normalizer),
        "a3": float(pred_fit[9] * normalizer),

        "e1": float(pred_fit[10]),
        "e2": float(pred_fit[11]),

        "k": float(pred_fit[12]),
        "b": float(pred_fit[13]),

        "cos_alpha": float(pred_fit[14]),
        "sin_alpha": float(pred_fit[15]),
    }

    return SE3, parameters


# ============================================================
# DSQNet inference class
# ============================================================

class DSQNetInference:

    def __init__(self, device="cuda:0"):

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA unavailable, use CPU.")
            device = "cpu"

        self.device = torch.device(device)

        pretrained_root = DSQNET_ROOT / "pretrained"

        # official pretrained configuration
        identifier = "dsqnet_config/example"
        config_file = "dsqnet_config.yml"
        checkpoint_file = "model_best.pkl"

        print("Loading DSQNet...")

        self.model, self.cfg = load_pretrained(
            identifier=identifier,
            config_file=config_file,
            ckpt_file=checkpoint_file,
            root=str(pretrained_root)
        )

        self.model.to(self.device)

        # VERY IMPORTANT
        self.model.eval()

        print("DSQNet loaded successfully.")
        print("Device:", self.device)


    def predict(self, ply_path):

        ply_path = str(ply_path)

        # ====================================================
        # 1. Read .ply
        # ====================================================

        print("\nLoading point cloud:")
        print(ply_path)

        pcd = o3d.io.read_point_cloud(ply_path)

        if pcd.is_empty():
            raise ValueError(
                f"Point cloud is empty: {ply_path}"
            )

        points_original = np.asarray(pcd.points)

        print("Original points:", len(points_original))


        # ====================================================
        # 2. Clean, validate, and resample the point cloud
        # ====================================================

        # Official evaluation uses 300 points for primitive fit.

        pcd_processed, pcd_frame = _preprocess_point_cloud(
            pcd,
            num_points=300
        )

        points = np.asarray(pcd_processed.points)
        _validate_geometry(points)

        print("Processed points:", len(points))


        # ====================================================
        # 3. Normalize
        # ====================================================

        points_normalized, normalizer = normalize_numpy_pc(
            points
        )
        if (
            not np.isfinite(normalizer)
            or normalizer <= 0
            or not np.isfinite(points_normalized).all()
        ):
            raise ValueError("Point cloud normalization produced invalid values.")


        # ====================================================
        # 4. numpy -> tensor
        #
        # N x 3
        # ->
        # 1 x 3 x N
        # ====================================================

        points_tensor = torch.tensor(
            points_normalized.T[None, :, :],
            dtype=torch.float32,
            device=self.device
        )

        print(
            "Input tensor shape:",
            tuple(points_tensor.shape)
        )


        # ====================================================
        # 5. DSQNet inference
        # ====================================================

        with torch.no_grad():

            prediction = self.model(
                points_tensor
            )

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )
        if prediction.shape != (16,) or not np.isfinite(prediction).all():
            raise RuntimeError("DSQNet must return 16 finite prediction values.")

        print(
            "DSQNet output shape:",
            prediction.shape
        )


        # ====================================================
        # 6. Recover physical scale + pose
        # ====================================================

        SE3, parameters = pred_to_parameters(
            prediction,
            normalizer,
            pcd_frame
        )


        # ====================================================
        # Result
        # ====================================================

        result = {
            "SE3": SE3,
            "parameters": parameters,
            "raw_prediction": prediction
        }

        return result
