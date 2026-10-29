"""Generate DSQNet grasps in table coordinates and save them in camera coordinates.

Uses the upstream hand frame (not a calibrated robot TCP). Collision checks cover
the fitted object and the table halfspace only, not other objects or motion paths.
The upstream gripper loads relative asset paths: generation temporarily changes
the process working directory and must be called sequentially, as in main.py.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path

import fcl
import numpy as np

from dsqnet_inference import DSQNET_ROOT
from functions.object_class import Object
from functions.primitives import DeformableSuperquadric
from grasping.grasp_points_sampler import generate_collision_free_grasp_pose
from grasping.gripper import Gripper


OUTPUT_FILES = ("grasp_pose.txt", "grasp_info.json")


def validate_pose(value, name):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return pose


@contextmanager
def gripper_asset_directory():
    for name in ("hand.ply", "finger.ply"):
        path = DSQNET_ROOT / "grasping" / "assets" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing DSQNet gripper asset: {path}")
    previous = Path.cwd()
    try:
        os.chdir(DSQNET_ROOT)
        yield
    finally:
        os.chdir(previous)


def generate_grasp(T_camera_object, parameters, T_camera_table, *, num_samples=200):
    """Return (camera grasp, metadata), or (None, metadata) if none are valid.

    Preserve the upstream width/height filters. Among finite, table-clear grasps,
    prefer the most downward approach (+gripper Z along -table Z), breaking ties
    by the smaller antipodal normal-alignment error.
    """
    T_camera_object = validate_pose(T_camera_object, "T_camera_object")
    T_camera_table = validate_pose(T_camera_table, "T_camera_table")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    T_table_object = np.linalg.inv(T_camera_table) @ T_camera_object
    primitive = DeformableSuperquadric(
        SE3=T_table_object, parameters=parameters, resolution=30, collisionBox=True,
    )
    if (not len(primitive.mesh.triangles)
            or not np.isfinite(np.asarray(primitive.mesh.vertices)).all()):
        raise ValueError("Invalid fitted geometry for grasp generation")
    # Upstream builds its BVH from local vertices before transforming the mesh.
    # Explicitly place that BVH in the same table frame as the sampling geometry.
    primitive.collisionBox.setTransform(
        fcl.Transform(T_table_object[:3, :3].copy(), T_table_object[:3, 3].copy())
    )
    obj = Object([primitive], transform=False, collisionBox=True)
    table = fcl.CollisionObject(
        fcl.Halfspace(np.array([0., 0., 1.]), 0.), fcl.Transform(),
    )
    best = None
    valid_count = 0
    with gripper_asset_directory():
        candidates, scores = generate_collision_free_grasp_pose(obj, num_samples)
        if len(candidates) != len(scores):
            raise ValueError("DSQNet grasp candidates and scores have different lengths")
        for index, ((candidate, width), score) in enumerate(zip(candidates, scores)):
            try:
                pose = validate_pose(candidate, "grasp candidate")
                width, score = float(width), float(score)
            except (TypeError, ValueError):
                continue
            if not np.isfinite([width, score]).all() or not 0 < width < 0.07 or score < 0:
                continue
            opening = min(width + 0.02, 0.08)
            gripper = Gripper(pose, width=opening, collisionBox=True)
            if any(fcl.collide(box, table, fcl.CollisionRequest(), fcl.CollisionResult())
                   for box in gripper.collisionBox):
                continue
            valid_count += 1
            top_down = float(-pose[2, 2])
            rank = (-top_down, score)
            if best is None or rank < best[0]:
                best = (rank, pose.copy(), index, width, opening, score, top_down)

    info = {
        "status": "no_valid_grasp" if best is None else "success",
        "coordinate_frame": "camera",
        "transform_convention": "grasp_pose.txt maps DSQNet hand coordinates to camera coordinates",
        "units": "meters",
        "collision_checks": ["fitted_object", "table_z_le_0"],
        "unchecked": ["other_objects", "robot_reachability", "approach_path"],
        "selection": "largest top_down_score, then smallest antipodal_error",
        "num_samples": num_samples,
        "candidate_count": len(candidates),
        "valid_candidate_count": valid_count,
    }
    if best is None:
        info["reason"] = "No finite, width-valid candidate passed object/height/table checks"
        return None, info
    _, T_table_grasp, index, width, opening, score, top_down = best
    T_camera_grasp = T_camera_table @ T_table_grasp
    info.update({
        "selected_index": index,
        "contact_width_m": width,
        "opening_width_m": opening,
        "antipodal_error": score,
        "top_down_score": top_down,
        "T_object_grasp": (np.linalg.inv(T_camera_object) @ T_camera_grasp).tolist(),
    })
    return T_camera_grasp, info


def generate_and_save_grasp(result, T_camera_table, output_dir, *, num_samples=200):
    """Save a 4x4 grasp_pose.txt; preserve the object fit if grasping fails.

    No placeholder pose is written on failure. A separate grasp_info.json records
    success, lack of candidates, or a numerical/resource error.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        (output_dir / filename).unlink(missing_ok=True)
    try:
        pose, info = generate_grasp(
            result["SE3"], result["parameters"], T_camera_table, num_samples=num_samples,
        )
    except (ValueError, RuntimeError, OSError, ArithmeticError) as exc:
        pose = None
        info = {"status": "error", "reason": str(exc), "coordinate_frame": "camera"}
    if pose is not None:
        np.savetxt(output_dir / "grasp_pose.txt", pose, fmt="%.12g")
    (output_dir / "grasp_info.json").write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    if pose is None:
        print(f"[WARN] {output_dir.name}: grasp not saved: {info['reason']}")
    else:
        print(f"[OK] Grasp pose saved to {output_dir / 'grasp_pose.txt'}")
    return info
