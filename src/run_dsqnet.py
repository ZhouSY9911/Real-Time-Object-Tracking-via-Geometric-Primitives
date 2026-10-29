"""Fit one point cloud and save results; reusable by the multi-object pipeline."""

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from dsqnet_inference import DSQNetInference
from get_grasp import OUTPUT_FILES as GRASP_OUTPUT_FILES, generate_and_save_grasp
from functions.primitives import DeformableSuperquadric


OUTPUT_FILES = (
    "SE3.txt", "parameters.json", "fitted_primitive.ply",
    "fitted_primitive.obj", "dsqnet_visualization.png",
)


def clear_dsqnet_outputs(output_dir):
    """Remove only this stage's old artifacts so a failed rerun is unambiguous."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES + GRASP_OUTPUT_FILES:
        (output_dir / filename).unlink(missing_ok=True)


def save_fit_visualization(pcd, mesh, output_path):
    """Save four orthographic views without requiring an OpenGL window.

    Blue points overlay the orange fitted surface. All panels share the same
    scale and center in the input coordinate frame. Mesh simplification and
    point subsampling affect only this preview, never the saved geometry.
    """
    points = np.asarray(pcd.points)
    points = points[np.isfinite(points).all(axis=1)]
    vertices = np.asarray(mesh.vertices)
    if not len(points) or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Cannot visualize empty or non-finite geometry")
    lower = np.minimum(points.min(axis=0), vertices.min(axis=0))
    upper = np.maximum(points.max(axis=0), vertices.max(axis=0))
    center = (lower + upper) / 2
    extent = np.linalg.norm(upper - lower)
    if extent <= 0:
        raise ValueError("Cannot visualize geometry with zero extent")

    preview = mesh.simplify_quadric_decimation(4000)
    vertices = np.asarray(preview.vertices) - center
    triangles = np.asarray(preview.triangles)
    if len(points) > 6000:
        points = points[np.linspace(0, len(points) - 1, 6000, dtype=int)]
    points = points - center

    size = 640
    scale = size * 0.78 / extent
    canvas = np.full((size * 2 + 64, size * 2, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, "DSQNet fit | blue: input cloud | orange: fitted surface",
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2, cv2.LINE_AA)
    views = (
        ("Isometric", np.array([1., -1., 0.]) / np.sqrt(2),
         np.array([1., 1., 2.]) / np.sqrt(6)),
        ("XY view", np.array([1., 0., 0.]), np.array([0., 1., 0.])),
        ("XZ view", np.array([1., 0., 0.]), np.array([0., 0., 1.])),
        ("YZ view", np.array([0., 1., 0.]), np.array([0., 0., 1.])),
    )
    for index, (title, right, up) in enumerate(views):
        basis = np.column_stack((right, up, np.cross(right, up)))
        projected = vertices @ basis
        pixels = np.rint(projected[:, :2] * [scale, -scale] + size / 2).astype(np.int32)
        panel = np.full((size, size, 3), 250, dtype=np.uint8)
        faces = projected[triangles]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        brightness = 0.55 + 0.45 * np.abs(normals[:, 2]) / np.maximum(lengths, 1e-15)
        # Draw the farthest triangles first; positive view Z faces the viewer.
        for face_id in np.argsort(faces[:, :, 2].mean(axis=1)):
            color = tuple(int(c * brightness[face_id]) for c in (26, 110, 245))
            cv2.fillConvexPoly(panel, pixels[triangles[face_id]], color, cv2.LINE_AA)
        point_pixels = np.rint((points @ basis)[:, :2] * [scale, -scale] + size / 2).astype(np.int32)
        for xy in point_pixels:
            cv2.circle(panel, tuple(xy), 1, (225, 125, 30), -1, cv2.LINE_AA)
        cv2.putText(panel, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (40, 40, 40), 2, cv2.LINE_AA)
        origin = np.array([70, size - 70])
        for axis, color in enumerate(((50, 50, 210), (50, 150, 50), (210, 80, 50))):
            endpoint = origin + np.rint(basis[axis, :2] * [42, -42]).astype(int)
            if np.linalg.norm(endpoint - origin) > 5:
                cv2.arrowedLine(panel, tuple(origin), tuple(endpoint), color, 2, cv2.LINE_AA)
                cv2.putText(panel, "XYZ"[axis], tuple(endpoint + [3, -3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        row, col = divmod(index, 2)
        canvas[64 + row * size:64 + (row + 1) * size, col * size:(col + 1) * size] = panel
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Failed to save visualization: {output_path}")


def predict_and_save(
    model, ply_path, output_dir=None, *, show=False,
    T_camera_table=None, grasp_num_samples=200,
):
    """Predict once, then save pose, parameters, mesh and PNG beside the cloud.

    SE3 maps the fitted primitive's local frame into the input cloud frame;
    translations and a1/a2/a3 retain the input cloud's units (meters in main.py).
    PLY is in the input cloud frame for visualization. OBJ is in the primitive's
    local frame for FoundationPose: applying SE3 to OBJ reproduces the PLY mesh.
    When T_camera_table is supplied, generate grasp_pose.txt from the same fit.
    Its hand-to-camera transform uses meters, just like SE3.txt. Without a table
    transform, retain the fit-only behavior used by the standalone baselines.
    """
    ply_path = Path(ply_path)
    output_dir = Path(output_dir) if output_dir is not None else ply_path.parent
    clear_dsqnet_outputs(output_dir)
    result = model.predict(ply_path)
    pose = np.asarray(result["SE3"])
    parameters = {key: float(value) for key, value in result["parameters"].items()}
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("DSQNet returned an invalid pose")
    # Save numerical results first so they remain available if rendering fails.
    np.savetxt(output_dir / "SE3.txt", pose, fmt="%.12g")
    (output_dir / "parameters.json").write_text(
        json.dumps(parameters, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    primitive = DeformableSuperquadric(
        SE3=pose, parameters=parameters, resolution=30, color=[1.0, 0.3, 0.1],
    )
    mesh = primitive.mesh
    if (not len(mesh.vertices) or not len(mesh.triangles)
            or not np.isfinite(np.asarray(mesh.vertices)).all()):
        raise ValueError("DSQNet parameters produced an empty or invalid mesh")
    mesh_path = output_dir / "fitted_primitive.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise OSError(f"Failed to save fitted mesh: {mesh_path}")

    # Export a reusable object model without the first frame's camera pose.
    model_mesh = copy.deepcopy(mesh)
    model_mesh.transform(np.linalg.inv(pose))
    model_mesh.compute_vertex_normals()
    model_path = output_dir / "fitted_primitive.obj"
    if not o3d.io.write_triangle_mesh(str(model_path), model_mesh, write_triangle_uvs=False):
        raise OSError(f"Failed to save FoundationPose mesh: {model_path}")

    if T_camera_table is not None:
        generate_and_save_grasp(
            result, T_camera_table, output_dir, num_samples=grasp_num_samples,
        )

    pcd = o3d.io.read_point_cloud(str(ply_path))
    save_fit_visualization(pcd, mesh, output_dir / "dsqnet_visualization.png")

    if show:
        pcd.paint_uniform_color([0.2, 0.6, 1.0])
        o3d.visualization.draw_geometries([pcd, mesh], window_name="DSQNet primitive fitting")
    return result

