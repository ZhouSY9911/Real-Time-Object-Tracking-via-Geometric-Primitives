from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def save_rgbd_frame(output_dir, bgr, depth_raw, intr, depth_scale):
    """Save the exact aligned frame for FoundationPose; depth0.npy is in meters."""
    output_dir = Path(output_dir)
    depth_m = np.asarray(depth_raw, dtype=np.float32) * depth_scale
    if bgr.shape[:2] != depth_m.shape or depth_m.shape != (intr.height, intr.width):
        raise ValueError("Color, aligned depth and intrinsics must have matching dimensions")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_dir / "frame0.png"), bgr):
        raise OSError(f"Failed to save frame0.png in {output_dir}")
    np.save(output_dir / "depth0.npy", depth_m, allow_pickle=False)
    np.savetxt(output_dir / "cam_K.txt", K, fmt="%.12g")


def color_frame_to_bgr(color_frame):
    """Convert the actual stream format to an owned BGR image."""
    color = np.asanyarray(color_frame.get_data())
    color_format = color_frame.profile.format()
    if color_format == rs.format.bgr8:
        return color.copy()

    conversions = {
        rs.format.rgb8: cv2.COLOR_RGB2BGR,
        rs.format.rgba8: cv2.COLOR_RGBA2BGR,
        rs.format.bgra8: cv2.COLOR_BGRA2BGR,
        rs.format.y8: cv2.COLOR_GRAY2BGR,
        rs.format.yuyv: cv2.COLOR_YUV2BGR_YUY2,
    }
    if color_format not in conversions:
        raise ValueError(f"Unsupported color frame format: {color_format}")
    return cv2.cvtColor(color, conversions[color_format])


def iter_frames_from_bag(bag_path, skip_frames=0):
    """
    Yield aligned RGB-D frames after skipping skip_frames valid framesets.
    Playback is offline so processing does not drop frames. Close the generator
    when stopping early to release the RealSense pipeline.

    Yields:
        bgr: Color image in BGR format.
        depth_u16: Aligned raw depth image.
        intr: Color camera intrinsics.
        depth_scale: Scale factor for converting depth values to meters.
    """
    if isinstance(skip_frames, bool) or not isinstance(skip_frames, int) or skip_frames < 0:
        raise ValueError("skip_frames must be a non-negative integer")
    if not Path(bag_path).is_file():
        raise FileNotFoundError(f"RealSense bag not found: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device_from_file(
        str(bag_path),
        repeat_playback=False,
    )
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)

    try:
        # Offline playback should wait for processing rather than drop frames.
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align = rs.align(rs.stream.color)

        valid_frames = 0
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError as exc:
                if valid_frames > skip_frames and playback.current_status() == rs.playback_status.stopped:
                    return
                raise RuntimeError(
                    f"Could not read valid frameset {skip_frames + 1} from {bag_path}; "
                    f"received {valid_frames} valid framesets. "
                    "Playback ended or frame acquisition failed/timed out."
                ) from exc

            if not frames.get_color_frame() or not frames.get_depth_frame():
                continue
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            valid_frames += 1
            if valid_frames <= skip_frames:
                continue
            bgr = color_frame_to_bgr(color_frame)
            depth_u16 = np.asanyarray(depth_frame.get_data()).copy()
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            yield bgr, depth_u16, intr, depth_scale

    finally:
        pipeline.stop()


def read_first_frame_from_bag(bag_path, skip_frames=0):
    """Read one aligned frame; see iter_frames_from_bag for tuple contents."""
    frames = iter_frames_from_bag(bag_path, skip_frames=skip_frames)
    try:
        return next(frames)
    finally:
        frames.close()
