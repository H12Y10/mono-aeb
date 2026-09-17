"""逐视频离线自标定 CLI：GPS ego 速度 + 免标定尺度 TTC → 反解测距尺度 f·H。

用法：
    python calibrate_video.py --stem 0571873b-faf718b2
    python calibrate_video.py --video /path/to/samples-1k/videos/xxxx.mov --max-frames 200

跑一遍视频，对「静止目标」用 f·H = v_ego·τ·(y_bottom − horizon) 反解，
取中位数作为该视频的测距尺度，并与默认占位 1100×1.5=1650 对比。
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from aeb.calibration import estimate_ground_plane
from aeb.config import CameraConfig
from aeb.detectors import YOLOv8Detector
from aeb.ego_speed import ego_speed_at, load_ego_speed, load_ego_speed_series

VIDEO_DIR = Path(os.environ.get("BDD100K_VIDEO_DIR", "."))
WEIGHTS = "yolov8n.pt"


def main():
    ap = argparse.ArgumentParser(description="逐视频离线自标定 f·H")
    ap.add_argument("--video", type=str, default=None, help="视频完整路径")
    ap.add_argument("--stem", type=str, default=None, help="samples-1k/videos 下的片段 stem")
    ap.add_argument("--weights", type=str, default=WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.stem:
        path = VIDEO_DIR / f"{args.stem}.mov"
    elif args.video:
        path = Path(args.video)
    else:
        raise SystemExit("需要 --stem 或 --video")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频 {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

    camera = CameraConfig()  # horizon=360 默认；focal_px/cam_height_m 是待标定的占位
    detector = YOLOv8Detector(args.weights, conf=args.conf, device="0")

    ts, speeds = load_ego_speed_series(path)
    if ts is not None and len(ts) > 0:
        ego_speed_fn = lambda i: ego_speed_at(ts, speeds, i / fps)
    else:
        v_median = load_ego_speed(path, agg="median") or 12.0
        ego_speed_fn = lambda i, v=v_median: v

    def frames():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1

    fh, horizon, n_samples, n_tracks = estimate_ground_plane(
        detector, frames(), camera, ego_speed_fn,
        fps=fps, max_frames=args.max_frames,
    )
    cap.release()

    print(f"视频: {path.stem}")
    print(f"采样: {n_samples} 个 (跨 {n_tracks} 条轨迹)")
    if fh is None:
        print("标定失败：静止目标样本不足（可增大 --max-frames 或换片段）")
        return
    print(f"标定前 f·H = {camera.focal_height:.1f}（占位 1100×1.5）  "
          f"horizon = {camera.horizon_y:.0f}（默认）")
    print(f"标定后 f·H = {fh:.1f}  horizon = {horizon:.1f}")
    camera.calibrate_scale(fh)
    camera.calibrate_horizon(horizon)
    print(f"  → focal_px = {camera.focal_px:.1f}"
          f"（cam_height_m 保持 {camera.cam_height_m} m）")


if __name__ == "__main__":
    main()
