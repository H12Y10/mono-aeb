"""Replace the detector backend: a 30-line, zero-dependency example.

Every backend is just a callable that turns one BGR frame into a list of
``Detection`` objects. Nothing in the tracking / ranging / TTC / decision chain
knows which model produced them, which is what "detector-agnostic" means in
practice.

This file is deliberately model-free: it thresholds the frame for bright blobs and
returns the largest one. That keeps the example runnable on any machine (no
weights, no torch, no ultralytics) while still exercising the real pipeline.

Run it:

    python examples/custom_detector.py            # synthetic approaching target
    python examples/custom_detector.py --video v.mp4   # your own footage

See docs/detector_adapter.md for the contract in full.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeb.config import AEBConfig
from aeb.detectors import BaseDetector
from aeb.pipeline import AEBPipeline
from aeb.types import Detection


class BrightBlobDetector(BaseDetector):
    """Largest bright region as one car detection. A stand-in for a real model."""

    def __init__(self, threshold: int = 180, min_area: int = 400, class_id: int = 2):
        self.threshold = threshold
        self.min_area = min_area
        self.class_id = class_id

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue
            if best is None or w * h > best[2] * best[3]:
                best = (x, y, w, h)
        if best is None:
            return []  # 允许空列表：这一帧就是没有目标
        x, y, w, h = best
        return [Detection(float(x), float(y), float(x + w), float(y + h), 0.99, self.class_id)]


def synthetic_frames(n: int = 90, fps: float = 30.0):
    """一个亮方块从远处逼近：面积随时序增大，驱动尺度 TTC。"""
    h, w = 720, 1280
    for i in range(n):
        frame = np.full((h, w, 3), 40, np.uint8)
        t = i / fps
        # 框高按 h = f·H / D 缩小，D 从 45 m 匀速减到 8 m
        d = max(45.0 - 10.0 * t, 8.0)
        box_h = int(1650.0 / d)
        box_w = int(box_h * 1.2)
        cx, cy = w // 2, int(360 + 1650.0 / d)
        cv2.rectangle(
            frame,
            (cx - box_w // 2, cy - box_h),
            (cx + box_w // 2, cy),
            (255, 255, 255),
            -1,
        )
        yield frame


def main() -> int:
    ap = argparse.ArgumentParser(description="custom detector adapter example")
    ap.add_argument("--video", default=None, help="可选：用真实视频代替合成帧")
    ap.add_argument("--fps", type=float, default=30.0, help="图片/合成帧没有固有帧率时必须给")
    ap.add_argument("--frames", type=int, default=150, help="合成模式下的帧数")
    ap.add_argument("--ego-speed", type=float, default=12.0)
    args = ap.parse_args()

    cfg = AEBConfig(fps=args.fps)
    cfg.ego_speed_mps = args.ego_speed
    pipe = AEBPipeline(cfg, BrightBlobDetector())

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"cannot open {args.video}")
            return 1

        def _read():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()

        frames = _read()
    else:
        frames = synthetic_frames(args.frames, args.fps)

    last = None
    for i, frame in enumerate(frames):
        result = pipe.process(frame, i)
        level = result.global_level.name
        if level != last:
            print(f"frame {i:4d}: {level}")
            last = level
    print("done — the chain ran on a detector that is not YOLO and not D-FINE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
