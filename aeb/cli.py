"""mono-aeb command-line interface, shipped inside the package.

Subcommands:
    mono-aeb demo    zero-data synthetic demo: a target closing at constant speed
                     drives the full tracking -> ranging -> TTC -> decision chain,
                     printing the risk level per frame
    mono-aeb check   dependency self-check only (no demo)
    mono-aeb run     real video / image directory through the full chain
    mono-aeb calib   per-video offline self-calibration of f·H and horizon_y

`pip install mono-aeb` exposes them as the `mono-aeb` console script; the same
commands also work as `python -m aeb ...`. examples/quickstart.py,
examples/demo_aeb.py and examples/calibrate_video.py are thin wrappers that
translate the old flag style into these subcommands -- the implementation lives
here so that it ships in the wheel.
"""

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np

from aeb.calibration import estimate_ground_plane
from aeb.config import AEBConfig, CameraConfig
from aeb.detectors import BaseDetector
from aeb.ego_speed import (
    ego_speed_at,
    load_ego_speed,
    load_ego_speed_or,
    load_ego_speed_series,
)
from aeb.pipeline import AEBPipeline
from aeb.types import AEB_CLASS_NAMES, Detection, RiskLevel

LEVEL_COLORS = {
    RiskLevel.NORMAL: (80, 220, 80),
    RiskLevel.ATTENTION: (0, 220, 220),
    RiskLevel.FCW: (0, 140, 255),
    RiskLevel.AEB_WARNING: (0, 0, 255),
}
LEVEL_NAMES = {
    RiskLevel.NORMAL: "NORMAL",
    RiskLevel.ATTENTION: "ATTENTION",
    RiskLevel.FCW: "FCW",
    RiskLevel.AEB_WARNING: "AEB",
}


# ---------------------------------------------------------------------------
# demo / check（原 examples/quickstart.py 的实现整体搬入）
# ---------------------------------------------------------------------------

# Core dependencies (enough for the demo): needed by tracker / ranging / TTC; no weights.
CORE_DEPS = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "lap": "lap (install lapx instead if no lap wheel exists for your platform)",
}


def missing_deps(need_detector: bool = False) -> list:
    """返回缺失依赖的人话列表；need_detector=True 时额外检查 ultralytics。"""
    deps = dict(CORE_DEPS)
    if need_detector:
        deps["ultralytics"] = "ultralytics"
    missing = []
    for mod, pkg in deps.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(f"{mod} (pip install {pkg})")
    return missing


class SyntheticDetector(BaseDetector):
    """demo 专用：生成一个匀速逼近的目标，驱动完整决策链。

    目标以 v_close 从 d0 匀速逼近，底边 / 框高按地面平面模型反解
    （y_bottom = horizon + f·H/d，框高 h = f·H_real/d，与相机同高时
    h = y_bottom − horizon）。这样 DistanceTTC 与 ScaleTTC 对 (t, D) 与
    (t, 1/h) 的 Theil-Sen 拟合都能恢复 TTC = D/v_close，状态自然升级。
    """

    def __init__(
        self,
        fps: float = 30.0,
        v_close: float = 10.0,
        d0: float = 45.0,
        focal_height: float = 1650.0,
        horizon: float = 360.0,
        cx: float = 640.0,
        aspect: float = 1.2,
        d_min: float = 5.0,
    ):
        self.fps = fps
        self.v_close = v_close
        self.d0 = d0
        self.fh = focal_height
        self.horizon = horizon
        self.cx = cx
        self.aspect = aspect
        self.d_min = d_min
        self._frame = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        i = self._frame
        self._frame += 1
        t = i / self.fps
        d = max(self.d0 - self.v_close * t, self.d_min)

        y_bottom = self.horizon + self.fh / d
        h = self.fh / d  # 与相机同高目标：框高 = 底边到 horizon 的距离
        w = h * self.aspect
        x1 = self.cx - w / 2
        x2 = self.cx + w / 2
        y1 = y_bottom - h
        y2 = y_bottom
        return [
            Detection(float(x1), float(y1), float(x2), float(y2), 0.9, 2)
        ]  # score=0.9, class=2 (car)


def make_frame() -> np.ndarray:
    """生成 demo 用的深色背景帧（天空 + 路面 + horizon 线）。"""
    frame = np.full((720, 1280, 3), (40, 40, 40), np.uint8)
    frame[360:] = (70, 70, 70)  # 路面略亮
    cv2.line(frame, (0, 360), (1280, 360), (180, 180, 180), 1)
    return frame


def draw_synthetic(frame: np.ndarray, result, pipeline: AEBPipeline) -> np.ndarray:
    """画 ROI + 目标框 + 距离/TTC + 全局风险横幅（合成 demo 用）。"""
    frame = pipeline.roi.draw(frame, color=(100, 100, 100))
    risk_by_id = {r.track_id: r for r in result.risks}
    for t in result.tracks:
        r = risk_by_id.get(t.track_id)
        level = r.level if r is not None else RiskLevel.NORMAL
        color = LEVEL_COLORS[level]
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        label = f"#{t.track_id} car"
        if r is not None:
            d = r.distance
            dist_s = f"{d:.1f}m" if np.isfinite(d) else "inf"
            ttc_s = f"{r.ttc_fused:.1f}s" if np.isfinite(r.ttc_fused) else "inf"
            label += f"  d={dist_s} ttc={ttc_s}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    g = result.global_level
    banner = np.full((44, frame.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(
        banner,
        f"GLOBAL: {LEVEL_NAMES[g]}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        LEVEL_COLORS[g],
        2,
        cv2.LINE_AA,
    )
    return np.vstack([banner, frame])


def run_demo(args) -> int:
    missing = missing_deps(need_detector=False)
    if missing:
        print("[quickstart] missing dependencies, install them first:")
        for m in missing:
            print(f"  - {m}")
        return 1

    cfg = AEBConfig(fps=args.fps)
    det = SyntheticDetector(fps=cfg.fps, v_close=args.v_close)
    pipe = AEBPipeline(cfg, det)

    print(
        f"[quickstart] ego speed = {cfg.ego_speed_mps:.0f} m/s, "
        f"AEB threshold = {cfg.risk.ladder(cfg.ego_speed_mps)[2]:.2f}s"
    )
    print(
        f"[quickstart] target closing at a constant {args.v_close:.0f} m/s; "
        "printing the risk state per frame:"
    )

    writer = None
    prev_level = None
    for i in range(args.frames):
        frame = make_frame()
        result = pipe.process(frame, i)
        out = draw_synthetic(frame, result, pipe)

        level = result.global_level
        if level != prev_level:
            r = result.risks[0] if result.risks else None
            d = r.distance if r is not None else float("inf")
            ttc = r.ttc_fused if r is not None else float("inf")
            print(f"  frame {i:3d}: {LEVEL_NAMES[level]:<9s}  d={d:5.1f}m  ttc={ttc:5.2f}s")
            prev_level = level

        if writer is None and args.out:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        if writer is not None:
            writer.write(out)
        if args.show:
            cv2.imshow("mono-aeb demo", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    if args.show:
        # 仅在实际开过窗口时销毁。无 GUI 的 OpenCV 构建（opencv-python-headless，
        # 常见于服务器 / Docker / CI）里 destroyAllWindows() 不是 no-op，而是抛
        # cv2.error“The function is not implemented”，因此无头环境必须跳过。
        cv2.destroyAllWindows()
    print(
        f"[quickstart] done, {args.frames} frames"
        + (f", video written to {args.out}" if args.out else "")
    )
    return 0


def run_check(args=None) -> int:
    core = missing_deps(need_detector=False)
    det = missing_deps(need_detector=True)
    print("[quickstart] core dependencies (needed by --demo):")
    print("  " + ("all set" if not core else "\n  ".join(core)))
    print("[quickstart] detector dependencies (needed by examples/demo_aeb.py on real video):")
    print("  " + ("all set" if not det else "\n  ".join(det)))
    return 0 if not core else 1


# ---------------------------------------------------------------------------
# run（原 examples/demo_aeb.py 的实现整体搬入）
# ---------------------------------------------------------------------------


def iter_source(source: str):
    """图片目录按文件名排序逐帧读；否则按视频读。"""
    p = Path(source)
    if p.is_dir():
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += sorted(p.glob(ext))
        if not paths:
            raise SystemExit(f"no images in directory: {source}")
        for fp in paths:
            img = cv2.imread(str(fp))
            if img is not None:
                yield img
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {source}")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield img
    cap.release()


def draw_video(frame: np.ndarray, result, pipeline: AEBPipeline) -> np.ndarray:
    """画 ROI + 目标框 + 距离/TTC + 全局风险横幅（真实视频用，带类别名）。"""
    frame = pipeline.roi.draw(frame)
    risk_by_id = {r.track_id: r for r in result.risks}

    for t in result.tracks:
        r = risk_by_id.get(t.track_id)
        level = r.level if r is not None else RiskLevel.NORMAL
        color = LEVEL_COLORS[level]
        x1, y1, x2, y2 = (int(v) for v in t.bbox)

        label = f"#{t.track_id} {AEB_CLASS_NAMES[t.class_id]}"
        if r is not None:
            d = r.distance
            dist_s = f"{d:.1f}m" if np.isfinite(d) else "inf"
            ttc_s = f"{r.ttc_fused:.1f}s" if np.isfinite(r.ttc_fused) else "inf"
            label += f" d={dist_s} ttc={ttc_s}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # 全局风险横幅
    g = result.global_level
    banner = np.full((44, frame.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(
        banner,
        f"GLOBAL: {LEVEL_NAMES[g]}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        LEVEL_COLORS[g],
        2,
        cv2.LINE_AA,
    )
    return np.vstack([banner, frame])


def run_video(args) -> int:
    # 检测器后端在此处才 import：ultralytics / torch 只有真视频路径才需要，
    # 保证 demo / check 在只装核心依赖的环境下不触发重依赖（aeb.detectors 懒加载）。
    from aeb.detectors import COCO_TO_AEB, DFineDetector, YOLOv8Detector

    cfg = AEBConfig(fps=args.fps)
    if args.ego_speed is not None:
        cfg.ego_speed_mps = args.ego_speed
    elif not Path(args.source).is_dir():
        v = load_ego_speed(args.source)
        if v is not None:
            cfg.ego_speed_mps = v
    if args.detector == "dfine":
        det_kwargs = {"conf": args.conf, "device": args.device}
        if args.dfine_weights:
            det_kwargs["weights"] = args.dfine_weights
        if args.dfine_config:
            det_kwargs["config"] = args.dfine_config
        det = DFineDetector(**det_kwargs)
    else:
        class_map = COCO_TO_AEB if args.coco else None
        det = YOLOv8Detector(args.weights, conf=args.conf, device=args.device, class_map=class_map)
    pipe = AEBPipeline(cfg, det)
    print(
        f"[demo] ego speed = {cfg.ego_speed_mps:.1f} m/s "
        f"({cfg.ego_speed_mps * 3.6:.0f} km/h)  "
        f"AEB threshold = {cfg.risk.ladder(cfg.ego_speed_mps)[2]:.2f}s"
    )

    writer = None
    frame_idx = 0
    for frame in iter_source(args.source):
        result = pipe.process(frame, frame_idx)
        out = draw_video(frame, result, pipe)

        if writer is None and args.out:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

        if writer is not None:
            writer.write(out)
        if args.show:
            cv2.imshow("AEB", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[demo] frame {frame_idx}  global={result.global_level.name}")

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[demo] done, {frame_idx} frames")
    return 0


# ---------------------------------------------------------------------------
# calib（原 examples/calibrate_video.py 的实现整体搬入）
# ---------------------------------------------------------------------------


def run_calib(args) -> int:
    from aeb.detectors import YOLOv8Detector

    # --stem 模式：视频目录经 BDD100K_VIDEO_DIR 环境变量定位（读调用时的值）
    video_dir = Path(os.environ.get("BDD100K_VIDEO_DIR", "."))

    if args.stem:
        path = video_dir / f"{args.stem}.mov"
    elif args.video:
        path = Path(args.video)
    else:
        raise SystemExit("either --stem or --video is required")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

    camera = CameraConfig()  # horizon=360 默认；focal_px/cam_height_m 是待标定的占位
    detector = YOLOv8Detector(args.weights, conf=args.conf, device="0")

    ts, speeds = load_ego_speed_series(path)
    if ts is not None and len(ts) > 0:

        def ego_speed_fn(i):
            return ego_speed_at(ts, speeds, i / fps)
    else:
        v_median = load_ego_speed_or(path, default=12.0, agg="median")

        def ego_speed_fn(i, v=v_median):
            return v

    def frames():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1

    fh, horizon, n_samples, n_tracks = estimate_ground_plane(
        detector,
        frames(),
        camera,
        ego_speed_fn,
        fps=fps,
        max_frames=args.max_frames,
    )
    cap.release()

    print(f"video: {path.stem}")
    print(f"samples: {n_samples} (across {n_tracks} tracks)")
    if fh is None:
        print(
            "calibration failed: not enough static-target samples "
            "(increase --max-frames or try another clip)"
        )
        return 0
    print(
        f"before calibration: f·H = {camera.focal_height:.1f} (placeholder 1100x1.5)  "
        f"horizon = {camera.horizon_y:.0f} (default)"
    )
    print(f"after calibration:  f·H = {fh:.1f}  horizon = {horizon:.1f}")
    camera.calibrate_scale(fh)
    camera.calibrate_horizon(horizon)
    print(f"  → focal_px = {camera.focal_px:.1f} (cam_height_m kept at {camera.cam_height_m} m)")
    return 0


# ---------------------------------------------------------------------------
# CLI 接线
# ---------------------------------------------------------------------------


def package_version() -> str:
    """已安装发行版版本号；纯源码 checkout（未安装）时退化为 unknown。"""
    try:
        return importlib.metadata.version("mono-aeb")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mono-aeb",
        description="mono-aeb: detector-agnostic monocular AEB/FCW decision chain",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the installed mono-aeb version and exit",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_demo = sub.add_parser("demo", help="zero-data synthetic demo (no video / no weights)")
    p_demo.add_argument(
        "--frames", type=int, default=120, help="number of demo frames (default 120)"
    )
    p_demo.add_argument("--fps", type=float, default=30.0)
    p_demo.add_argument(
        "--v-close", type=float, default=10.0, help="closing speed of the synthetic target (m/s)"
    )
    p_demo.add_argument("--out", default=None, help="output video path (optional)")
    p_demo.add_argument("--show", action="store_true", help="show a live window")
    p_demo.set_defaults(func=run_demo)

    p_check = sub.add_parser("check", help="dependency self-check only, no demo")
    p_check.set_defaults(func=run_check)

    p_run = sub.add_parser("run", help="run the chain on a real video / image directory")
    p_run.add_argument("--source", required=True, help="video file or image directory")
    p_run.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="detection weights; the default yolov8n.pt is COCO-pretrained (use with --coco); "
        "pass AEB-finetuned weights otherwise",
    )
    p_run.add_argument("--conf", type=float, default=0.25)
    p_run.add_argument("--device", default="0")
    p_run.add_argument("--fps", type=float, default=30.0)
    p_run.add_argument(
        "--coco",
        action="store_true",
        help="weights are COCO-pretrained (e.g. yolov8n.pt): map COCO classes to the AEB ones",
    )
    p_run.add_argument(
        "--detector",
        choices=["yolo", "dfine"],
        default="yolo",
        help="detector backend: yolo (default) or dfine",
    )
    p_run.add_argument(
        "--dfine-weights",
        default=None,
        help="D-FINE weights path; defaults to the MONO_AEB_DFINE_WEIGHTS environment variable",
    )
    p_run.add_argument(
        "--dfine-config",
        default=None,
        help="D-FINE config yaml; defaults to <DFINE_ROOT>/configs/dfine/dfine_hgnetv2_m_aeb.yml",
    )
    p_run.add_argument("--out", default=None, help="output video path (optional)")
    p_run.add_argument("--show", action="store_true", help="show a live window")
    p_run.add_argument(
        "--ego-speed",
        type=float,
        default=None,
        help="ego speed in m/s; by default the GPS speed is read from samples-1k/info/*.json",
    )
    p_run.set_defaults(func=run_video)

    p_calib = sub.add_parser("calib", help="per-video offline self-calibration of f·H")
    p_calib.add_argument("--video", type=str, default=None, help="full path to the video")
    p_calib.add_argument("--stem", type=str, default=None, help="clip stem under samples-1k/videos")
    p_calib.add_argument("--weights", type=str, default="yolov8n.pt")
    p_calib.add_argument("--conf", type=float, default=0.25)
    p_calib.add_argument("--max-frames", type=int, default=None)
    p_calib.add_argument("--fps", type=float, default=30.0)
    p_calib.set_defaults(func=run_calib)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"mono-aeb {package_version()}")
        return 0
    if args.command is None:
        # 不带任何参数：打印帮助并友好退出（装完敲一下就能看到用法）
        parser.print_help()
        return 0
    return args.func(args)
