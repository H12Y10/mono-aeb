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

from __future__ import annotations

import argparse
import contextlib
import glob as _glob
import importlib.metadata
import importlib.util
import json
import math
import os
import time
from pathlib import Path

from aeb.calib_io import apply_calibration, load_calibration, save_calibration
from aeb.detectors import BaseDetector  # 轻量：只导入 base，重后端仍懒加载
from aeb.types import AEB_CLASS_NAMES, Detection, RiskLevel

# 核心依赖改成「尽力导入」：`mono-aeb check` 的职责就是在缺依赖时给人话提示，
# 不能在 import 阶段抛 traceback（此前顶层 import cv2 / numpy / pipeline，
# 缺任何一个都会让 check 直接崩）。注解已由 `from __future__ import annotations`
# 变成字符串，np.ndarray 不再要求 numpy 在场。
try:  # pragma: no cover - 取决于运行环境
    import cv2
    import numpy as np

    from aeb.calibration import estimate_ground_plane
    from aeb.config import AEBConfig, CameraConfig
    from aeb.ego_speed import (
        ego_speed_at,
        load_ego_speed,
        load_ego_speed_or,
        load_ego_speed_series,
    )
    from aeb.imageio import imread_unicode
    from aeb.pipeline import AEBPipeline
except ImportError as _exc:
    _CORE_IMPORT_ERROR: Exception | None = _exc
else:
    _CORE_IMPORT_ERROR = None


def core_import_error() -> str | None:
    """核心依赖导入失败时返回可操作提示，正常时返回 None。"""
    if _CORE_IMPORT_ERROR is None:
        return None
    return (
        "core dependencies are not importable in this environment: "
        f"{_CORE_IMPORT_ERROR}. Install them with `pip install mono-aeb` "
        "(or `pip install numpy opencv-python scipy lap`), then re-run."
    )


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
    err = core_import_error()
    if err:
        print(f"[quickstart] {err}")
        return 1
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
    err = core_import_error()
    core = missing_deps(need_detector=False)
    det = missing_deps(need_detector=True)
    print("[quickstart] core dependencies (needed by --demo):")
    print("  " + ("all set" if not core else "\n  ".join(core)))
    print("[quickstart] detector dependencies (needed by examples/demo_aeb.py on real video):")
    print("  " + ("all set" if not det else "\n  ".join(det)))
    if err:
        # 依赖表能查出来但 import 仍失败，说明是装坏/版本冲突而不是单纯缺包，
        # 这时把原始 ImportError 一并打出来，否则用户只能看到 "all set" 却跑不起来。
        print("[quickstart] core import failed:")
        print(f"  {err}")
        return 1
    return 0 if not core else 1


# ---------------------------------------------------------------------------
# run（原 examples/demo_aeb.py 的实现整体搬入）
# ---------------------------------------------------------------------------


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _sorted_images(directory: Path) -> list[Path]:
    """目录下按扩展名收集图片并按文件名排序（jpg / jpeg / png / bmp 混排也稳定）。"""
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths += sorted(directory.glob(f"*{ext}"))
    return [p for p in paths if p.is_file()]


def _glob_images(pattern: str) -> list[Path]:
    """按 glob 收集图片（支持 ** 递归），结果按路径排序。"""
    hits = [Path(h) for h in _glob.glob(pattern, recursive=True)]
    return sorted(p for p in hits if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _list_images(list_file: Path) -> list[Path]:
    """文本清单：一行一个路径或 glob；空行与 # 开头忽略；相对路径相对清单所在目录。"""
    base = list_file.parent
    out: list[Path] = []
    with open(list_file, encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cand = Path(line)
            if not cand.is_absolute():
                cand = base / cand
            if any(ch in line for ch in "*?["):
                out += _glob_images(str(cand))
            elif cand.is_dir():
                out += _sorted_images(cand)
            elif cand.is_file() and cand.suffix.lower() in IMAGE_EXTS:
                out.append(cand)
            # 其余（视频、无效路径）静默跳过：清单里可以混放，视频由 --source 单独给
    # 去重保序
    seen, uniq = set(), []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def resolve_images(source: str) -> list[Path] | None:
    """把 --source 解析成图片序列；返回 None 表示按视频处理。

    支持四种写法：图片目录、glob（含 **）、文本清单（.txt，一行一个路径或 glob）、单张图片。
    """
    p = Path(source)
    if p.is_dir():
        return _sorted_images(p)
    if any(ch in source for ch in "*?["):
        return _glob_images(source)
    if p.is_file() and p.suffix.lower() == ".txt":
        return _list_images(p)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    return None


def iter_source(source: str):
    """图片序列（目录 / glob / 清单 / 单图）按解析顺序逐帧读；否则按视频读。"""
    paths = resolve_images(source)
    if paths is not None:
        if not paths:
            raise SystemExit(f"no images resolved from: {source}")
        for fp in paths:
            img = imread_unicode(fp)  # 走 unicode 封装：cv2.imread 在中文路径下静默返回 None
            if img is not None:
                yield img
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {source}")
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
    finally:
        # 消费者中途 break / 抛异常时也要释放解码器：否则文件句柄会挂到进程结束。
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


def _pick_fps(fps_arg, probed, *, is_dir: bool):
    """帧率优先级：显式 --fps > 源视频 CAP_PROP_FPS(>0) > 30。返回 (fps, 来源)。"""
    if fps_arg is not None:
        if fps_arg <= 0:
            raise SystemExit("[run] --fps must be positive")
        return float(fps_arg), "command line (--fps)"
    if is_dir:
        raise SystemExit(
            "[run] an image directory has no intrinsic frame rate; pass --fps explicitly"
        )
    if probed and probed > 0:
        return float(probed), "source video (CAP_PROP_FPS)"
    return 30.0, "default 30 (source reports no frame rate)"


def _probe_source(source: str):
    """返回 (是否目录, 源视频帧率或 None, (w, h) 或 None)；打不开视频直接退出。

    图片目录没有帧率，但**有分辨率**——标定文件按像素存 f·H，必须拿到尺寸才能
    校验，否则换分辨率的配置会被静默套用。
    """
    p = Path(source)
    paths = resolve_images(source)
    if paths is not None:
        size = None
        for fp in paths:
            img = imread_unicode(fp)
            if img is not None:
                size = (int(img.shape[1]), int(img.shape[0]))
                break
        return True, None, size
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return False, fps, size


def _finite_min(values) -> float | None:
    """有限值集合的最小值；没有有限值时返回 None（写进 JSON 是 null）。"""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(vals) if vals else None


def frame_record(result) -> dict:
    """单帧结构化记录：字段名与智能驾驶演示平台的 CSV 前 11 列保持一致。

    只做聚合（计数 / 取最小），**不重算链路里的任何数值**；TTC / 距离 / 速度
    全部直接取自 `FrameResult.risks`。`inf` 与 `nan` 一律写 `null`——JSON 没有
    Infinity，下游解析器也不该被迫处理它。
    """
    risks = result.risks
    return {
        "frame": int(result.frame_idx),
        "global_level": LEVEL_NAMES[result.global_level],
        "n_tracks": len(result.tracks),
        "n_in_path": sum(1 for r in risks if getattr(r, "in_path", False)),
        "min_ttc_fused": _finite_min(r.ttc_fused for r in risks),
        "min_ttc_distance": _finite_min(r.ttc_distance for r in risks),
        "min_ttc_scale": _finite_min(r.ttc_scale for r in risks),
        "min_distance_m": _finite_min(r.distance for r in risks),
        "min_closing_speed": _finite_min(getattr(r, "closing_speed", None) for r in risks),
        "n_fcw": sum(1 for r in risks if r.level == RiskLevel.FCW),
        "n_aeb": sum(1 for r in risks if r.level == RiskLevel.AEB_WARNING),
    }


def run_video(args) -> int:
    err = core_import_error()
    if err:
        print(f"[demo] {err}")
        return 1
    # 检测器后端按分支才 import：选 yolo 时不该拉起 torch（dfine extra），
    # 选 dfine 时也不该拉起 ultralytics（AGPL，yolo extra）。此前两支一起 import，
    # 只装一个 extra 的用户会因为另一个后端缺依赖而失败。

    is_dir, probed_fps, frame_size = _probe_source(args.source)
    fps, fps_src = _pick_fps(args.fps, probed_fps, is_dir=is_dir)
    cfg = AEBConfig(fps=fps)

    if args.calib:
        # 标定结果以像素为单位（f·H），换分辨率不能直接套用；apply_calibration 会在
        # 尺寸不符时抛 ValueError，这里转成一句人话 + 退出码 1。
        try:
            calib = load_calibration(args.calib)
            apply_calibration(cfg.camera, calib, frame_size=frame_size)
        except ValueError as exc:
            print(f"[demo] cannot use calibration: {exc}")
            return 1
        print(
            f"[demo] calibration loaded from {args.calib}: "
            f"f·H = {cfg.camera.focal_height:.1f}, horizon = {cfg.camera.horizon_y:.1f}"
        )

    # 速度来源必须显式：命令行、GPS 元数据、默认 12 m/s 三者的阈值与一致性筛选
    # 结果不同，日志里要能分辨，不能只打一个数。
    if args.ego_speed is not None:
        cfg.ego_speed_mps = args.ego_speed
        speed_src = "command line (--ego-speed)"
    elif not is_dir and (v := load_ego_speed(args.source)) is not None:
        cfg.ego_speed_mps = v
        speed_src = "metadata (GPS aggregate)"
    else:
        speed_src = f"default {cfg.ego_speed_mps:.1f} m/s"
    if args.detector == "dfine":
        from aeb.detectors import DFineDetector

        det_kwargs = {"conf": args.conf, "device": args.device}
        if args.dfine_weights:
            det_kwargs["weights"] = args.dfine_weights
        if args.dfine_config:
            det_kwargs["config"] = args.dfine_config
        det = DFineDetector(**det_kwargs)
    else:
        from aeb.detectors import COCO_TO_AEB, YOLOv8Detector

        class_map = COCO_TO_AEB if args.coco else None
        det = YOLOv8Detector(args.weights, conf=args.conf, device=args.device, class_map=class_map)
    pipe = AEBPipeline(cfg, det)
    print(f"[demo] fps = {fps:.2f}  source = {fps_src}")
    print(
        f"[demo] ego speed = {cfg.ego_speed_mps:.1f} m/s "
        f"({cfg.ego_speed_mps * 3.6:.0f} km/h)  source = {speed_src}  "
        f"AEB threshold = {cfg.risk.ladder(cfg.ego_speed_mps)[2]:.2f}s"
    )

    writer = None
    frame_idx = 0
    counts = {"NORMAL": 0, "ATTENTION": 0, "FCW": 0, "AEB": 0}
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        jf = stack.enter_context(open(args.jsonl, "w", encoding="utf-8")) if args.jsonl else None
        for frame in iter_source(args.source):
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            result = pipe.process(frame, frame_idx)
            counts[LEVEL_NAMES[result.global_level]] += 1
            if jf is not None:
                jf.write(json.dumps(frame_record(result), ensure_ascii=False) + "\n")
            out = draw_video(frame, result, pipe)

            if writer is None and args.out:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            if writer is not None:
                writer.write(out)
            if args.show:
                cv2.imshow("AEB", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"[demo] frame {frame_idx}  global={result.global_level.name}")
        # 异常也要收尾：半截 mp4 与没关闭的 JSONL 都比失败本身更难排查。
        if writer is not None:
            stack.callback(writer.release)
        if args.show:
            # 无 GUI 的 OpenCV（headless 构建）里 destroyAllWindows() 会抛 cv2.error。
            stack.callback(cv2.destroyAllWindows)

    elapsed = time.perf_counter() - started
    rate = frame_idx / elapsed if elapsed > 0 else 0.0
    print(
        f"[demo] summary: {frame_idx} frames in {elapsed:.2f}s "
        f"(processing {rate:.1f} fps)  "
        f"NORMAL {counts['NORMAL']} / ATTENTION {counts['ATTENTION']} / "
        f"FCW {counts['FCW']} / AEB {counts['AEB']}"
    )
    if args.jsonl:
        print(f"[demo] per-frame records written to {args.jsonl}")
    print(f"[demo] done, {frame_idx} frames")
    return 0


# ---------------------------------------------------------------------------
# calib（原 examples/calibrate_video.py 的实现整体搬入）
# ---------------------------------------------------------------------------


def run_calib(args) -> int:
    err = core_import_error()
    if err:
        print(f"[calib] {err}")
        return 1
    from aeb.detectors import COCO_TO_AEB, YOLOv8Detector

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
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera = CameraConfig()  # horizon=360 默认；focal_px/cam_height_m 是待标定的占位
    if src_w > 0 and src_h > 0:
        # 标定值以像素为单位，记录实际分辨率，run 侧据此拒绝对不上的配置。
        camera.img_w, camera.img_h = src_w, src_h
    class_map = COCO_TO_AEB if args.coco else None
    detector = YOLOv8Detector(args.weights, conf=args.conf, device=args.device, class_map=class_map)

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
        # 失败必须是失败：此前返回 0，脚本编排与 CI 会把它当成功。
        return 1
    print(
        f"before calibration: f·H = {camera.focal_height:.1f} (placeholder 1100x1.5)  "
        f"horizon = {camera.horizon_y:.0f} (default)"
    )
    print(f"after calibration:  f·H = {fh:.1f}  horizon = {horizon:.1f}")
    camera.calibrate_scale(fh)
    camera.calibrate_horizon(horizon)
    print(f"  → focal_px = {camera.focal_px:.1f} (cam_height_m kept at {camera.cam_height_m} m)")

    if args.out_config:
        out = save_calibration(
            args.out_config,
            camera,
            source_path=str(path),
            source_width=src_w or camera.img_w,
            source_height=src_h or camera.img_h,
            fps=fps,
            focal_height=fh,
            samples=n_samples,
            tracks=n_tracks,
            version=package_version(),
        )
        print(f"  → saved calibration config: {out}")
        print(f'    reuse it with: mono-aeb run --calib "{out}" --source <video> ...')
    else:
        print("  → not saved (pass --out-config <file.json> to reuse it with `run --calib`)")
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
    p_run.add_argument(
        "--fps",
        type=float,
        default=None,
        help="frame rate used for the time axis; default: the video's own CAP_PROP_FPS, "
        "falling back to 30. Image directories have no intrinsic frame rate and must "
        "pass this explicitly",
    )
    p_run.add_argument(
        "--calib",
        default=None,
        help="calibration JSON written by `mono-aeb calib --out-config`; applies f·H and "
        "horizon, and refuses a file estimated at a different resolution",
    )
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
    p_run.add_argument(
        "--jsonl",
        default=None,
        help="write one JSON object per frame (frame / global_level / counts / min TTC and "
        "distance); inf and nan are written as null",
    )
    p_run.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="process at most this many frames (useful for a quick check or a demo)",
    )
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
    p_calib.add_argument(
        "--device",
        default="0",
        help="inference device for the calibration detector (e.g. cpu, 0)",
    )
    p_calib.add_argument(
        "--coco",
        action="store_true",
        help="weights are COCO-pretrained (e.g. the default yolov8n.pt): map COCO classes "
        "to the AEB ones; without it the default weights are filtered by the AEB class ids",
    )
    p_calib.add_argument("--max-frames", type=int, default=None)
    p_calib.add_argument("--fps", type=float, default=30.0)
    p_calib.add_argument(
        "--out-config",
        default=None,
        help="write the estimate to this JSON so `mono-aeb run --calib <file>` can reuse it",
    )
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
