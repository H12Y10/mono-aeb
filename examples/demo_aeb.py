"""AEB 链路离线 demo：视频 / 图片序列 → 检测+跟踪+测距+TTC+决策 → 可视化。

用法：
    python demo_aeb.py --source 视频.mp4 --weights yolov8n.pt --out out.mp4
    python demo_aeb.py --source ./frames_dir --fps 30 --show
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeb.config import AEBConfig
from aeb.detectors import COCO_TO_AEB, DFineDetector, YOLOv8Detector
from aeb.pipeline import AEBPipeline
from aeb.types import AEB_CLASS_NAMES, RiskLevel
from aeb.ego_speed import load_ego_speed

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


def iter_source(source: str):
    """图片目录按文件名排序逐帧读；否则按视频读。"""
    p = Path(source)
    if p.is_dir():
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += sorted(p.glob(ext))
        if not paths:
            raise SystemExit(f"目录里没有图片：{source}")
        for fp in paths:
            img = cv2.imread(str(fp))
            if img is not None:
                yield img
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{source}")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield img
    cap.release()


def draw(frame: np.ndarray, result, pipeline: AEBPipeline) -> np.ndarray:
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
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 全局风险横幅
    g = result.global_level
    banner = np.full((44, frame.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(banner, f"GLOBAL: {LEVEL_NAMES[g]}", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, LEVEL_COLORS[g], 2, cv2.LINE_AA)
    return np.vstack([banner, frame])


def main():
    ap = argparse.ArgumentParser(description="AEB 链路离线 demo")
    ap.add_argument("--source", required=True, help="视频文件或图片目录")
    ap.add_argument("--weights",
                    default="yolov8n.pt",
                    help="检测权重；默认 COCO 预训练 yolov8n.pt（配合 --coco），AEB 精调权重另传")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--coco", action="store_true",
                    help="权重是 COCO 预训练（如 yolov8n.pt），做 COCO→AEB 映射")
    ap.add_argument("--detector", choices=["yolo", "dfine"], default="yolo",
                    help="检测器：yolo（默认）或 dfine")
    ap.add_argument("--dfine-weights", default=None,
                    help="D-FINE 权重路径；缺省用 MONO_AEB_DFINE_WEIGHTS 环境变量")
    ap.add_argument("--dfine-config", default=None,
                    help="D-FINE 配置 yaml；缺省用 dfine_hgnetv2_m_aeb.yml")
    ap.add_argument("--out", default=None, help="输出视频路径（可选）")
    ap.add_argument("--show", action="store_true", help="实时显示")
    ap.add_argument("--ego-speed", type=float, default=None,
                    help="自车速度 m/s；默认自动从 samples-1k/info/*.json 读 GPS 速度")
    args = ap.parse_args()

    cfg = AEBConfig(fps=args.fps)
    if args.ego_speed is not None:
        cfg.ego_speed_mps = args.ego_speed
    elif not Path(args.source).is_dir():
        v = load_ego_speed(args.source)
        if v is not None:
            cfg.ego_speed_mps = v
    if args.detector == "dfine":
        det_kwargs = dict(conf=args.conf, device=args.device)
        if args.dfine_weights:
            det_kwargs["weights"] = args.dfine_weights
        if args.dfine_config:
            det_kwargs["config"] = args.dfine_config
        det = DFineDetector(**det_kwargs)
    else:
        class_map = COCO_TO_AEB if args.coco else None
        det = YOLOv8Detector(args.weights, conf=args.conf, device=args.device,
                             class_map=class_map)
    pipe = AEBPipeline(cfg, det)
    print(f"[demo] 自车速度 = {cfg.ego_speed_mps:.1f} m/s "
          f"({cfg.ego_speed_mps*3.6:.0f} km/h)  "
          f"AEB阈值 = {cfg.risk.ladder(cfg.ego_speed_mps)[2]:.2f}s")

    writer = None
    frame_idx = 0
    for frame in iter_source(args.source):
        result = pipe.process(frame, frame_idx)
        out = draw(frame, result, pipe)

        if writer is None and args.out:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

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
    print(f"[demo] 完成，共 {frame_idx} 帧")


if __name__ == "__main__":
    main()
