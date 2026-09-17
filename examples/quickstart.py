"""mono-aeb 一键体验：依赖自检 + 合成 demo。

用法：
    python examples/quickstart.py --demo              # 合成逼近目标 → 完整决策链 → 状态升级（默认）
    python examples/quickstart.py --check             # 仅做依赖自检
    python examples/quickstart.py --demo --out a.mp4  # 输出可视化视频
    python examples/quickstart.py --demo --show       # 实时窗口显示

说明：
    --demo 用 SyntheticDetector 生成一个匀速逼近的目标，驱动
    「跟踪 → 测距 → TTC → 决策」全链，不依赖真实视频 / 检测权重，
    展示 NORMAL → ATTENTION → FCW → AEB 的状态演进。
    真实视频请用 examples/demo_aeb.py（默认 YOLOv8n 检测，--coco 映射）。
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from aeb.config import AEBConfig  # noqa: E402
from aeb.detectors import BaseDetector  # noqa: E402
from aeb.pipeline import AEBPipeline  # noqa: E402
from aeb.types import Detection, RiskLevel  # noqa: E402

# 核心依赖（--demo 即够用）：tracker/测距/TTC 需要，不含检测权重
CORE_DEPS = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "lap": "lap（无对应平台 wheel 时可换 lapx）",
}

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


def missing_deps(need_detector: bool = False) -> list:
    """返回缺失依赖的人话列表；need_detector=True 时额外检查 ultralytics。"""
    deps = dict(CORE_DEPS)
    if need_detector:
        deps["ultralytics"] = "ultralytics"
    missing = []
    for mod, pkg in deps.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(f"{mod}（pip install {pkg}）")
    return missing


class SyntheticDetector(BaseDetector):
    """--demo 专用：生成一个匀速逼近的目标，驱动完整决策链。

    目标以 v_close 从 d0 匀速逼近，底边 / 框高按地面平面模型反解
    （y_bottom = horizon + f·H/d，框高 h = f·H_real/d，与相机同高时
    h = y_bottom − horizon）。这样 DistanceTTC 与 ScaleTTC 对 (t, D) 与
    (t, 1/h) 的 Theil-Sen 拟合都能恢复 TTC = D/v_close，状态自然升级。
    """

    def __init__(self, fps: float = 30.0, v_close: float = 10.0,
                 d0: float = 45.0, focal_height: float = 1650.0,
                 horizon: float = 360.0, cx: float = 640.0,
                 aspect: float = 1.2, d_min: float = 5.0):
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
        h = self.fh / d            # 与相机同高目标：框高 = 底边到 horizon 的距离
        w = h * self.aspect
        x1 = self.cx - w / 2
        x2 = self.cx + w / 2
        y1 = y_bottom - h
        y2 = y_bottom
        return [Detection(float(x1), float(y1), float(x2), float(y2),
                          0.9, 2)]  # score=0.9, class=2 (car)


def make_frame() -> np.ndarray:
    """生成 demo 用的深色背景帧（天空 + 路面 + horizon 线）。"""
    frame = np.full((720, 1280, 3), (40, 40, 40), np.uint8)
    frame[360:] = (70, 70, 70)  # 路面略亮
    cv2.line(frame, (0, 360), (1280, 360), (180, 180, 180), 1)
    return frame


def draw(frame: np.ndarray, result, pipeline: AEBPipeline) -> np.ndarray:
    """画 ROI + 目标框 + 距离/TTC + 全局风险横幅。"""
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
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    g = result.global_level
    banner = np.full((44, frame.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(banner, f"GLOBAL: {LEVEL_NAMES[g]}", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, LEVEL_COLORS[g], 2, cv2.LINE_AA)
    return np.vstack([banner, frame])


def run_demo(args) -> int:
    missing = missing_deps(need_detector=False)
    if missing:
        print("[quickstart] 缺少依赖，请先安装：")
        for m in missing:
            print(f"  - {m}")
        return 1

    cfg = AEBConfig(fps=args.fps)
    det = SyntheticDetector(fps=cfg.fps, v_close=args.v_close)
    pipe = AEBPipeline(cfg, det)

    print(f"[quickstart] 自车速度 = {cfg.ego_speed_mps:.0f} m/s，"
          f"AEB 阈值 = {cfg.risk.ladder(cfg.ego_speed_mps)[2]:.2f}s")
    print("[quickstart] 目标以 %.0f m/s 匀速逼近，逐帧打印状态变化："
          % args.v_close)

    writer = None
    prev_level = None
    for i in range(args.frames):
        frame = make_frame()
        result = pipe.process(frame, i)
        out = draw(frame, result, pipe)

        level = result.global_level
        if level != prev_level:
            r = result.risks[0] if result.risks else None
            d = r.distance if r is not None else float("inf")
            ttc = r.ttc_fused if r is not None else float("inf")
            print(f"  frame {i:3d}: {LEVEL_NAMES[level]:<9s}"
                  f"  d={d:5.1f}m  ttc={ttc:5.2f}s")
            prev_level = level

        if writer is None and args.out:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        if writer is not None:
            writer.write(out)
        if args.show:
            cv2.imshow("mono-aeb demo", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[quickstart] 完成，共 {args.frames} 帧"
          + (f"，视频已写入 {args.out}" if args.out else ""))
    return 0


def main():
    ap = argparse.ArgumentParser(description="mono-aeb 一键体验")
    ap.add_argument("--demo", action="store_true", default=True,
                    help="合成 demo（默认）")
    ap.add_argument("--check", action="store_true",
                    help="仅做依赖自检，不跑 demo")
    ap.add_argument("--frames", type=int, default=120,
                    help="demo 帧数（默认 120）")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--v-close", type=float, default=10.0,
                    help="合成目标的接近速度 m/s（默认 10）")
    ap.add_argument("--out", default=None, help="输出视频路径（可选）")
    ap.add_argument("--show", action="store_true", help="实时显示")
    args = ap.parse_args()

    if args.check:
        core = missing_deps(need_detector=False)
        det = missing_deps(need_detector=True)
        print("[quickstart] 核心依赖（--demo 需要）：")
        print("  " + ("全部就绪" if not core else "\n  ".join(core)))
        print("[quickstart] 检测依赖（真实视频 examples/demo_aeb.py 需要）：")
        print("  " + ("全部就绪" if not det else "\n  ".join(det)))
        return 0 if not core else 1

    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
