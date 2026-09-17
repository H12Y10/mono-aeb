"""逐视频离线自标定：用「GPS ego 速度 + 免标定尺度 TTC」联合反解
测距尺度 f·H 与地平线 horizon_y。

原理：
  对（近似）静止目标，碰撞时间 τ = D / v_ego（接近速度 ≈ 自车速度），
  尺度 TTC 免标定给出 τ，故 D = v_ego · τ。
  地面平面模型：y_bottom = horizon_y + f·H / D，即
      y_bottom = horizon_y + f·H · (1/D)
  对 (1/D, y_bottom) 做 Theil-Sen 稳健直线拟合，一次同时解出：
      - 斜率   = f·H（测距尺度）
      - 截距   = horizon_y（地平线 / 消失点，之前默认 360 未标定）

  早期实现（已移除）只解 f·H（horizon 固定 360），近距离目标 denom
  ≈30px 时 ±38px 的地平线误差就会造成约 2 倍 f·H 偏差；联合拟合消除该偏差。
"""

import numpy as np

from .config import CameraConfig
from .detectors import BaseDetector
from .tracker import ByteTrackTracker
from .ttc import ScaleTTC, TrackHistory

# f·H（= focal_px × 相机高）的物理合理范围：典型 dashcam focal_px≈400~1600、
# 相机高≈1.2~1.7m ⇒ f·H 约 500~2700。放宽到 [300, 3000] 只拦「明显失真」——
# 坏 track 的尺度反解会把 f·H 拉到 ~37 或 ~5000 这种任何车载相机都不可能的值，
# 此时测距整体缩放错 10×，宁可回落默认相机参数（f·H=1650）也不用垃圾值。
FOCAL_HEIGHT_MIN = 300.0
FOCAL_HEIGHT_MAX = 3000.0


def _theil_sen_line(x: np.ndarray, y: np.ndarray, max_pts: int = 2000):
    """对 (x, y) 做 Theil-Sen 稳健直线拟合，返回 (slope, intercept)。

    对约 50% 离群不敏感；样本过多时系统抽稀到 max_pts 以控制 O(N²) 内存。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) > max_pts:
        idx = np.linspace(0, len(x) - 1, max_pts).astype(int)
        x, y = x[idx], y[idx]
    n = len(x)
    if n < 2:
        return None, None
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    valid = upper & (dx != 0)
    if not valid.any():
        return None, None
    slopes = dy[valid] / dx[valid]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def _collect_ground_samples(
    detector: BaseDetector,
    frames,
    ego_speed_fn,
    fps: float,
    min_ego_speed: float,
    min_history: int,
    max_frames: int | None,
):
    """跑一遍 frames，收集 (track_id, D, y_bottom) 样本（D = v_ego·τ）。

    ego_speed_fn(frame_idx) -> 自车速度 (m/s)。
    返回 (samples, n_tracks)，samples 为 [(track_id, D, y_bottom), ...]。
    """
    tracker = ByteTrackTracker(fps=fps)
    scale_ttc = ScaleTTC(fps)
    history = TrackHistory(max_len=10)

    samples = []
    track_ids = set()

    for frame_idx, frame in frames:
        if max_frames is not None and frame_idx >= max_frames:
            break
        v_ego = float(ego_speed_fn(frame_idx))
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame)

        active = set()
        for t in tracks:
            active.add(t.track_id)
            history.push(t.track_id, frame_idx, 0.0, t.height)
        history.prune(active)

        if v_ego < min_ego_speed:
            continue  # 车速太低时 τ = D/v 数值不稳

        for t in tracks:
            if len(history.get(t.track_id)) < min_history:
                continue
            ttc = scale_ttc.estimate(t.track_id, history)
            if not np.isfinite(ttc) or ttc <= 0:
                continue
            D = v_ego * ttc
            y_bottom = t.bbox[3]
            samples.append((t.track_id, D, y_bottom))
            track_ids.add(t.track_id)

    return samples, len(track_ids)


def estimate_ground_plane(
    detector: BaseDetector,
    frames,
    camera: CameraConfig,
    ego_speed_fn,
    fps: float = 30.0,
    min_ego_speed: float = 2.0,
    min_history: int = 5,
    min_samples: int = 30,
    min_tracks: int = 1,
    max_frames: int | None = None,
):
    """联合反解 (f·H, horizon_y)：对 (1/D, y_bottom) 做 Theil-Sen 直线拟合。

    返回 (focal_height, horizon_y, n_samples, n_tracks)；
    标定失败时 focal_height/horizon_y 为 None。
    """
    samples, n_tracks = _collect_ground_samples(
        detector, frames, ego_speed_fn, fps,
        min_ego_speed, min_history, max_frames,
    )
    if len(samples) < min_samples or n_tracks < min_tracks:
        return None, None, len(samples), n_tracks

    D = np.array([s[1] for s in samples], dtype=np.float64)
    y = np.array([s[2] for s in samples], dtype=np.float64)
    x = 1.0 / D

    # 1) horizon_y：联合拟合截距（对 D 的乘性偏差不敏感，稳健）
    _, horizon = _theil_sen_line(x, y)
    if not np.isfinite(horizon) or not (0.0 <= horizon <= camera.img_h):
        return None, None, len(samples), n_tracks  # 地平线必须在可见图像内

    # 2) f·H：用解出的 horizon 做每轨迹中位（Theil-Sen 斜率对 Kalman 平滑滞后
    #    放大更明显，中位法对 f·H 更稳），再跨轨迹取中位
    per_track = {}
    for tid, D_i, y_bottom in samples:
        denom = y_bottom - horizon
        if denom <= 0:
            continue
        per_track.setdefault(tid, []).append(D_i * denom)
    track_medians = [float(np.median(v)) for v in per_track.values() if v]
    if not track_medians:
        return None, None, len(samples), n_tracks
    fh = float(np.median(track_medians))

    # 3) f·H 物理合理性：超出 [300, 3000] 判标定失败（见模块级常量注释）
    if not (FOCAL_HEIGHT_MIN <= fh <= FOCAL_HEIGHT_MAX):
        return None, None, len(samples), n_tracks

    return fh, float(horizon), len(samples), n_tracks
