"""从 BDD100K samples-1k 的 info/*.json 读取真实自车 GPS 速度。

info/*.json 里 `locations[].speed` 是 1Hz 采样的 GPS 速度 (m/s)，
与视频同名（`{stem}.mov` ↔ `info/{stem}.json`）。用整段的均值作为
「假设自车速度」喂给 AEB 状态机，实现速度自适应阈值。

用法：
    from aeb.ego_speed import load_ego_speed
    v = load_ego_speed("/path/to/samples-1k/videos/xxxx.mov")
"""

import json
from pathlib import Path

import numpy as np


def load_ego_speed(video_path, info_dir=None, agg="mean"):
    """返回该视频的平均/中位 GPS 速度 (m/s)，找不到则返回 None。

    info_dir 默认自动推断为 videos 的同级 info/ 目录（samples-1k 结构）。
    agg: "mean" | "median"
    """
    p = Path(video_path)
    if info_dir is None:
        info_dir = p.parent.parent / "info"   # videos/ -> ../info
    info_file = Path(info_dir) / f"{p.stem}.json"
    if not info_file.exists():
        return None
    data = json.loads(info_file.read_text(encoding="utf-8"))
    locs = data.get("locations", [])
    speeds = [l["speed"] for l in locs if isinstance(l.get("speed"), (int, float))]
    if not speeds:
        return None
    if agg == "median":
        speeds_sorted = sorted(speeds)
        mid = len(speeds_sorted) // 2
        return float(speeds_sorted[mid])
    return float(sum(speeds) / len(speeds))


def load_ego_speed_series(video_path, info_dir=None):
    """返回 (times_sec, speeds)：相对片段起点的 GPS 速度时间序列 (m/s)。

    times_sec 以首个 timestamp 为 0（秒）。无数据返回 (None, None)。
    """
    p = Path(video_path)
    if info_dir is None:
        info_dir = p.parent.parent / "info"   # videos/ -> ../info
    info_file = Path(info_dir) / f"{p.stem}.json"
    if not info_file.exists():
        return None, None
    data = json.loads(info_file.read_text(encoding="utf-8"))
    locs = data.get("locations", [])
    ts, sp = [], []
    for l in locs:
        t, s = l.get("timestamp"), l.get("speed")
        if isinstance(t, (int, float)) and isinstance(s, (int, float)):
            ts.append(float(t))
            sp.append(float(s))
    if not ts:
        return None, None
    ts = np.asarray(ts, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)
    ts = (ts - ts[0]) / 1000.0
    return ts, sp


def ego_speed_at(times_sec, speeds, t_sec, default=0.0):
    """按时间线性插值返回 ego 速度 (m/s)；无数据返回 default。"""
    if times_sec is None or len(times_sec) == 0:
        return default
    return float(np.interp(t_sec, times_sec, speeds))


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        v = load_ego_speed(path)
        if v is None:
            print(f"{path}: 无 info 数据")
        else:
            print(f"{path}: {v:.2f} m/s = {v*3.6:.1f} km/h")
