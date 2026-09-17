"""AEB 主链路：检测 → 跟踪 → In-Path → 测距 → TTC → 决策。"""

import numpy as np

from .config import AEBConfig
from .decision import RiskStateMachine
from .detectors import BaseDetector
from .distance import GroundPlaneDistance
from .in_path import FixedTrapezoidROI
from .tracker import ByteTrackTracker
from .ttc import DistanceTTC, ScaleTTC, TTCFusion, TrackHistory
from .types import FrameResult, RiskFeature, RiskLevel


class AEBPipeline:
    def __init__(self, config: AEBConfig, detector: BaseDetector):
        self.cfg = config
        self.detector = detector

        self.tracker = ByteTrackTracker(
            track_thresh=config.track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            fps=config.fps,
        )
        self.roi = FixedTrapezoidROI(config.camera, config.ego_path)
        self.distance = GroundPlaneDistance(config.camera)

        self.history = TrackHistory(max_len=config.history_len)
        self.ttc_dist = DistanceTTC(config.fps)
        self.ttc_scale = ScaleTTC(config.fps)
        self.fusion = TTCFusion(config.persist_k)
        self.sm = RiskStateMachine(config.risk,
                                   ego_speed_mps=config.ego_speed_mps)

    def set_ego_speed(self, v_mps: float):
        """运行时更新自车速度（阈值梯随车速平移）。"""
        self.sm.set_ego_speed(v_mps)

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)

        # 先灌入历史（距离 + bbox 高度），再算 TTC（同帧数据齐全）
        active_ids = set()
        dists = {}
        for t in tracks:
            active_ids.add(t.track_id)
            d = self.distance.estimate(t, frame)
            dists[t.track_id] = d
            self.history.push(t.track_id, frame_idx, d, t.height)
        self.history.prune(active_ids)

        risks = []
        for t in tracks:
            d = dists[t.track_id]
            in_path = self.roi.in_path(t)
            ttc_d = self.ttc_dist.estimate(t.track_id, self.history)
            ttc_s = self.ttc_scale.estimate(t.track_id, self.history)
            ttc_f = self.fusion.fuse(t.track_id, frame_idx, ttc_d, ttc_s)
            v_close = self.ttc_dist.closing_speed(t.track_id, self.history)

            risk = RiskFeature(
                track_id=t.track_id,
                in_path=in_path,
                distance=d,
                closing_speed=v_close,
                ttc_distance=ttc_d,
                ttc_scale=ttc_s,
                ttc_fused=ttc_f,
                required_deceleration=self._required_decel(d, v_close),
                detection_confidence=t.score,
                track_age=t.age,
            )

            # 仅在自车路径内参与决策；路径外保持 NORMAL
            level = self.sm.decide(risk) if in_path else RiskLevel.NORMAL
            risk.level = level
            risks.append(risk)

        global_level = max((r.level for r in risks), default=RiskLevel.NORMAL)
        return FrameResult(
            frame_idx=frame_idx,
            detections=detections,
            tracks=tracks,
            risks=risks,
            global_level=global_level,
        )

    def _required_decel(self, distance: float, closing_speed: float) -> float:
        """所需减速度 a_req = v² / (2·(D - d_safe))，D<=d_safe 时视为已达极限。"""
        d_safe = self.cfg.risk.d_safe
        if not np.isfinite(distance) or distance <= d_safe or closing_speed <= 0:
            return 0.0
        return closing_speed ** 2 / (2 * (distance - d_safe))
