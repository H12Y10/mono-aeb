"""V0：固定梯形 ROI（自车路径走廊投影）。

复用测距的同一套相机模型，把「自车车道走廊」这块地面矩形投影成图像梯形，
而非手写任意像素坐标——这样 ROI 与测距自洽，且随标定参数一起调整（贴合实际视角）。
"""

import cv2
import numpy as np

from .base import InPathFilter
from ..config import CameraConfig, EgoPathConfig
from ..types import Track


def ego_path_polygon(cam: CameraConfig, ego: EgoPathConfig) -> np.ndarray:
    """地面车道走廊 → 图像梯形，返回 (4,2) 像素坐标。

    顺序：左下 → 右下 → 右上 → 左上。顶点超出图像范围则裁剪到边界。
    """
    cx, f, H, hy = cam.principal_x, cam.focal_px, cam.cam_height_m, cam.horizon_y
    half_w = ego.lane_width_m / 2.0
    x_left = ego.lateral_offset_m - half_w
    x_right = ego.lateral_offset_m + half_w

    def proj(X, Z):
        u = cx + f * X / Z
        v = hy + f * H / Z
        return (u, v)

    pts = [
        proj(x_left, ego.z_near_m),    # 左下
        proj(x_right, ego.z_near_m),   # 右下
        proj(x_right, ego.z_far_m),    # 右上
        proj(x_left, ego.z_far_m),     # 左上
    ]
    poly = np.array(pts, dtype=np.float32)
    poly[:, 0] = np.clip(poly[:, 0], 0, cam.img_w)
    poly[:, 1] = np.clip(poly[:, 1], 0, cam.img_h)
    return poly


class FixedTrapezoidROI(InPathFilter):
    def __init__(self, camera: CameraConfig, ego_path: EgoPathConfig):
        self.poly = ego_path_polygon(camera, ego_path)

    def in_path(self, track: Track) -> bool:
        cx, cy = track.bottom_center
        return cv2.pointPolygonTest(self.poly, (cx, cy), False) >= 0

    def draw(self, frame: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """可视化 ROI（调试 / demo 用）。"""
        return cv2.polylines(frame.copy(), [self.poly.astype(np.int32)],
                             isClosed=True, color=color, thickness=thickness)
