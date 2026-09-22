"""集中配置：相机标定 / 自车路径 / 风险阈值 / 检测跟踪参数。

相机标定（BDD100K 众包采集、相机逐视频不同，无统一内参）：
  - 测距只用 f·H 这一个「尺度」量（focal_height 属性），f、H 存在尺度简并；
  - horizon_y 由车道线消失点逐视频求；
  - 尺度逐视频自标定（实际入口）：examples/calibrate_video.py 用 calibration.py 的
    estimate_ground_plane() 联合反解 f·H 与 horizon_y，再经 calibrate_scale()/
    calibrate_horizon() 落到本类；
  - calibrate_from_ego_speed()（ego 速度 + 尺度 TTC）与 calibrate_from_known_width()
    （已知车宽）为**接口预留，当前无调用点**；
  - 下面这组默认值只是逐视频标定的初始化/fallback，不是全局真值；
  - 免标定尺度 TTC（TTC_scale = h/ḣ）不经过本类，见 ttc 模块。
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class CameraConfig:
    """地面平面相机模型（测距 + ROI 共用同一套参数）。"""
    cam_height_m: float = 1.5     # 相机离地高度 (m)；与 focal_px 只以乘积进入测距
    focal_px: float = 1100.0      # 焦距 (像素，占位值)；ROI 横向投影需单独用 f
    horizon_y: float = 360.0      # 地平线像素行（受 pitch 影响，720/2 近似，逐视频校正）
    principal_x: float = 640.0    # 主点 x（图像中心）
    img_w: int = 1280
    img_h: int = 720

    @property
    def focal_height(self) -> float:
        """测距唯一尺度参数 f·H（像素·米）。

        地面平面公式 D = f·H / (y_bottom − horizon_y) 中 f、H 只以乘积出现
        （尺度简并），逐视频标定只需拟合这一个量；横向 ROI 投影仍用 focal_px。
        """
        return self.focal_px * self.cam_height_m

    def calibrate_horizon(self, horizon_y: float) -> None:
        """逐视频设置地平线（pitch），来自车道线消失点。"""
        self.horizon_y = float(horizon_y)

    def calibrate_scale(self, focal_height: float) -> None:
        """逐视频设置测距尺度 f·H，落到 focal_px 上（cam_height_m 保持物理含义）。"""
        fh = float(focal_height)
        if fh <= 0:
            raise ValueError(f"focal_height 必须为正，得到 {fh}")
        self.focal_px = fh / self.cam_height_m

    def calibrate_from_ego_speed(
        self, ttc_scale: float, y_bottom: float, ego_speed_mps: float
    ) -> None:
        """用 ego 速度 + 尺度 TTC 反解测距尺度 f·H（**接口预留，当前无调用点**）。

        实际逐视频标定走 examples/calibrate_video.py → calibration.estimate_ground_plane()，
        其结果经 calibrate_scale() 落到本类；本方法保留为单次反解的对外接口。

        尺度 TTC 免标定可得：τ = Δt·s/Δs（s 为跟踪框高/宽）。
        对（近似）静止目标：D = ego_speed_mps · τ，
        再 D = f·H/(y_bottom − horizon_y) ⇒ f·H = D·(y_bottom − horizon_y)。
        多帧/多目标取中位数以抑制噪声（聚合由调用方负责，这里做单次反解）。
        """
        denom = float(y_bottom) - self.horizon_y
        if ttc_scale <= 0 or denom <= 0:
            raise ValueError(
                f"非法反解输入：ttc_scale={ttc_scale}, y_bottom-horizon={denom}"
            )
        d = float(ego_speed_mps) * ttc_scale
        self.calibrate_scale(d * denom)

    def calibrate_from_known_width(
        self,
        widths_px: Sequence[float],
        y_bottoms: Sequence[float],
        real_width_m: float = 1.8,
        horizon_y: Optional[float] = None,
    ) -> None:
        """由已知目标宽度反标定相机高度 H（轿车 real_width_m≈1.8m）。

        w_px = (W/H)·(y_bottom − horizon_y)：像素宽对 (y_bottom − horizon_y)
        是过原点直线，斜率 = W/H ⇒ H = W/斜率。
        注意：宽度只决定 H 与真实尺寸的比例，不决定绝对尺度 f；
        绝对尺度还需 calibrate_from_ego_speed() 或额外假设相机高度。
        """
        if horizon_y is not None:
            self.calibrate_horizon(horizon_y)
        xs = [float(y) - self.horizon_y for y in y_bottoms]
        ws = [float(w) for w in widths_px]
        if len(xs) != len(ws) or len(xs) == 0:
            raise ValueError("widths_px 与 y_bottoms 需等长且非空")
        if any(x <= 0 for x in xs) or any(w <= 0 for w in ws):
            raise ValueError("y_bottom 需在 horizon 之下，宽度需为正")
        # 过原点最小二乘：slope = Σ(w·x) / Σ(x²)
        slope = sum(w * x for w, x in zip(ws, xs)) / sum(x * x for x in xs)
        if slope <= 0:
            raise ValueError("拟合斜率为非正，无法标定")
        self.cam_height_m = float(real_width_m) / slope


@dataclass
class EgoPathConfig:
    """自车路径走廊：地面平面上的矩形 → 投影成图像梯形。"""
    lane_width_m: float = 3.5     # 车道宽 (m)
    z_near_m: float = 4.5         # 走廊近端纵向距离 (m)（≈图像底边对应距离）
    z_far_m: float = 80.0         # 走廊远端纵向距离 (m)
    lateral_offset_m: float = 0.0  # 自车中心横向偏移 (m)，正=右


@dataclass
class RiskThresholds:
    """状态机阈值。

    AEB 触发点随自车速度自适应：固定 TTC 阈值在高速时太激进、
    低速时太保守，改为按制动模型动态计算
        TTC_AEB(v) = t_react + v / (2·a_max)
    FCW / ATTENTION 在此基础上加固定前置余量，整条决策梯度随车速整体平移。
    """
    t_react: float = 0.5               # 驾驶员反应时间 (s)
    a_max: float = 6.86                # 紧急制动最大减速度 (m/s²，≈0.7g)
    ttc_fcw_margin: float = 1.0        # FCW 相对 AEB 的前置余量 (s)
    ttc_attention_margin: float = 2.0  # ATTENTION 相对 AEB 的前置余量 (s)
    d_safe: float = 5.0                # 安全距离 (m)，用于 a_req

    def ladder(self, v_ego_mps: float) -> tuple[float, float, float]:
        """返回 (ttc_attention, ttc_fcw, ttc_aeb)，随自车速度动态变化。"""
        v = max(v_ego_mps, 0.0)
        aeb = self.t_react + v / (2.0 * self.a_max)
        return (aeb + self.ttc_attention_margin,
                aeb + self.ttc_fcw_margin,
                aeb)


@dataclass
class AEBConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    ego_path: EgoPathConfig = field(default_factory=EgoPathConfig)
    risk: RiskThresholds = field(default_factory=RiskThresholds)

    # 检测器
    det_conf: float = 0.25
    # 跟踪（ByteTrack 默认）
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    fps: float = 30.0
    # 时序
    history_len: int = 10         # N = 5~15
    persist_k: int = 3            # 连续 K 帧低于阈值才升级
    # 自车速度 (m/s)：demo/probe 从 samples-1k/info/*.json 的 GPS speed 读真实值，
    # 未提供时用 12 m/s（≈43 km/h 城市）作为 fallback 默认
    ego_speed_mps: float = 12.0
