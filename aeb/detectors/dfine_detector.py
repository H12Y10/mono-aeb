"""D-FINE 检测器适配（可选高精度后端，optional high-accuracy backend）。

加载 `dfine_hgnetv2_m_aeb.yml` 训练的权重（best_stg1.pth），用 D-FINE 官方
`tools/inference/torch_inf.py` 的 deploy 方式推理，输出与本链一致的
`Detection`（AEB 7 类 id，无需再映射）。

预处理与训练 val 完全一致（见 configs/dfine/include/dataloader.yml）：
  Resize((640, 640)) 直接拉伸 + ToTensor（[0,1]，无 mean/std 归一化）。
D-FINE 是 DETR 风格，boxes 在归一化 [0,1] 坐标里回归，postprocessor 再乘
原图尺寸还原，所以拉伸不破坏检测。

用法：
    from aeb.detectors import DFineDetector
    det = DFineDetector(weights="/path/to/best_stg1.pth")
    dets = det.detect(bgr_frame)   # -> list[Detection]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import BaseDetector
from ..types import AEB_CLASS_NAMES, Detection

# D-FINE 后端依赖上游仓库，通过环境变量定位，避免硬编码本地路径：
#   DFINE_ROOT              D-FINE 上游仓库根目录（内含 src/）
#   MONO_AEB_DFINE_WEIGHTS  微调权重路径（BDD100K 许可受限，权重走受限分发，需自行获取）
_DFINE_ROOT = os.environ.get("DFINE_ROOT")
_DEFAULT_WEIGHTS = os.environ.get("MONO_AEB_DFINE_WEIGHTS")


def _resolve_device(device: str):
    """把 '0'/'cpu'/'cuda:0' 归一化成 torch.device，缺 CUDA 时退回 CPU。"""
    import torch

    if device is None or device == "cpu":
        return torch.device("cpu")
    if str(device).isdigit():
        device = f"cuda:{device}"
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        print("[dfine] CUDA 不可用，退回 CPU")
        d = torch.device("cpu")
    return d


class DFineDetector(BaseDetector):
    def __init__(self, weights: str = _DEFAULT_WEIGHTS,
                 config: str | None = None,
                 conf: float = 0.25, device: str = "cuda:0"):
        import torch
        import torch.nn as nn
        import torchvision.transforms as T

        if not _DFINE_ROOT:
            raise RuntimeError(
                "未设置 DFINE_ROOT 环境变量（D-FINE 上游仓库路径），无法加载 D-FINE 后端")
        if str(_DFINE_ROOT) not in sys.path:
            sys.path.insert(0, str(_DFINE_ROOT))
        from src.core import YAMLConfig

        weights = weights or _DEFAULT_WEIGHTS
        if not weights:
            raise RuntimeError(
                "未提供 D-FINE 权重：请传 weights 参数或设置 MONO_AEB_DFINE_WEIGHTS 环境变量")
        config = Path(config) if config else (
            Path(_DFINE_ROOT) / "configs" / "dfine" / "dfine_hgnetv2_m_aeb.yml")
        if not config.exists():
            raise FileNotFoundError(f"D-FINE 配置不存在：{config}")
        if not Path(weights).exists():
            raise FileNotFoundError(f"D-FINE 权重不存在：{weights}")

        cfg = YAMLConfig(str(config), resume=str(weights))
        # 用训练权重整体加载，禁掉 HGNetv2 预训练 backbone（避免重复加载）
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(weights, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
        cfg.model.load_state_dict(state)

        class _Model(nn.Module):
            def __init__(self, model, postprocessor):
                super().__init__()
                self.model = model
                self.postprocessor = postprocessor

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)

        self.conf = conf
        self._device = _resolve_device(device)
        self._model = _Model(cfg.model.deploy(), cfg.postprocessor.deploy())
        self._model.to(self._device).eval()
        # 预处理尺寸与 config 的 eval_spatial_size（[h, w]）一致，训练/推理分布不漂移。
        # 640 配置→640 拉伸；960 续训配置→960 拉伸（小目标优化）。
        eval_h, eval_w = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
        self._transform = T.Compose([T.Resize((eval_h, eval_w)), T.ToTensor()])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import torch
        from PIL import Image

        im_pil = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]], device=self._device)

        im_data = self._transform(im_pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            labels, boxes, scores = self._model(im_data, orig_size)

        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        dets: list[Detection] = []
        for c, (x1, y1, x2, y2), s in zip(labels, boxes, scores):
            if s < self.conf:
                continue
            c = int(c)
            if 0 <= c < len(AEB_CLASS_NAMES):
                dets.append(Detection(float(x1), float(y1),
                                      float(x2), float(y2), float(s), c))
        return dets
