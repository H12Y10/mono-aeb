"""YOLOv8 检测器适配。

默认假设模型输出的 class id 已是 AEB 7 类 id（person=0…motor=6），
即 `data.yaml` 精调的 baseline / D-FINE 都是这种。若用 COCO 预训练的
原始 yolov8n.pt，则传入 `class_map=COCO_TO_AEB` 做一次映射。
"""

import numpy as np
from ultralytics import YOLO

from .base import BaseDetector
from ..types import AEB_CLASS_NAMES, Detection

# COCO id → AEB id（person/rider/car/truck/bus/bike/motor）
COCO_TO_AEB = {
    0: 0,   # person  → person
    1: 5,   # bicycle → bike
    2: 2,   # car     → car
    3: 6,   # motorcycle → motor
    5: 4,   # bus     → bus
    7: 3,   # truck   → truck
    # 注意：COCO 无 "rider"，baseline 里 rider 不出现（正常）
}


class YOLOv8Detector(BaseDetector):
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640,
                 device: str = "cuda:0", class_map: dict | None = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.class_map = class_map  # None = 模型已输出 AEB id

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        dets = []
        for (x1, y1, x2, y2), s, c in zip(xyxy, confs, clss):
            aeb_c = int(c) if self.class_map is None else self.class_map.get(int(c))
            if aeb_c is not None and 0 <= aeb_c < len(AEB_CLASS_NAMES):
                dets.append(Detection(float(x1), float(y1), float(x2), float(y2),
                                      float(s), int(aeb_c)))
        return dets
