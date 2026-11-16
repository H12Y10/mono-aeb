# Plugging in your own detector

`mono-aeb` ships two backends (YOLOv8, D-FINE), but neither is special: the decision
chain only consumes a list of boxes. If you have your own model — a different YOLO
head, a transformer, a classical CV pipeline — you plug it in by implementing one
method.

## The contract

```python
from aeb.detectors import BaseDetector
from aeb.types import Detection


class MyDetector(BaseDetector):
    def detect(self, frame) -> list[Detection]: ...
```

| 约定 | 要求 |
|---|---|
| `frame` | `numpy.ndarray`，**BGR**，`uint8`，形状 `(H, W, 3)`，就是原视频帧，未缩放 |
| 坐标 | 原图像素坐标 `x1, y1, x2, y2`；不要返回归一化坐标 |
| `score` | 0–1 的置信度；会对下游的跟踪门限产生影响 |
| `class_id` | 0–6，对应 `aeb.types.AEB_CLASS_NAMES`（person / rider / car / truck / bus / bike / motor） |
| 返回值 | `list[Detection]`；**允许返回空列表**（这一帧没有目标），也允许返回多个框 |
| 状态 | 检测器可以有自己的状态（例如帧缓冲），但不要假设帧索引连续 |

`Detection` 是一个 dataclass：

```python
Detection(x1=..., y1=..., x2=..., y2=..., score=..., class_id=...)
```

## 接到链路上

```python
from aeb.config import AEBConfig
from aeb.pipeline import AEBPipeline

cfg = AEBConfig(fps=30.0)  # 非视频源必须显式给帧率
cfg.ego_speed_mps = 12.0
pipe = AEBPipeline(cfg, MyDetector())

for i, frame in enumerate(frames):
    result = pipe.process(frame, i)
    print(i, result.global_level.name)
```

`AEBPipeline` 之后什么都不用改：跟踪、路径筛选、测距、双源 TTC、状态机都工作在同一套框上。

## 最小可运行例子

`examples/custom_detector.py` 是一个**不依赖任何模型**的实现（亮块阈值 + 最大外接矩形），
可以直接跑：

```bash
python examples/custom_detector.py            # 合成逼近目标，跑到四级风险
python examples/custom_detector.py --video v.mp4
```

它只用了 30 行，能跑通完整链路——这正是"检测器无关"的可验证含义。

## 常见错误

1. **在检测器里缩放画面**：坐标必须是原图坐标；缩放后要自己换算回来，否则测距与路径筛选都会错。
2. **返回归一化坐标**：链路的测距公式按像素计算，归一化坐标会得到量级错误的距离。
3. **忘记 `class_id`**：7 类之外的下标会被当成人/车之外的目标，跟踪仍然工作，但语义错。
4. **空列表当成错误**：返回 `[]` 是合法且常见的（无目标帧）。
5. **假设只调用一次**：同一个实例会被反复调用，缓存要自己管理。

## 与 CLI 的关系

命令行只内置 `--detector yolo|dfine`。自定义检测器目前走 Python API（如上）。
若要加进 CLI，最小改法是在 `aeb/cli.py` 的 `run_video` 里按名字构造你的类——
注意保持"只在被选中时才 import"，避免默认安装拉入额外依赖。
