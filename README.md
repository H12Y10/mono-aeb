# mono-aeb

检测器无关的单目 **AEB（自动紧急制动）/ FCW（前向碰撞预警）** 决策链：检测 → 跟踪 → In-Path 过滤 → 测距 → TTC → 风险决策。

A detector-agnostic monocular AEB/FCW pipeline: detection → tracking → in-path filtering → ranging → TTC → risk decision.

核心优势是**免标定尺度 TTC**（只依赖目标的图像尺寸变化率，不依赖相机内参/畸变标定），以及**速度自适应的风险阈值**（AEB 触发点随车速平移）。检测器可插拔：默认后端 YOLOv8，可选高精度后端 D-FINE，二者实现同一个 `BaseDetector` 接口，决策链只消费 `[x1, y1, x2, y2, score, class]`。

---

## 特性

- **检测器无关契约**：决策链与具体检测器解耦，7 类 AEB 目标（person / rider / car / truck / bus / bike / motor）。
- **免标定尺度 TTC**：`τ = s / ṡ`，对 `(t, 1/s)` 做 Theil-Sen 稳健拟合，无需相机标定。
- **距离 TTC + 融合**：地面平面测距反推接近速度，与尺度 TTC 取最小值，`persist_k` 帧保持抑制抖动。
- **速度自适应阈值**：`τ_AEB(v) = t_react + v / (2·a_max)`，FCW / ATTENTION 在此之上加固定前置余量，整条决策梯度随车速平移。
- **逐视频自标定**：用 GPS ego 速度 + 免标定尺度 TTC 联合反解测距尺度 `f·H` 与地平线 `horizon_y`，消除 BDD100K 逐视频相机参数不一致的问题。
- **内置 ByteTrack**：跟踪器已 vendored（`third_party/ByteTrack`），无需额外安装。

## 决策链

```
视频帧 → 检测(YOLOv8 / D-FINE)
       → 跟踪(ByteTrack)
       → In-Path 过滤(固定梯形 ROI，自车路径走廊投影)
       → 测距(地面平面模型，D = f·H / (y_bottom − horizon))
       → TTC(尺度 τ 与距离 τ 融合，取 min)
       → 决策(状态机 NORMAL → ATTENTION → FCW → AEB_WARNING)
```

风险状态机仅在目标位于自车路径走廊内（`in_path = True`）时参与升级；路径外目标保持 `NORMAL`，避免旁车道车辆误触发。

---

## 安装

需要 Python ≥ 3.10。

```bash
pip install -e .
```

运行依赖：

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| `numpy` / `scipy` | 数值计算 / Theil-Sen 拟合 | scipy 同时是 ByteTrack 的传递依赖 |
| `opencv-python` | 图像读写、ROI、可视化 | |
| `lap` | ByteTrack 匹配（线性指派） | 见下方说明 |
| `ultralytics` | 默认检测后端 YOLOv8 | |

**关于 `lap`**：`lap` 是 ByteTrack 匹配的线性指派求解器，个别平台缺少对应 wheel。此时可改用 `lapx`（提供同名的 `lapjv` 接口）：

```bash
pip install lapx
```

代码已在 `third_party/ByteTrack/yolox/tracker/matching.py` 中内置兜底（`import lap` 失败时自动回退到 `lapx`）。

若 `pip install -e .` 因 `lap` 无对应 wheel 而中止，可先装 `lapx`，再跳过依赖解析安装本包：

```bash
pip install lapx
pip install -e . --no-deps
```

**可选高精度后端 D-FINE** 需额外安装 PyTorch 生态：

```bash
pip install -e ".[dfine]"   # torch / torchvision / pillow
```

**关于 Windows**：torch 依赖 Microsoft Visual C++ 运行库，个别机器（尤其机房 / 纯净环境）未安装时会出现
`OSError: [WinError 126] ... c10.dll` 加载失败。装一次 [VC++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) 即可。
（`--demo` 零数据演示不 import torch，无需此项。）

---

## 快速开始

### 零数据 demo（无需视频 / 权重）

```bash
python examples/quickstart.py --demo
```

该命令用合成检测器生成一个匀速逼近的目标，跑通「跟踪 → 测距 → TTC → 决策」全链，逐帧打印状态变化：

```
[quickstart] 自车速度 = 12 m/s，AEB 阈值 = 1.37s
[quickstart] 目标以 10 m/s 匀速逼近，逐帧打印状态变化：
  frame   0: NORMAL     d= 45.0m  ttc=  infs
  frame  39: ATTENTION  d= 32.0m  ttc= 3.36s
  frame  69: FCW        d= 22.0m  ttc= 2.36s
  frame  99: AEB        d= 12.1m  ttc= 1.36s
[quickstart] 完成，共 120 帧
```

可用参数：`--out 视频.mp4`（输出可视化视频）、`--show`（实时窗口）、`--frames` / `--fps` / `--v-close`。`--demo` 模式**不需要** `ultralytics` / `torch`，仅依赖 numpy / scipy / opencv-python / lap。

依赖自检：

```bash
python examples/quickstart.py --check
```

### 真实视频

```bash
python examples/demo_aeb.py --source 视频.mp4 --coco --out out.mp4
```

- `--coco`：权重为 COCO 预训练（如 `yolov8n.pt`）时做 COCO → AEB 7 类映射；使用 AEB 精调权重时不加该参数。
- `--ego-speed`：自车速度 (m/s)；缺省时自动从 BDD100K `samples-1k/info/*.json` 读 GPS 速度，无数据则回退 `12 m/s`。
- `--detector dfine`：切换到 D-FINE 后端（见下）。

也可传入图片目录 `--source ./frames_dir`。

---

## 检测器后端

### 默认：YOLOv8

```python
from aeb.detectors import YOLOv8Detector, COCO_TO_AEB

det = YOLOv8Detector("yolov8n.pt", conf=0.25, device="0",
                     class_map=COCO_TO_AEB)  # 精调权重则 class_map=None
```

### 可选高精度后端：D-FINE

D-FINE 依赖其上游仓库，通过环境变量定位，避免硬编码本地路径：

| 环境变量 | 含义 |
| --- | --- |
| `DFINE_ROOT` | D-FINE 上游仓库根目录（内含 `src/`、`configs/`） |
| `MONO_AEB_DFINE_WEIGHTS` | 微调权重路径（`.pth`） |

```bash
export DFINE_ROOT=/path/to/D-FINE
export MONO_AEB_DFINE_WEIGHTS=/path/to/best_stg1.pth
python examples/demo_aeb.py --source 视频.mp4 --detector dfine
```

也可在代码中直接传参：

```python
from aeb.detectors import DFineDetector
det = DFineDetector(weights="/path/to/best_stg1.pth")  # config 缺省用 dfine_hgnetv2_m_aeb.yml
```

> ⚠️ **注意**：D-FINE 的微调权重基于 BDD100K 训练，受该数据许可约束（见下），不随本仓库公开。

---

## 权重与数据

> ⚠️ **合规提示**：本仓库不包含任何模型权重与 BDD100K 原始视频/标注。

- **COCO 预训练 YOLOv8 权重**（如 `yolov8n.pt`）由 ultralytics 提供，按 ultralytics 的许可获取。
- **AEB 精调权重**（YOLOv8 或 D-FINE）基于 BDD100K 微调，而 BDD100K 许可为「仅学术 / 非商业使用、禁止再分发原始数据」。因此这些权重不随本仓库公开，走受限分发（申请获取）。需要者在取得合法数据与许可后，可用本仓库的训练配置自行复现。

---

## 项目结构

```
mono-aeb/
├── aeb/
│   ├── config.py           # 相机 / 自车路径 / 风险阈值 / 检测跟踪参数
│   ├── types.py            # Detection / Track / RiskFeature / RiskLevel 契约
│   ├── pipeline.py         # AEBPipeline 主链路
│   ├── calibration.py      # 逐视频自标定（f·H 与 horizon 联合反解）
│   ├── ego_speed.py        # 从 BDD100K info/*.json 读 GPS 速度
│   ├── detectors/          # BaseDetector + YOLOv8 + D-FINE
│   ├── tracker/            # ByteTrack 适配 + cython_bbox numpy shim
│   ├── distance/           # 地面平面测距
│   ├── ttc/                # 尺度 TTC / 距离 TTC / 融合 / 轨迹历史
│   ├── in_path/            # 固定梯形 ROI（自车路径走廊）
│   └── decision/           # 风险状态机
├── examples/
│   ├── quickstart.py       # 一键 demo（--demo / --check）
│   ├── demo_aeb.py         # 真实视频离线 demo
│   └── calibrate_video.py  # 逐视频自标定 CLI
├── third_party/ByteTrack/  # vendored ByteTrack（MIT，最小子集）
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 核心原理

### 测距（地面平面模型）

假设路面为地平面，相机离地高度 `H`、焦距 `f`（二者仅以乘积 `f·H` 进入测距，存在尺度简并）：

$$
D = \frac{f \cdot H}{y_{bottom} - y_{horizon}}
$$

`f·H` 与 `horizon_y` 随视频标定（见下），不依赖全局内参。

### 免标定尺度 TTC

对目标的图像尺寸 `s`（框高或框宽），其随时间的变化率与接近速度相关，不依赖相机标定：

$$
\tau_{scale} = \frac{s}{\dot{s}}
$$

实现上对 `(t, 1/s)` 做 Theil-Sen 稳健线性拟合，得斜率 `b`，则 `τ_scale = −1 / (b · s_now)`。Theil-Sen 对约 50% 离群不敏感，优于普通最小二乘。

### 距离 TTC 与融合

由测距序列 `(t, D)` 拟合接近速度 `v_close = −slope`，`τ_distance = D / v_close`。二者融合取更保守者：

$$
\tau_{fused} = \min(\tau_{scale}, \tau_{distance})
$$

并加 `persist_k` 帧保持，抑制单帧抖动。

### 风险状态机

```
NORMAL(0) → ATTENTION(1) → FCW(2) → AEB_WARNING(3)
```

AEB 触发阈值随自车速度自适应（`t_react` 反应时间、`a_max` 紧急制动最大减速度）：

$$
\tau_{AEB}(v) = t_{react} + \frac{v}{2 \cdot a_{max}}
$$

FCW / ATTENTION 在此之上加固定前置余量（默认 `+1.0s` / `+2.0s`）。升级立即、降级带迟滞（`hysteresis`），避免状态在阈值附近振荡。所需减速度（**预留，当前未接入状态机决策**）：

$$
a_{req} = \frac{v_{close}^2}{2 \cdot (D - d_{safe})}
$$

默认参数见 `aeb/config.py`（`t_react=0.5`、`a_max=6.86 m/s²`、`d_safe=5.0 m`、`history_len=10`、`persist_k=3`、`fps=30`）。

---

## 逐视频标定

BDD100K 由众包采集，相机逐视频不同，无统一内参。本仓库提供两种标定：

1. **ego 速度 + 尺度 TTC**（主入口）：对近似静止目标 `D = v_ego · τ_scale`，再对 `(1/D, y_bottom)` 做 Theil-Sen 直线拟合，一次同时解出斜率 `f·H` 与截距 `horizon_y`（`aeb/calibration.py: estimate_ground_plane`）。
2. **已知目标宽度（预留）**：`CameraConfig.calibrate_from_known_width()`（当前无调用点，仅作接口预留），由像素宽随 `(y_bottom − horizon)` 的变化反解相机高度。

CLI：

```bash
python examples/calibrate_video.py --stem 0571873b-faf718b2
# 或
python examples/calibrate_video.py --video /path/to/samples-1k/videos/xxxx.mov --max-frames 200
```

`--stem` 模式从 `BDD100K_VIDEO_DIR` 环境变量定位视频目录，缺省为当前目录。

---

## 许可

本项目代码以 **Apache-2.0** 许可发布，详见 [LICENSE](LICENSE)。

## 第三方依赖与致谢

| 组件 | 许可 | 用途 |
| --- | --- | --- |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | MIT | 多目标跟踪（已 vendored 最小子集于 `third_party/ByteTrack/`） |
| [D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 | 可选高精度检测后端 |
| [ultralytics (YOLOv8)](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | 默认检测后端 |
| [BDD100K](https://bdd-data.berkeley.edu/) | 仅学术 / 非商业、禁止再分发 | 训练数据（权重受限分发） |

> ⚠️ **许可提示**：`ultralytics` 采用 AGPL-3.0，对衍生作品与网络服务场景存在 copyleft 要求。若你的分发场景需避免这些义务，可改用 Apache-2.0 许可的 D-FINE 作为检测后端（`--detector dfine`）。本仓库自身代码以 Apache-2.0 发布。
