# 本目录是 ByteTrack（MIT License）的最小 vendor 子集，仅供 aeb.tracker 使用。
# 只保留 yolox/tracker 下的跟踪逻辑；原版顶层 __init__ 会 from .utils import
# configure_module（连带导入 torch/cv2 等重依赖），故此处替换为空实现。
