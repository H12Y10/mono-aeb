"""cython_bbox 的纯 numpy 替身。

Windows 上 `cython-bbox` 需要 MSVC 编译、常装不上。ByteTrack 的
matching.py 只用 `bbox_overlaps` 这一个函数，这里用 numpy 等价实现顶替，
不改动 ByteTrack 源码（保持 MIT 原样）。语义与 cython_bbox 一致。
"""

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """计算两集合 [x1, y1, x2, y2] 框之间的 IoU。

    :param boxes: (N, 4)
    :param query_boxes: (K, 4)
    :return: (N, K) IoU 矩阵
    """
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float64)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1)
            * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                        + box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps
