"""输入源解析：目录 / glob / 文本清单 / 单图，以及"没有固有帧率必须显式给"。

扩展前 `--source` 只认"图片目录或视频文件"，复现者手上常见的
「一批评测图放在若干子目录、由脚本生成清单」这类输入只能自己改写代码。
本文件把四种写法钉住，并保证顺序稳定（同一批帧不会两次跑出不同顺序）。
"""

import numpy as np
import pytest

from aeb.cli import _pick_fps, iter_source, resolve_images

cv2 = pytest.importorskip("cv2")


def _img(path, size=(48, 64)):
    h, w = size
    cv2.imwrite(str(path), np.full((h, w, 3), 80, np.uint8))


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _img(tmp_path / "a" / "02.png")
    _img(tmp_path / "a" / "01.png")
    _img(tmp_path / "b" / "01.jpg")
    return tmp_path


def test_directory_resolves_sorted(tree):
    paths = resolve_images(str(tree / "a"))
    assert [p.name for p in paths] == ["01.png", "02.png"]


def test_glob_resolves_across_subdirectories(tree):
    paths = resolve_images(str(tree / "**" / "*.png"))
    assert [p.name for p in paths] == ["01.png", "02.png"]


def test_text_list_supports_relative_paths_and_comments(tree):
    lst = tree / "frames.txt"
    lst.write_text(
        "# 这行是注释\n\nb/01.jpg\na/*.png\n",
        encoding="utf-8",
    )
    paths = resolve_images(str(lst))
    assert [str(p.relative_to(tree)).replace("\\", "/") for p in paths] == [
        "b/01.jpg",
        "a/01.png",
        "a/02.png",
    ]


def test_single_image_source(tree):
    paths = resolve_images(str(tree / "b" / "01.jpg"))
    assert len(paths) == 1


def test_video_like_source_returns_none(tree):
    assert resolve_images(str(tree / "not-a-video.mp4")) is None


def test_iter_source_yields_frames_for_glob(tree):
    frames = list(iter_source(str(tree / "a" / "*.png")))
    assert len(frames) == 2
    assert all(f.shape == (48, 64, 3) for f in frames)


def test_glob_without_fps_is_rejected():
    """非视频源没有固有帧率，必须显式给 --fps。"""
    with pytest.raises(SystemExit, match="frame rate"):
        _pick_fps(None, None, is_dir=True)
