"""Capture checks in scripts/run_colmap.py: settings summary, blur and static-frame flags."""

import json

import numpy as np
from PIL import Image, ImageFilter

from scripts import run_colmap as rc


def _photo(path, exposure, iso, pattern):
    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    sub[0x829A] = exposure
    sub[0x8827] = iso
    Image.fromarray(pattern).save(path, exif=exif.tobytes())


def _checkerboard(n=256, square=8):
    tile = (np.indices((n, n)) // square).sum(0) % 2
    return (np.stack([tile] * 3, -1) * 255).astype(np.uint8)


def test_flag_blurry_and_static():
    scores = {"a.jpg": 400.0, "b.jpg": 380.0, "c.jpg": 90.0, "d.jpg": 410.0, "e.jpg": 150.0}
    assert rc.flag_blurry(scores) == ["c.jpg", "e.jpg"]
    assert rc.flag_blurry({}) == []
    assert rc.flag_static([("f2", 30.0), ("f3", 0.4), ("f4", 12.0)]) == ["f3"]


def test_capture_settings_summary(tmp_path):
    board = _checkerboard()
    for i, (t, iso) in enumerate([(1 / 60, 200), (1 / 60, 200), (1 / 40, 160)]):
        _photo(tmp_path / f"IMG_{i}.jpg", t, iso, board)
    settings = rc.capture_settings(sorted(tmp_path.glob("*.jpg")))
    assert settings["ExposureTime"] == {"1/60 s": 2, "1/40 s": 1}
    assert settings["ISO"] == {"200": 2, "160": 1}
    assert settings["LensModel"] == {"missing": 3}
    assert rc.varying_settings(settings) == ["ExposureTime", "ISO"]


def test_check_capture_flags_blurry_photo_and_static_frames(tmp_path):
    photos, inp = tmp_path / "photos", tmp_path / "input"
    photos.mkdir()
    inp.mkdir()
    board = _checkerboard()
    for i in range(4):
        img = Image.fromarray(np.roll(board, 3 * i, axis=1))
        if i == 2:
            img = img.filter(ImageFilter.GaussianBlur(4))
        img.save(inp / f"IMG_{i}.jpg", quality=95)
        _photo(photos / f"IMG_{i}.jpg", 1 / 60, 200, board)
    checks = rc.check_capture(str(photos), str(inp), is_video=False)
    assert checks["blurry_images"] == ["IMG_2.jpg"]
    assert rc.varying_settings(checks["capture_settings"]) == []
    json.dumps(checks)  # must fit in colmap_summary.json

    frames = tmp_path / "frames"
    frames.mkdir()
    for i, shift in enumerate([0, 40, 40, 80]):  # frame 2 repeats frame 1: the camera stopped
        Image.fromarray(np.roll(board, shift, axis=1)).save(frames / f"frame_{i:05d}.jpg", quality=95)
    checks = rc.check_capture("video.mp4", str(frames), is_video=True)
    assert checks["static_frames"] == ["frame_00002.jpg"]
    assert isinstance(checks["capture_settings"], str)
