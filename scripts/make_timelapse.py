"""Assemble training snapshots into a timelapse GIF and a contact sheet.

Reads <model_path>/snapshots/iter_XXXXXX_view{k}_{centers,render}.png (written when
training with --snapshot_iterations) and writes, per view k:
    snapshots/timelapse_view{k}.gif    frames: [centers | render], labeled with the iteration
    snapshots/contact_view{k}.png      one row per iteration

Usage:
    python scripts/make_timelapse.py output/<scene>/<run> [--frame_ms 700]
"""

import argparse
import os
import re
import sys
from collections import defaultdict

from PIL import Image, ImageDraw

PATTERN = re.compile(r"iter_(\d+)_view(\d+)_(centers|render)\.png$")
LABEL_H = 22


def collect(snap_dir):
    """{view: {iteration: {"centers": path, "render": path}}}"""
    found = defaultdict(lambda: defaultdict(dict))
    for name in os.listdir(snap_dir):
        m = PATTERN.match(name)
        if m:
            it, view, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            found[view][it][kind] = os.path.join(snap_dir, name)
    return found


def compose_frame(paths, iteration):
    """Side-by-side [centers | render] with an iteration label strip on top."""
    tiles = [Image.open(paths[k]).convert("RGB") for k in ("centers", "render") if k in paths]
    w, h = sum(t.width for t in tiles), max(t.height for t in tiles)
    frame = Image.new("RGB", (w, h + LABEL_H), "black")
    x = 0
    for t in tiles:
        frame.paste(t, (x, LABEL_H))
        x += t.width
    draw = ImageDraw.Draw(frame)
    draw.text((6, 5), f"iteration {iteration:,}", fill="white")
    if len(tiles) == 2:
        draw.text((tiles[0].width + 6, 5), "render", fill="white")
        draw.text((tiles[0].width - 60, 5), "centers", fill="white")
    return frame


def build(model_path, frame_ms=700):
    snap_dir = os.path.join(model_path, "snapshots")
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"no snapshots/ folder in {model_path} (train with --snapshot_iterations)")
    found = collect(snap_dir)
    if not found:
        raise FileNotFoundError(f"no iter_*_view*_*.png files in {snap_dir}")

    written = []
    for view, by_iter in sorted(found.items()):
        frames = [compose_frame(by_iter[it], it) for it in sorted(by_iter)]
        gif = os.path.join(snap_dir, f"timelapse_view{view}.gif")
        # hold the last frame longer so the final result is readable
        durations = [frame_ms] * (len(frames) - 1) + [frame_ms * 3]
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0)

        sheet = Image.new("RGB", (max(f.width for f in frames), sum(f.height for f in frames)), "black")
        y = 0
        for f in frames:
            sheet.paste(f, (0, y))
            y += f.height
        png = os.path.join(snap_dir, f"contact_view{view}.png")
        sheet.save(png)
        written += [gif, png]
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path")
    parser.add_argument("--frame_ms", type=int, default=700)
    args = parser.parse_args(argv)
    try:
        for path in build(args.model_path, args.frame_ms):
            print("wrote", path)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
