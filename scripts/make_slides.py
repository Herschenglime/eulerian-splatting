#!/usr/bin/env python3
"""Two 16:9 slide images for the final presentation, from finished e2e runs (no GPU needed):

    results/slides/heldout_comparison.png  one held-out view per scene: real photo above the render
    results/slides/psnr_table.png          held-out PSNR / SSIM / LPIPS per scene, plus takeaways

    uv run python scripts/make_slides.py [--views nemesys_vid=5 office=1 ...] [--out results/slides]

Images are cut from each run's test_visualization_65000_*.png (the GT | render | depth figure that
training saves at the final evaluation). That figure shows up to 9 held-out views as a grid;
--views picks one per scene by its position in that grid (row-major, from 0). Metrics are the
"[ITER N] Evaluating test/train" lines of the run's SLURM logs: test = mean over every held-out
view, train = mean over 5 fixed training views. Scene descriptions below are editorial.
"""

import argparse
import glob
import json
import math
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Display order (both slides): the two NemeSys captures side by side, then the rest.
# name: (subject, capture kind, capture description for the table, default view in the test grid)
SCENES = {
    "nemesys_vid": ("NemeSys robot", "video", "Video", 4),
    "nemesys_photos": ("NemeSys robot", "photos", "Photos", 4),
    "office": ("Office", "video", "Video, inside-out", 1),
    "testudo": ("Testudo statue", "photos", "Photos, outdoor", 2),
    "gator_statue": ("Gator statue", "photos", "Photos", 5),
}
TAKEAWAYS = [
    ("More overlapping views win.", "The two videos lead; the two smallest photo sets trail and overfit "
     "(their training views score 9–12 dB higher)."),
    ("The error is in the background.", "Every object reconstructs well; backgrounds seen from few views "
     "smear into streaks. The NemeSys photos are wide shots full of lab; the video frames are close-ups."),
    ("PSNR is not perceived quality.", "The in-focus NemeSys photos are sharper and higher-contrast than the "
     "video frames and arguably look better, yet score 5 dB lower: PSNR punishes fine detail that is "
     "slightly off more than it punishes blur."),
    ("Testudo: inconsistent photos.", "Auto exposure swung from 1/61 to 1/1167 s and "
     "people walked through, so even training views only reach 24 dB."),
    ("Refinement sharpens but overfits.", "60K → 65K lowers LPIPS in every scene, yet costs held-out "
     "PSNR on the sparse photo captures."),
]

INK, MUTED, RULE, ACCENT, BG = "#1d1d1f", "#6e6e73", "#d2d2d7", "#0a5fb4", "white"
W, H = 1920, 1080  # layout units: pixels of a 1080p slide (saved at 3840x2160)
# The figure is a 13.33 x 7.5 in PowerPoint 16:9 slide, so font sizes are the template's points
# when the image fills the slide. Arial as in our template; Liberation Sans has Arial's metrics.
SLIDE_IN = (40 / 3, 7.5)
HEADING_PT = 25
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


def find_run(name):
    runs = sorted(glob.glob(os.path.join(REPO, "output", name, "*", "test_visualization_65000_*.png")))
    if not runs:
        raise SystemExit(f"no finished run with test_visualization_65000 for {name}")
    return os.path.dirname(runs[-1])


def log_metrics(run):
    """{(iteration, 'test'|'train'): {psnr, ssim, lpips}} from every SLURM log that mentions the run."""
    rel = os.path.relpath(run, REPO)
    out = {}
    for log in sorted(glob.glob(os.path.join(REPO, "logs", "*.log"))):
        text = open(log, errors="replace").read()
        if rel not in text:
            continue
        for m in re.finditer(r"\[ITER (\d+)\] Evaluating (test|train): L1 \S+ PSNR (\S+) SSIM (\S+) LPIPS (\S+)", text):
            out[(int(m[1]), m[2])] = {"psnr": float(m[3]), "ssim": float(m[4]), "lpips": float(m[5])}
    return out


def _runs_of_true(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def test_tiles(run, n_test):
    """(gt, render) tile arrays cut from test_visualization_65000_*.png (see utils/viz_utils.py)."""
    png = glob.glob(os.path.join(run, "test_visualization_65000_*.png"))[0]
    img = np.asarray(Image.open(png).convert("RGB"))
    nonwhite = (img < 245).any(-1)
    r0, r1 = max(_runs_of_true(nonwhite.sum(1) > img.shape[1] * 0.3), key=lambda r: r[1] - r[0])
    panels = [r for r in _runs_of_true(nonwhite[r0:r1].mean(0) > 0.5) if r[1] - r[0] > 50]
    if len(panels) != 3:  # bright content split a panel; every figure so far has this layout
        panels = [(10, 1157), (1172, 2319), (2333, 3480)]
    b = min(9, n_test)
    ncols = math.ceil(b ** 0.5)
    nrows = math.ceil(b / ncols)
    tiles = []
    for c0, c1 in panels[:2]:
        panel = img[r0:r1, c0:c1]
        th, tw = panel.shape[0] / nrows, panel.shape[1] / ncols
        tiles.append([panel[int(round(th * (i // ncols))):int(round(th * (i // ncols + 1))),
                            int(round(tw * (i % ncols))):int(round(tw * (i % ncols + 1)))] for i in range(b)])
    return list(zip(*tiles))


def scene_data(views):
    rows = []
    for name, (subject, kind, capture, default_view) in SCENES.items():
        run = find_run(name)
        n_all = len(json.load(open(os.path.join(run, "cameras.json"))))
        n_test = math.ceil(n_all / 8)  # holdout: every 8th image by filename
        m = log_metrics(run)
        view = views.get(name, default_view)
        rows.append(dict(name=name, label=f"{subject} ({kind})", subject=subject, kind=kind, capture=capture, run=run, n_all=n_all, n_test=n_test,
                         n_train=n_all - n_test, m=m, tile=test_tiles(run, n_test)[view], view=view))
    return rows


def _fig():
    fig = plt.figure(figsize=SLIDE_IN, dpi=W / SLIDE_IN[0], facecolor=BG)
    return fig


def _text(fig, x, y, s, size, **kw):
    """Text at layout pixel (x, y from the top)."""
    kw.setdefault("color", INK)
    kw.setdefault("va", "baseline")
    return fig.text(x / W, 1 - y / H, s, fontsize=size, **kw)


def comparison_slide(rows, path):
    fig = _fig()
    _text(fig, 60, 84, "Held-out views: real photo vs. render", HEADING_PT, weight="bold")

    gap, x0, right = 18, 110, 36
    aspects = [t[0].shape[1] / t[0].shape[0] for t in (r["tile"] for r in rows)]
    th = min((W - x0 - right - gap * (len(rows) - 1)) / sum(aspects), 380)
    top1 = 268
    top2 = top1 + th + 14
    _text(fig, 76, top1 + th / 2, "Held-out photo", 13.5, color=MUTED, rotation=90, ha="center", va="center")
    _text(fig, 76, top2 + th / 2, "Render", 13.5, color=MUTED, rotation=90, ha="center", va="center")
    x = x0
    for r, a in zip(rows, aspects):
        tw = th * a
        test = r["m"][(65000, "test")]
        cx = x + tw / 2
        _text(fig, cx, top1 - 76, r["subject"], 13.5, weight="bold", ha="center")
        _text(fig, cx, top1 - 48, f"{r['kind']}, {r['n_train']} views", 11.5, color=MUTED, ha="center")
        _text(fig, cx, top1 - 14, f"{test['psnr']:.1f} dB", 15, color=ACCENT, weight="bold", ha="center")
        for img, top in zip(r["tile"], (top1, top2)):
            ax = fig.add_axes([x / W, 1 - (top + th) / H, tw / W, th / H])
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
        x += tw + gap
    fig.savefig(path, dpi=3840 / SLIDE_IN[0], facecolor=BG)
    plt.close(fig)


def table_slide(rows, path):
    fig = _fig()
    _text(fig, 60, 84, "Held-out image quality per scene", HEADING_PT, weight="bold")
    _text(fig, 60, 126, "Mean over every held-out view at 65K iterations (after refinement). "
          "PSNR and SSIM: higher is better. LPIPS: lower is better.", 12.5, color=MUTED)

    cols = [  # (header, x, align)
        ("Scene", 60, "left"), ("Capture", 430, "left"), ("Train / test views", 900, "right"),
        ("Held-out PSNR", 1110, "right"), ("60K → 65K", 1275, "right"), ("SSIM", 1395, "right"),
        ("LPIPS", 1505, "right"), ("Train PSNR", 1690, "right"), ("Gap", 1860, "right"),
    ]
    y = 190
    for head, x, ha in cols:
        _text(fig, x, y, head, 11, color=MUTED, ha=ha, weight="bold")
    fig.add_artist(plt.Line2D([60 / W, 1860 / W], [1 - (y + 16) / H] * 2, color=INK, lw=1.2))
    y += 62
    for r in rows:
        t60, t65, tr = r["m"][(60000, "test")], r["m"][(65000, "test")], r["m"][(65000, "train")]
        d = t65["psnr"] - t60["psnr"]
        cells = [r["label"], r["capture"], f"{r['n_train']} / {r['n_test']}", f"{t65['psnr']:.1f} dB",
                 f"{d:+.2f}", f"{t65['ssim']:.2f}", f"{t65['lpips']:.2f}", f"{tr['psnr']:.1f} dB",
                 f"{tr['psnr'] - t65['psnr']:.1f} dB"]
        for (head, x, ha), s in zip(cols, cells):
            kw = {"weight": "bold", "color": ACCENT} if head == "Held-out PSNR" else {}
            if head == "60K → 65K":
                kw = {"color": "#1a7f37" if d > 0.05 else "#c4314b" if d < -0.05 else MUTED}
            _text(fig, x, y, s, 13, ha=ha, **kw)
        fig.add_artist(plt.Line2D([60 / W, 1860 / W], [1 - (y + 20) / H] * 2, color=RULE, lw=0.8))
        y += 58

    y += 50
    _text(fig, 60, y, "Why they differ", 16, weight="bold")
    y += 44
    for head, body in TAKEAWAYS:
        _text(fig, 60, y, head, 12.5, weight="bold")
        lines = textwrap.wrap(body, 112)
        for i, line in enumerate(lines):
            _text(fig, 560, y + 29 * i, line, 12.5)
        y += 29 * len(lines) + 14
    fig.savefig(path, dpi=3840 / SLIDE_IN[0], facecolor=BG)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", nargs="*", default=[], metavar="NAME=I", help="test-grid position per scene")
    p.add_argument("--out", default=os.path.join(REPO, "results", "slides"))
    args = p.parse_args(argv)
    views = {k: int(v) for k, v in (s.split("=") for s in args.views)}
    os.makedirs(args.out, exist_ok=True)
    rows = scene_data(views)
    for r in rows:
        m = r["m"]
        print(f"{r['name']:15s} view {r['view']}  test PSNR 60K {m[(60000, 'test')]['psnr']:.2f} "
              f"65K {m[(65000, 'test')]['psnr']:.2f}  train 65K {m[(65000, 'train')]['psnr']:.2f}  "
              f"{os.path.relpath(r['run'], REPO)}")
    comparison_slide(rows, os.path.join(args.out, "heldout_comparison.png"))
    table_slide(rows, os.path.join(args.out, "psnr_table.png"))
    print("wrote", os.path.join(args.out, "heldout_comparison.png"), "and psnr_table.png")


if __name__ == "__main__":
    main()
