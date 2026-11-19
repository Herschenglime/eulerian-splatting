#!/usr/bin/env python3
"""Side-by-side held-out views: real photo | 3DGS | Eulerian, from render.py + metrics.py outputs.

    uv run python scripts/compare_3dgs.py <eulerian_run> <ours_iter> <3dgs_model> <3dgs_iter> out.png

Both renders/test/ours_N/ folders hold the same held-out views in the same order (both loaders sort
images by name and hold out every 8th). Picks the view where each method beats the other by the most,
plus the two closest to the median difference.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


def load(model, it):
    res = json.load(open(os.path.join(model, "results.json")))[f"ours_{it}"]
    per = json.load(open(os.path.join(model, "per_view.json")))[f"ours_{it}"]
    d = os.path.join(model, "renders", "test", f"ours_{it}")  # ours; the original 3DGS uses <model>/test/
    return res, per, d if os.path.isdir(d) else os.path.join(model, "test", f"ours_{it}")


def main(ours_model, ours_it, gs_model, gs_it, out):
    ours, ours_pv, ours_dir = load(ours_model, ours_it)
    gs, gs_pv, gs_dir = load(gs_model, gs_it)
    names = sorted(ours_pv["PSNR"])
    diff = {n: ours_pv["PSNR"][n] - gs_pv["PSNR"][n] for n in names}
    by = sorted(names, key=diff.get)
    mid = len(by) // 2
    picks = list(dict.fromkeys([by[0], by[-1], by[mid], by[mid - 1]]))[:4]

    # 16:9 slide (13.33 x 7.5 in, so font sizes are slide points): two rows of two triplets.
    W, H = 1920, 1080  # layout units
    fig = plt.figure(figsize=(40 / 3, 7.5), dpi=W / (40 / 3), facecolor="white")

    def text(x, y, s, size, **kw):
        fig.text(x / W, 1 - y / H, s, fontsize=size, color=kw.pop("color", "#1d1d1f"), **kw)

    text(60, 84, "Testudo: 3DGS vs. Eulerian Splatting on held-out photos", 25, weight="bold")
    text(60, 124, f"Mean over all {len(names)} held-out photos:  3DGS {gs['PSNR']:.2f} dB, SSIM {gs['SSIM']:.2f}, "
         f"LPIPS {gs['LPIPS']:.2f}   ·   Eulerian {ours['PSNR']:.2f} dB, SSIM {ours['SSIM']:.2f}, "
         f"LPIPS {ours['LPIPS']:.2f}", 12.5, color="#6e6e73")
    w0, h0 = Image.open(os.path.join(ours_dir, "gt", picks[0])).size
    inner, between, left = 10, 50, 60
    tw = (W - 2 * left - 4 * inner - between) / 6
    th = tw * h0 / w0
    for k, n in enumerate(picks[:4]):
        row, col = divmod(k, 2)
        top = 190 + row * (th + 70)
        tiles = [(os.path.join(ours_dir, "gt", n), "Held-out photo"),
                 (os.path.join(gs_dir, "renders", n), f"3DGS  {gs_pv['PSNR'][n]:.1f} dB"),
                 (os.path.join(ours_dir, "renders", n), f"Eulerian  {ours_pv['PSNR'][n]:.1f} dB")]
        for j, (path, label) in enumerate(tiles):
            x = left + col * (3 * tw + 2 * inner + between) + j * (tw + inner)
            ax = fig.add_axes([x / W, 1 - (top + th) / H, tw / W, th / H])
            ax.imshow(Image.open(path), interpolation="lanczos")
            ax.set_axis_off()
            text(x + tw / 2, top - 10, label, 12, ha="center", weight="bold" if j else "normal")
    fig.savefig(out, dpi=3840 / (40 / 3), facecolor="white")
    print("views:", {n: round(diff[n], 2) for n in picks}, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:6])
