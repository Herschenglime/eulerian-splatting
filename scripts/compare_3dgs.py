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

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3.9 * len(picks)), facecolor="white")
    for row, n in zip(axes, picks):
        tiles = [(os.path.join(ours_dir, "gt", n), "Held-out photo"),
                 (os.path.join(gs_dir, "renders", n), f"3DGS  {gs_pv['PSNR'][n]:.1f} dB"),
                 (os.path.join(ours_dir, "renders", n), f"Eulerian  {ours_pv['PSNR'][n]:.1f} dB")]
        for ax, (path, label) in zip(row, tiles):
            ax.imshow(Image.open(path))
            ax.set_title(label, fontsize=12)
            ax.set_axis_off()
    fig.suptitle(f"Testudo, held-out views (mean over all {len(names)}):\n"
                 f"3DGS {gs['PSNR']:.2f} dB / SSIM {gs['SSIM']:.3f} / LPIPS {gs['LPIPS']:.3f}    "
                 f"Eulerian {ours['PSNR']:.2f} dB / SSIM {ours['SSIM']:.3f} / LPIPS {ours['LPIPS']:.3f}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.9 * len(picks))))
    fig.savefig(out, dpi=150)
    print("views:", {n: round(diff[n], 2) for n in picks}, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:6])
