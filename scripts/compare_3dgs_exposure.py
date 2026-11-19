#!/usr/bin/env python3
"""16:9 slide: held-out photo | 3DGS | 3DGS + exposure compensation | Eulerian, on the right half of each
held-out photo (3DGS's --train_test_exp fits each test photo's exposure on its left half and renders and
scores only the right half, so every method is cropped and scored the same way here).

    uv run python scripts/compare_3dgs_exposure.py <eulerian_run> <ours_iter> <3dgs> <3dgs_exp> <3dgs_iter> out.png

PSNR is recomputed here from the saved PNGs, right halves only, averaged per image like metrics.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


def folder(model, it):
    d = os.path.join(model, "renders", "test", f"ours_{it}")  # ours; the original 3DGS uses <model>/test/
    return d if os.path.isdir(d) else os.path.join(model, "test", f"ours_{it}")


def right_half(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    return a[:, a.shape[1] // 2:] if a.shape[1] > a.shape[0] * 0.6 else a  # exposure renders are already halves


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return 10 * np.log10(1 / mse)


def main(ours_model, ours_it, gs_model, exp_model, gs_it, out):
    dirs = {"3DGS": folder(gs_model, gs_it), "3DGS + exposure": folder(exp_model, gs_it),
            "Eulerian": folder(ours_model, ours_it)}
    gt_dir = os.path.join(dirs["Eulerian"], "gt")
    names = sorted(os.listdir(gt_dir))
    gt = {n: right_half(os.path.join(gt_dir, n)) for n in names}
    imgs = {m: {n: right_half(os.path.join(d, "renders", n)) for n in names} for m, d in dirs.items()}
    for m in imgs:  # the exposure run's GT halves can differ by a pixel in width; align on the right edge
        for n in names:
            w = min(imgs[m][n].shape[1], gt[n].shape[1])
            imgs[m][n] = imgs[m][n][:, -w:]
    p = {m: {n: psnr(imgs[m][n], gt[n][:, -imgs[m][n].shape[1]:]) for n in names} for m in imgs}
    mean = {m: np.mean(list(v.values())) for m, v in p.items()}
    gain = {n: p["3DGS + exposure"][n] - p["3DGS"][n] for n in names}
    by = sorted(names, key=gain.get)
    best_gs = max(names, key=lambda n: p["3DGS"][n] - p["Eulerian"][n])
    best_ours = max(names, key=lambda n: p["Eulerian"][n] - p["3DGS"][n])
    picks = list(dict.fromkeys([by[-1], best_gs, best_ours, by[len(by) // 2], by[len(by) // 2 - 1]]))[:4]

    W, H = 1920, 1080
    fig = plt.figure(figsize=(40 / 3, 7.5), dpi=W / (40 / 3), facecolor="white")

    def text(x, y, s, size, **kw):
        fig.text(x / W, 1 - y / H, s, fontsize=size, color=kw.pop("color", "#1d1d1f"), **kw)

    text(60, 84, "Exposure correction", 25, weight="bold")
    text(60, 124, f"Right half of each held-out photo, PSNR over all {len(names)}:  3DGS {mean['3DGS']:.2f} dB   ·   "
         f"3DGS + exposure {mean['3DGS + exposure']:.2f} dB   ·   Eulerian {mean['Eulerian']:.2f} dB", 12.5,
         color="#6e6e73")
    h0, w0 = gt[picks[0]].shape[:2]
    th = (H - 190 - 70 - 40) / 2
    tw = th * w0 / h0
    inner, between = 8, 60
    block = 4 * tw + 3 * inner
    left = (W - 2 * block - between) / 2
    for k, n in enumerate(picks):
        row, col = divmod(k, 2)
        top = 205 + row * (th + 70)
        tiles = [(gt[n], "Photo")] + [(imgs[m][n], f"{m}\n{p[m][n]:.1f} dB") for m in dirs]
        for j, (img, label) in enumerate(tiles):
            x = left + col * (block + between) + j * (tw + inner)
            ax = fig.add_axes([x / W, 1 - (top + th) / H, tw / W, th / H])
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
            text(x + tw / 2, top - 10, label, 11, ha="center", weight="bold" if j else "normal")
    fig.savefig(out, dpi=3840 / (40 / 3), facecolor="white")
    print("mean right-half PSNR:", {m: round(v, 2) for m, v in mean.items()})
    print("views:", {n: {m: round(p[m][n], 1) for m in p} for n in picks}, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:7])
