#!/usr/bin/env python3
"""Copy one training run off HiPerGator and package it as a zip with a guide (README.md).

Run it on YOUR machine, from a clone of this repo (standard library only, plus `rsync` and `ssh`):

    python3 scripts/pull_results.py <user>@hpg.rc.ufl.edu output/testudo/20261117-133402
    #   (no system python, e.g. NixOS: uv run --no-project python scripts/pull_results.py ...)

One rsync connection (one Duo prompt) pulls, mirroring the repo layout so render.py works after
extracting the zip into a repo clone:
    output/<name>/<run>/   final splat, raw PLYs, snapshots + timelapse, test grids, configs
                           (not checkpoints/ or learned_grids/ unless asked)
    data/<name>/           the COLMAP dataset: images*/, sparse/, colmap_summary.json
    logs/                  the SLURM logs and GPU-memory logs of the jobs that wrote this run
Then it writes README.md (docs/results-guide.md plus this run's numbers) and zips the folder.

Options:
    --lite        skip the big files: raw point_cloud/ PLYs, snapshot PLYs, dataset images
                  (keeps the final splat, the timelapse, all PNGs, configs, sparse/)
    --grids       also pull learned_grids/ (the probability pyramid + attribute hash grids, ~4 GB)
    --checkpoint  also pull checkpoints/ (~12 GB, only useful to continue training)
    --dry_run     list what would be transferred, with sizes; copy nothing
    --no_zip      stop after copying and writing README.md
    --local       treat the source as a local repo path (no ssh). On HiPerGator this builds the
                  zip there, so only one file needs downloading. The staging folder is hard links
                  (no extra space); the zip itself is a full copy, so delete it once downloaded:
                    python3 scripts/pull_results.py --local . output/<name>/<run>
                    # then, on your machine (resumable):
                    rsync -P <user>@hpg.rc.ufl.edu:/blue/mdjahiduislam/<user>/eulerian-splatting/results/<name>_<run>.zip .
"""

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(REPO, "docs", "results-guide.md")
STORED = {".ply", ".pt", ".pth", ".png", ".gif", ".jpg", ".jpeg", ".mp4", ".bin", ".zip"}


def filter_rules(name, run, args):
    """rsync filter rules, relative to the remote repo root. First match wins."""
    out = f"/output/{name}/{run}"
    rules = ["+ /output/", f"+ /output/{name}/", f"+ {out}/"]
    if not args.checkpoint:
        rules.append(f"- {out}/checkpoints/")
    if not args.grids:
        rules.append(f"- {out}/learned_grids/")
    if args.lite:
        rules += [f"- {out}/point_cloud/", f"- {out}/snapshots/*.ply"]
    rules += [f"+ {out}/***", "+ /data/", f"+ /data/{name}/",
              f"- /data/{name}/input/", f"- /data/{name}/distorted/", f"- /data/{name}/point_cloud/"]
    if args.lite:
        rules.append(f"- /data/{name}/images*/")
    rules += [f"+ /data/{name}/***", "+ /logs/", "+ /logs/eus-*", "- *"]
    return rules


def rsync(src, dest, rules, dry_run, link_dest=None):
    cmd = ["rsync", "-a", "--prune-empty-dirs", "--human-readable"]
    if link_dest:  # local source on the same filesystem: hard links, no second copy of the data
        cmd.append(f"--link-dest={link_dest}")
    cmd += ["--dry-run", "--stats", "--out-format=%l %n"] if dry_run else ["--info=progress2,stats1"]
    for rule in rules:
        cmd += ["--filter", rule]
    cmd += [src, dest]
    print("$", " ".join(f"'{c}'" if " " in c or "*" in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def keep_run_logs(bundle, run_rel):
    """Keep only the logs of jobs that worked on this run (their log mentions the run folder)."""
    logs = sorted(glob.glob(os.path.join(bundle, "logs", "*.log")))
    kept = []
    for log in logs:
        with open(log, "rb") as f:
            mentions = run_rel.encode() in f.read()
        stem = os.path.splitext(log)[0]
        if mentions:
            kept.append(os.path.basename(log))
            continue
        for path in (log, stem + "_gpu.csv"):
            if os.path.exists(path):
                os.remove(path)
    for csv in glob.glob(os.path.join(bundle, "logs", "*_gpu.csv")):
        if not os.path.exists(csv[:-len("_gpu.csv")] + ".log"):
            os.remove(csv)
    return kept


def ply_count(path):
    try:
        with open(path, "rb") as f:
            for _ in range(200):
                line = f.readline()
                if line.startswith(b"element vertex"):
                    return int(line.split()[-1])
                if line.strip() == b"end_header":
                    break
    except OSError:
        pass
    return None


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def run_summary(bundle, name, run, kept_logs):
    """Markdown section with this run's facts, read from the pulled files."""
    out_dir = os.path.join(bundle, "output", name, run)
    lines = [f"## This run: `{name}` / `{run}`", ""]

    cfg_path = os.path.join(out_dir, "e2e_config.yaml")
    if os.path.exists(cfg_path):
        text = open(cfg_path).read()
        pick = {k: (re.search(rf"^\s*{k}:\s*(.+)$", text, re.M) or [None, "?"])[1].strip()
                for k in ("preset", "holdout")}
        lines.append(f"- Settings: preset `{pick['preset']}`, held-out views: `{pick['holdout']}` "
                     f"(everything in `output/{name}/{run}/e2e_config.yaml`)")

    summary_path = os.path.join(bundle, "data", name, "colmap_summary.json")
    if os.path.exists(summary_path):
        s = json.load(open(summary_path))
        cam = s.get("camera", {})
        lines.append(f"- COLMAP: {s.get('num_registered')}/{s.get('num_input_images')} images registered, "
                     f"{s.get('num_points3D', 0):,} points, mean reprojection error "
                     f"{s.get('mean_reprojection_error_px', float('nan')):.2f} px, "
                     f"undistorted size {cam.get('width')}x{cam.get('height')}")
        if s.get("blurry_images"):
            lines.append(f"- Flagged blurry: {len(s['blurry_images'])} images (see `colmap_summary.json`)")

    final = os.path.join(out_dir, f"{name}.ply")
    if os.path.exists(final):
        lines.append(f"- Final splat `{name}.ply`: {ply_count(final):,} Gaussians, {human(os.path.getsize(final))}")
    else:
        lines.append(f"- **No final splat `{name}.ply`: the run had not finished (or failed) when this was pulled.**")
    light = sorted(glob.glob(os.path.join(out_dir, "*_light*.ply")))
    for p in light:
        lines.append(f"- Lighter viewing copy `{os.path.basename(p)}`: {ply_count(p):,} Gaussians, "
                     f"{human(os.path.getsize(p))}")

    grids = sorted(glob.glob(os.path.join(out_dir, "test_visualization_*_for_PSNR=*.png")))
    if grids:
        lines.append("- Held-out PSNR during training (one random sample each; see the guide):")
        for g in grids:
            m = re.search(r"test_visualization_(\d+)_for_PSNR=([\d.]+)\.png", g)
            if m:
                lines.append(f"  - iteration {int(m[1]):,}: {float(m[2]):.2f} dB (`{os.path.basename(g)}`)")

    results = os.path.join(out_dir, "results.json")
    if os.path.exists(results):
        lines.append("- Scores from render.py + metrics.py (all held-out views):")
        for method, scores in json.load(open(results)).items():
            lines.append(f"  - {method}: " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items()))

    for log in kept_logs:
        text = open(os.path.join(bundle, "logs", log), errors="replace").read()
        bits = [m[0] for m in re.finditer(r"Finished in [\d.]+ min|peak GPU memory:\s*\d+ MiB", text)]
        if bits:
            lines.append(f"- `logs/{log}`: " + "; ".join(bits))
    lines += ["", f"Pulled {datetime.datetime.now():%Y-%m-%d %H:%M}.", ""]
    return "\n".join(lines)


def make_zip(bundle):
    path = bundle.rstrip("/") + ".zip"
    base = os.path.dirname(bundle.rstrip("/"))
    files = [os.path.join(d, f) for d, _, fs in os.walk(bundle) for f in fs]
    total = sum(os.path.getsize(f) for f in files)
    done = 0
    with zipfile.ZipFile(path, "w", allowZip64=True) as z:
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            z.write(f, os.path.relpath(f, base),
                    compress_type=zipfile.ZIP_STORED if ext in STORED else zipfile.ZIP_DEFLATED)
            done += os.path.getsize(f)
            print(f"\rzipping {human(done)} / {human(total)}", end="", flush=True)
    print(f"\nwrote {path} ({human(os.path.getsize(path))})")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="user@hpg.rc.ufl.edu, or a local repo path with --local")
    p.add_argument("run", help="output/<name>/<run>, relative to the repo")
    p.add_argument("--remote_repo", help="repo path on HiPerGator (default /blue/mdjahiduislam/<user>/eulerian-splatting)")
    p.add_argument("--dest", default=os.path.join(REPO, "results"), help="local folder (default results/, gitignored)")
    for flag in ("lite", "grids", "checkpoint", "dry_run", "no_zip", "local"):
        p.add_argument(f"--{flag}", action="store_true")
    args = p.parse_args(argv)

    parts = args.run.strip("/").split("/")
    if len(parts) != 3 or parts[0] != "output":
        p.error("run must look like output/<name>/<run>")
    _, name, run = parts
    if args.local:
        src = os.path.abspath(args.source).rstrip("/") + "/"
    else:
        user = args.source.split("@")[0] if "@" in args.source else os.environ.get("USER", "")
        repo = args.remote_repo or f"/blue/mdjahiduislam/{user}/eulerian-splatting"
        src = f"{args.source}:{repo.rstrip('/')}/"
    bundle = os.path.join(os.path.abspath(args.dest), f"{name}_{run}")
    os.makedirs(bundle, exist_ok=True)

    rsync(src, bundle + "/", filter_rules(name, run, args), args.dry_run,
          link_dest=src if args.local else None)
    if args.dry_run:
        return
    if not os.path.isdir(os.path.join(bundle, "output", name, run)):
        sys.exit(f"nothing pulled for {args.run}: check the run path and --remote_repo")

    kept = keep_run_logs(bundle, f"output/{name}/{run}")
    guide = open(GUIDE).read() if os.path.exists(GUIDE) else "# Results\n"
    with open(os.path.join(bundle, "README.md"), "w") as f:
        f.write(guide.replace("<name>", name).replace("<run>", run).rstrip() + "\n\n"
                + run_summary(bundle, name, run, kept))
    print(f"wrote {os.path.join(bundle, 'README.md')}")
    if not args.no_zip:
        make_zip(bundle)


if __name__ == "__main__":
    main()
