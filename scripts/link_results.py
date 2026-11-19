#!/usr/bin/env python3
"""Shortcuts at the top of results/ to each run's main files, so they're one click away:

    python3 scripts/link_results.py [results_dir]        # default: results/ in this repo

A results package (scripts/pull_results.py) mirrors the repo, so its files sit 4 levels down:
    results/<name>_<run>/output/<name>/<run>/<name>.ply
This makes relative symlinks next to the packages, for the latest run of each <name>:
    <name>.ply              the final splat (standard 3DGS format, upright)
    <name>_orbit.mp4/.gif   the turntable from scripts/orbit_vkgs.py, if rendered
    <name>_<tag>.mp4/.gif   any other render in <run>/<tag>/ (scripts/room_views_vkgs.py: pan, path)
    <name>_timelapse.gif    training from iteration 1 (snapshots/timelapse_view0.gif)
    <name>_test_views.png   held-out photos vs. renders at the last evaluation, if holdout was on
    <name>_run/             the run folder itself, for everything else
Older runs with the same name get the same links with their run id: <name>_<run>.ply, etc.

The files the links point to carry the same names: a link dragged into an upload or a web page may
be resolved to the real file, and every run's timelapse is snapshots/timelapse_view0.gif. So a file
whose own name differs gets a hard link (a second name for the same file, no extra space) next to
it, e.g. snapshots/<name>_timelapse.gif, and the symlink points there. The original names stay,
because the package README and e2e.py --resume expect them. Build zips before running this (as
pull_results.py does), or a zip stores those files twice.

It runs by itself at the end of scripts/pull_results.py and scripts/orbit_vkgs.py. After copying
packages in some other way (Globus, unzipping by hand), run it once. Re-running is safe: it only
removes or replaces symlinks that point into a package's output/ folder, never real files.
Standard library only.
"""

import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_runs(results_dir):
    """(name, run, run_dir) for every run folder in the packages under results_dir."""
    runs = []
    for run_dir in glob.glob(os.path.join(results_dir, "*", "output", "*", "*")):
        if os.path.isdir(run_dir) and not os.path.islink(run_dir) and (
                os.path.exists(os.path.join(run_dir, "cameras.json"))
                or os.path.exists(os.path.join(run_dir, "cfg_args"))):
            name = os.path.basename(os.path.dirname(run_dir))
            runs.append((name, os.path.basename(run_dir), run_dir))
    return sorted(runs)


def last_test_grid(run_dir):
    """The test_visualization PNG of the highest iteration."""
    def iteration(path):
        m = re.search(r"test_visualization_(\d+)_", os.path.basename(path))
        return int(m.group(1)) if m else -1
    grids = glob.glob(os.path.join(run_dir, "test_visualization_*.png"))
    return max(grids, key=iteration) if grids else None


def first_existing(run_dir, *names):
    """The first of these paths (relative to run_dir) that exists; older renders used orbit/orbit.*."""
    for n in names:
        path = os.path.join(run_dir, n)
        if os.path.exists(path):
            return path
    return None


def targets(name, run_dir):
    """Link suffix -> file or folder in the run, for the ones that exist."""
    found = {
        ".ply": os.path.join(run_dir, f"{name}.ply"),
        "_orbit.mp4": first_existing(run_dir, f"orbit/{name}_orbit.mp4", "orbit/orbit.mp4"),
        "_orbit.gif": first_existing(run_dir, f"orbit/{name}_orbit.gif", "orbit/orbit.gif"),
        "_timelapse.gif": os.path.join(run_dir, "snapshots", "timelapse_view0.gif"),
        "_test_views.png": last_test_grid(run_dir),
        "_run": run_dir,
    }
    # Renders from orbit_vkgs.py / room_views_vkgs.py: <run>/<tag>/<name>_<tag>.mp4 and .gif
    for video in sorted(glob.glob(os.path.join(run_dir, "*", f"{name}_*.mp4"))
                        + glob.glob(os.path.join(run_dir, "*", f"{name}_*.gif"))):
        if os.path.basename(os.path.dirname(video)) != "snapshots":
            found.setdefault(os.path.basename(video)[len(name):], video)
    return {suffix: path for suffix, path in found.items() if path and os.path.exists(path)}


def run_named(target, want, ours):
    """A name for target that carries the run's name (e.g. snapshots/<name>_timelapse.gif next to
    snapshots/timelapse_view0.gif), as a hard link: the same file under a second name, no extra space.
    A link dragged into an upload or a web page may be resolved to the real file's name, so the
    real name must tell the runs apart too. Hard links we made earlier under another of `ours`
    (a run that has since become an older one) are removed. Returns the path to link to."""
    folder = os.path.dirname(target)
    want_path = os.path.join(folder, want)
    inode = os.stat(target).st_ino
    for name in set(ours) - {want, os.path.basename(target)}:
        stale = os.path.join(folder, name)
        if os.path.isfile(stale) and not os.path.islink(stale) and os.stat(stale).st_ino == inode:
            os.remove(stale)
    if os.path.basename(target) == want:
        return target
    if os.path.lexists(want_path):
        st = os.lstat(want_path)
        if st.st_ino == inode:
            return want_path
        if os.path.isfile(want_path) and not os.path.islink(want_path) and st.st_nlink > 1:
            os.remove(want_path)  # our hard link to an older file (e.g. an earlier test grid)
        else:
            print(f"kept {target}'s own name: {want_path} exists and isn't ours", file=sys.stderr)
            return target
    try:
        os.link(target, want_path)
    except OSError as e:  # filesystem without hard links
        print(f"kept {target}'s own name: can't hard-link ({e})", file=sys.stderr)
        return target
    return want_path


def is_ours(link, results_dir):
    """A symlink made by this script: it points into some package's output/ folder."""
    if not os.path.islink(link):
        return False
    target = os.path.normpath(os.path.join(results_dir, os.readlink(link)))
    rel = os.path.relpath(target, results_dir).split(os.sep)
    return len(rel) >= 3 and rel[0] not in ("..", ".") and rel[1] == "output"


def link_results(results_dir=None, quiet=False):
    results_dir = os.path.abspath(results_dir or os.path.join(REPO, "results"))
    if not os.path.isdir(results_dir):
        return []
    for entry in os.listdir(results_dir):
        path = os.path.join(results_dir, entry)
        if is_ours(path, results_dir):
            os.remove(path)

    by_name = {}
    for name, run, run_dir in find_runs(results_dir):
        by_name.setdefault(name, []).append((run, run_dir))

    made = []
    for name, runs in sorted(by_name.items()):
        runs.sort()  # run ids are timestamps (YYYYMMDD-HHMMSS): the last one is the latest
        for i, (run, run_dir) in enumerate(runs):
            prefix = name if i == len(runs) - 1 else f"{name}_{run}"
            for suffix, target in targets(name, run_dir).items():
                if os.path.isfile(target):
                    target = run_named(target, prefix + suffix, (name + suffix, f"{name}_{run}{suffix}"))
                link = os.path.join(results_dir, prefix + suffix)
                if os.path.lexists(link):  # a real file or folder of the same name: leave it alone
                    print(f"skipped {link}: something else already has that name", file=sys.stderr)
                    continue
                os.symlink(os.path.relpath(target, results_dir), link)
                made.append(link)
    if not quiet:
        print(f"{len(made)} shortcuts in {results_dir} for {sum(map(len, by_name.values()))} runs "
              f"({', '.join(sorted(by_name)) or 'none'})")
    return made


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")):
        sys.exit(__doc__)
    link_results(sys.argv[1] if len(sys.argv) == 2 else None)
