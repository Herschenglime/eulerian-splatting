"""Render a turntable orbit of an exported splat with NVIDIA's vk_gaussian_splatting viewer (vkgs),
then assemble an MP4 and a GIF with ffmpeg.

    python scripts/orbit_vkgs.py <run_dir> [--seconds_per_turn 12] [--size 1080 1080] [--elevation 20]

Spin speed is --seconds_per_turn (default 12 s, i.e. 30 degrees/s); the frame count follows from
it and --fps (12 s x 30 fps = 360 frames, about 1.5 min to render a 15M-Gaussian splat on 8 GB).

<run_dir> is an e2e run folder: it needs cameras.json, cfg_args and the exported <name>.ply.
Writes <run_dir>/orbit/: cameras_orbit.json, orbit.cfg, frames/, <name>_orbit.mp4, <name>_orbit.gif
(named after the run, so files from different runs stay distinct once copied or uploaded).

How the orbit is placed:
- The exported PLY lives in the training frame (cameras PCA-aligned, recentred and scaled; see
  utils/camera_utils.transform_poses_pca), rotated upright by utils/ply_export.UPRIGHT (up = -y).
  cameras.json holds the COLMAP poses, so they go through the same two transforms here.
- The orbit centre is the point closest to every training camera's optical axis, i.e. what the
  capture was looking at. The radius and height default to the training cameras' medians, so the
  orbit sweeps through the viewpoints the model was trained on.
- vkgs imports INRIA cameras.json presets (camera-to-world rotation, OpenCV axes), which is the
  format written here. It ignores fx/fy and renders at its own 60 degree field of view.

Installing vkgs (Linux, NVIDIA GPU; prebuilt, no Vulkan SDK needed): download
vkgs-<version>-linux-64bits.tar.gz from https://github.com/nvpro-samples/vk_gaussian_splatting/releases,
extract it, `chmod +x vk_gaussian_splatting`, and put it (or a wrapper script named vkgs) on PATH.
Otherwise pass --vkgs /path/to/vk_gaussian_splatting or set $VKGS. Tested with release 2026.2.12.
"""

import argparse
import ast
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scene  # noqa: E402,F401  -- import first: scene and utils.camera_utils import each other
del scene
from utils.camera_utils import transform_poses_pca  # noqa: E402
from utils.ply_export import UPRIGHT  # noqa: E402



def read_cfg_args(path):
    """cfg_args is a printed argparse Namespace: Namespace(a=1, b='x', ...)."""
    with open(path) as f:
        text = f.read().strip()
    call = ast.parse(text, mode="eval").body
    out = {}
    for kw in call.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:  # non-literal values (e.g. objects); not needed here
            pass
    return out


def load_cameras(run_dir):
    """All cameras of a run as camera-to-world (N, 4, 4) in the exported PLY's frame,
    plus a mask of the training cameras."""
    with open(os.path.join(run_dir, "cameras.json")) as f:
        cams = sorted(json.load(f), key=lambda c: c["img_name"])  # the loader sorts the same way
    cfg = read_cfg_args(os.path.join(run_dir, "cfg_args"))
    llffhold = cfg.get("llffhold", 8)
    train = np.array([not cfg.get("eval", False) or i % llffhold != 0 for i in range(len(cams))])

    c2w = np.tile(np.eye(4), (len(cams), 1, 1))
    for i, c in enumerate(cams):
        c2w[i, :3, :3] = np.array(c["rotation"])
        c2w[i, :3, 3] = np.array(c["position"])

    # Same normalisation as training (fit on training cameras only), then the export's rotation.
    _, norm = transform_poses_pca(c2w[train, :3, :4])
    out = norm["transform"] @ c2w
    out[:, :3, 3] *= norm["scale"]
    up4 = np.eye(4)
    up4[:3, :3] = UPRIGHT
    return up4 @ out, train, cams


def look_at_point(c2w):
    """Least-squares point closest to all optical axes (OpenCV: +z is forward)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for m in c2w:
        o, d = m[:3, 3], m[:3, 2] / np.linalg.norm(m[:3, 2])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def look_at(eye, target, down):
    """OpenCV camera-to-world rotation (x right, y down, z forward) looking from eye at target."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def orbit(c2w_train, frames, elevation_deg=None, distance=1.0, turns=1.0):
    up = np.array([0.0, -1.0, 0.0])  # exported PLYs: scene up is -y
    centre = look_at_point(c2w_train)
    offsets = c2w_train[:, :3, 3] - centre
    heights = offsets @ up
    horiz = offsets - np.outer(heights, up)
    radius = np.median(np.linalg.norm(offsets, axis=1)) * distance
    if elevation_deg is None:
        elevation = math.atan2(np.median(heights), np.median(np.linalg.norm(horiz, axis=1)))
    else:
        elevation = math.radians(elevation_deg)
    # Start at the first training camera's azimuth, so frame 0 is a familiar view.
    e1 = horiz[0] / np.linalg.norm(horiz[0])
    e2 = np.cross(up, e1)
    poses = []
    for k in range(frames):
        a = 2 * math.pi * turns * k / frames
        direction = math.cos(elevation) * (math.cos(a) * e1 + math.sin(a) * e2) + math.sin(elevation) * up
        eye = centre + radius * direction
        m = np.eye(4)
        m[:3, :3] = look_at(eye, centre, -up)
        m[:3, 3] = eye
        poses.append(m)
    return np.array(poses), centre, radius, math.degrees(elevation)


def inria_json(c2w, names, size, focal):
    return [{"id": i, "img_name": n, "width": size[0], "height": size[1],
             "position": m[:3, 3].tolist(), "rotation": m[:3, :3].tolist(), "fx": focal, "fy": focal}
            for i, (m, n) in enumerate(zip(c2w, names))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--ply", help="splat to render (default: <run_dir>/<name>.ply from e2e)")
    p.add_argument("--seconds_per_turn", type=float, default=12.0,
                   help="spin speed: seconds for one full turn (default 12, i.e. 30 degrees/s)")
    p.add_argument("--frames", type=int,
                   help="total frames, overriding seconds_per_turn (speed then = frames / fps per --turns)")
    p.add_argument("--fps", type=int, default=30, help="playback rate of the MP4")
    p.add_argument("--size", type=int, nargs=2, default=[1080, 1080], metavar=("W", "H"))
    p.add_argument("--elevation", type=float, help="degrees above the orbit centre (default: training cameras' median)")
    p.add_argument("--distance", type=float, default=1.0, help="orbit radius as a multiple of the cameras' median distance")
    p.add_argument("--turns", type=float, default=1.0)
    p.add_argument("--training_views", action="store_true",
                   help="render the training/test camera poses instead of an orbit (to check alignment)")
    p.add_argument("--shformat", type=int, default=1, help="vkgs SH storage: 0=fp32 1=fp16 (default, fits 15M on 8 GB) 2=uint8")
    p.add_argument("--gif_width", type=int, default=540)
    p.add_argument("--gif_fps", type=int, default=15)
    p.add_argument("--window", action="store_true", help="run vkgs with a visible window instead of headless")
    p.add_argument("--no_render", action="store_true", help="only write the cameras and the sequence file")
    p.add_argument("--vkgs", help="vk_gaussian_splatting executable (default: $VKGS, else vkgs or "
                                  "vk_gaussian_splatting on PATH)")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    name = os.path.basename(os.path.dirname(run_dir))  # output/<name>/<run>
    ply = args.ply or os.path.join(run_dir, f"{name}.ply")
    ply = os.path.abspath(ply)
    if not os.path.exists(ply):
        sys.exit(f"splat not found: {ply} (pass --ply)")

    c2w_all, train, cams = load_cameras(run_dir)
    out_dir = os.path.join(run_dir, "orbit_check" if args.training_views else "orbit")
    frames_dir = os.path.join(out_dir, "frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)

    focal = 0.5 * args.size[1] / math.tan(math.radians(30))  # vkgs renders at 60 degrees vertical
    if args.training_views:
        poses = c2w_all
        names = [c["img_name"] for c in cams]
    else:
        frames = args.frames or max(2, round(args.seconds_per_turn * args.turns * args.fps))
        poses, centre, radius, elev = orbit(c2w_all[train], frames, args.elevation, args.distance, args.turns)
        names = [f"orbit_{k:04d}" for k in range(len(poses))]
        print(f"orbit: centre {np.round(centre, 3).tolist()}, radius {radius:.3f}, elevation {elev:.1f} deg, "
              f"{len(poses)} frames = {len(poses) / args.fps:.1f} s at {args.fps} fps "
              f"({360 * args.turns * args.fps / len(poses):.0f} deg/s)")

    cams_path = os.path.join(out_dir, "cameras_orbit.json")
    with open(cams_path, "w") as f:
        json.dump(inria_json(poses, names, args.size, focal), f)

    # One SEQUENCE per frame. vkgs keeps its default camera at preset 0, so ours start at 1.
    # A sequence's parameters run before its frames are rendered, so --saveImage there captures
    # the previous sequence's camera: sequence k+1 saves frame k, and one last sequence saves the end.
    cfg_path = os.path.join(out_dir, "orbit.cfg")
    head = "--sequenceframes 4\n--sequenceaverages 1\n--sequenceresetframes 0\n"
    with open(cfg_path, "w") as f:
        f.write(f'SEQUENCE "setup"\n{head}--maxShDegree 3\n--saveImageBuffer 0\n')
        for k in range(len(names) + 1):
            f.write(f'\nSEQUENCE "{names[k] if k < len(names) else "final save"}"\n{head}')
            if k > 0:
                f.write(f"--saveImage {os.path.join(frames_dir, f'{k - 1:04d}.png')}\n")
            if k < len(names):
                f.write(f"--activateCameraPreset {k + 1}\n")
    print(f"wrote {cams_path}\n      {cfg_path}")
    if args.no_render:
        return

    vkgs = args.vkgs or os.environ.get("VKGS") or shutil.which("vkgs") or shutil.which("vk_gaussian_splatting")
    if not vkgs:
        sys.exit("vkgs not found: install it (see the top of this file), or pass --vkgs")
    cmd = [vkgs, "--headless", "0" if args.window else "1", "--benchmark", "1",
           "--size", str(args.size[0]), str(args.size[1]), "--shformat", str(args.shformat),
           "--inputFile", ply, "--loadCameraPresets", cams_path, "--sequencefile", cfg_path]
    print(" ".join(cmd))
    log_path = os.path.join(out_dir, "vkgs.log")
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=os.path.dirname(os.path.realpath(vkgs))).returncode
    pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    print(f"vkgs exit {rc}, {len(pngs)}/{len(names)} frames (log: {log_path})")
    if len(pngs) != len(names):
        sys.exit("missing frames; see the log")

    # vkgs appends the buffer name (_main) to each file; make the names sequential for ffmpeg.
    for f in pngs:
        os.rename(os.path.join(frames_dir, f), os.path.join(frames_dir, f.split("_")[0].split(".")[0] + ".png"))
    if args.training_views:
        return
    src = os.path.join(frames_dir, "%04d.png")
    mp4 = os.path.join(out_dir, f"{name}_orbit.mp4")
    gif = os.path.join(out_dir, f"{name}_orbit.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", src,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", src, "-vf",
                    f"fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", gif], check=True)
    print(f"wrote {mp4}\n      {gif}")

    # A run inside a results package (results/<name>_<run>/output/<name>/<run>): refresh the
    # shortcuts at the top of results/ so <name>_orbit.mp4/.gif appear there.
    parts = run_dir.split(os.sep)
    if len(parts) >= 5 and parts[-3] == "output" and parts[-4] == f"{parts[-2]}_{parts[-1]}":
        from link_results import link_results
        link_results(os.sep.join(parts[:-4]))


if __name__ == "__main__":
    main()
