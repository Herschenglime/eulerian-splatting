"""Camera moves for rooms (inside-out captures, e.g. office), rendered with NVIDIA's vkgs viewer.
An orbit around a room looks at the back of its walls; these stay where the capture was filmed from.

    python scripts/room_views_vkgs.py <run_dir> pan  [--at -0.6] [--height 0] [--pitch -8] [--tag pan_left]
    python scripts/room_views_vkgs.py <run_dir> path [--speed 0.3] [--turn_rate 40] [--span 0 1]

    pan   stand still and turn 360 degrees in place, horizon level. --at places the camera along the
          room's long axis (the main horizontal direction of the camera positions): 0 is the middle
          of the capture, -1 and 1 its two ends. --height raises it above the cameras' median height
          (scene units; the training cameras span [-1, 1]) and --pitch tilts it (degrees, - = down).
          Raised well above the capture height, it shows how the splat breaks down off the capture.
    path  fly along the capture's own camera path in recording order, smoothed, with the horizon
          levelled and the tilt clamped. Each step takes as long as the slower of moving at --speed
          (scene units/s) and turning at --turn_rate (degrees/s), so it slows down where the capture
          turned on the spot. --span flies only part of it (fractions of the capture).

Writes <run_dir>/<tag>/ (tag defaults to the move): cameras_orbit.json, orbit.cfg, frames/,
<name>_<tag>.mp4 and <name>_<tag>.gif. Rendering, the coordinate frame and the vkgs install are
shared with scripts/orbit_vkgs.py (see its docstring).
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbit_vkgs import load_cameras, look_at, make_videos, refresh_links, render_poses  # noqa: E402


UP = np.array([0.0, -1.0, 0.0])  # exported PLYs: scene up is -y


def forward_pose(eye, forward):
    """Camera-to-world pose at eye looking along forward, horizon level."""
    m = np.eye(4)
    m[:3, :3] = look_at(eye, eye + forward, -UP)
    m[:3, 3] = eye
    return m


def room_axis(c2w_train):
    """Centre (mean camera position), long horizontal axis and half-length of the camera spread."""
    pos = c2w_train[:, :3, 3]
    centre = pos.mean(axis=0)
    flat = pos - centre
    flat -= np.outer(flat @ UP, UP)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    axis = vt[0] - (vt[0] @ UP) * UP
    axis /= np.linalg.norm(axis)
    return centre, axis, float(np.percentile(np.abs(flat @ axis), 95))


def pan(c2w_train, frames, at=0.0, height=0.0, pitch_deg=-8.0, turns=1.0):
    """Turn in place at a point on the room's long axis, starting toward the room's far end."""
    centre, axis, half = room_axis(c2w_train)
    eye = centre + at * half * axis
    eye = eye - (eye @ UP) * UP + (float(np.median(c2w_train[:, :3, 3] @ UP)) + height) * UP
    start = -axis if at > 0 else axis
    side = np.cross(UP, start)
    p = math.radians(pitch_deg)
    poses = []
    for k in range(frames):
        a = 2 * math.pi * turns * k / frames
        horiz = math.cos(a) * start + math.sin(a) * side
        poses.append(forward_pose(eye, math.cos(p) * horiz + math.sin(p) * UP))
    return np.array(poses), eye


def _smooth(x, sigma):
    """Gaussian smoothing along axis 0, edges padded with the end values."""
    if sigma <= 0:
        return x.copy()
    r = int(3 * sigma) + 1
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    padded = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(padded[:, j], w, mode="valid") for j in range(x.shape[1])], axis=1)


def path(c2w_all, fps, speed=0.3, turn_rate_deg=40.0, smooth=4.0, max_pitch_deg=20.0, span=(0.0, 1.0)):
    """Fly along the capture's camera path in recording order (camera names sort by time).
    Positions and view directions are Gaussian-smoothed over `smooth` cameras, pitch is clamped to
    max_pitch_deg and roll removed (level horizon). Pace: each step takes as long as the slower of
    moving at `speed` and turning at `turn_rate_deg`, so the camera slows down where the capture
    turned on the spot."""
    n = len(c2w_all)
    lo, hi = int(round(span[0] * (n - 1))), int(round(span[1] * (n - 1)))
    pos = _smooth(c2w_all[lo:hi + 1, :3, 3], smooth)
    fwd = _smooth(c2w_all[lo:hi + 1, :3, 2], smooth)
    horiz = fwd - np.outer(fwd @ UP, UP)
    horiz /= np.linalg.norm(horiz, axis=1, keepdims=True)
    elev = np.clip(np.arcsin(np.clip((fwd / np.linalg.norm(fwd, axis=1, keepdims=True)) @ UP, -1, 1)),
                   -math.radians(max_pitch_deg), math.radians(max_pitch_deg))
    fwd = np.cos(elev)[:, None] * horiz + np.sin(elev)[:, None] * UP

    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) / speed
    turn = np.arccos(np.clip(np.sum(fwd[1:] * fwd[:-1], axis=1), -1, 1)) / math.radians(turn_rate_deg)
    dt = _smooth(np.maximum(step, turn)[:, None], 2.0)[:, 0]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    times = np.arange(0.0, t[-1], 1.0 / fps)
    u = np.interp(times, t, np.arange(len(t)))           # fractional camera index per frame
    i0 = np.clip(np.floor(u).astype(int), 0, len(t) - 2)
    w = (u - i0)[:, None]
    # Catmull-Rom between cameras i0 and i0+1 (neighbours clamped at the ends): smooth velocity.
    def cr(a):
        p0, p1 = a[np.maximum(i0 - 1, 0)], a[i0]
        p2, p3 = a[i0 + 1], a[np.minimum(i0 + 2, len(a) - 1)]
        return 0.5 * (2 * p1 + (p2 - p0) * w + (2 * p0 - 5 * p1 + 4 * p2 - p3) * w ** 2
                      + (3 * p1 - p0 - 3 * p2 + p3) * w ** 3)
    eyes, fwds = cr(pos), cr(fwd)
    fwds /= np.linalg.norm(fwds, axis=1, keepdims=True)
    return np.array([forward_pose(e, f) for e, f in zip(eyes, fwds)]), float(t[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    sub = p.add_subparsers(dest="move", required=True)
    pp = sub.add_parser("pan", help="turn 360 degrees in place")
    pp.add_argument("--at", type=float, default=0.0, help="position on the room's long axis, -1..1 (default 0)")
    pp.add_argument("--height", type=float, default=0.0, help="raise above the cameras' median height (scene units)")
    pp.add_argument("--pitch", type=float, default=-8.0, help="tilt in degrees, negative = down (default -8)")
    pp.add_argument("--seconds_per_turn", type=float, default=20.0, help="default 20 (18 degrees/s)")
    pp.add_argument("--turns", type=float, default=1.0)
    fp = sub.add_parser("path", help="fly along the capture path")
    fp.add_argument("--speed", type=float, default=0.3, help="scene units per second (default 0.3)")
    fp.add_argument("--turn_rate", type=float, default=40.0, help="max degrees per second of turning (default 40)")
    fp.add_argument("--smooth", type=float, default=4.0, help="smoothing width, in cameras (default 4)")
    fp.add_argument("--max_pitch", type=float, default=20.0, help="clamp up/down tilt (default 20 degrees)")
    fp.add_argument("--span", type=float, nargs=2, default=[0.0, 1.0], metavar=("START", "END"),
                    help="fly only this part of the capture, as fractions (default 0 1)")
    for sp in (pp, fp):
        sp.add_argument("--tag", help="output folder and file suffix (default: pan / path)")
        sp.add_argument("--ply", help="splat to render (default: <run_dir>/<name>.ply)")
        sp.add_argument("--size", type=int, nargs=2, default=[1920, 1080], metavar=("W", "H"))
        sp.add_argument("--fps", type=int, default=30)
        sp.add_argument("--shformat", type=int, default=1, help="vkgs SH storage: 0=fp32 1=fp16 2=uint8")
        sp.add_argument("--gif_width", type=int, default=640)
        sp.add_argument("--gif_fps", type=int, default=15)
        sp.add_argument("--window", action="store_true", help="run vkgs with a visible window")
        sp.add_argument("--settle_frames", type=int, default=1, help="frames vkgs draws per camera before saving (default 1)")
        sp.add_argument("--image_format", choices=["jpg", "png"], default="jpg", help="frame format (jpg encodes much faster)")
        sp.add_argument("--no_render", action="store_true", help="only write the cameras and the sequence file")
        sp.add_argument("--vkgs", help="vk_gaussian_splatting executable (default: $VKGS or PATH)")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    name = os.path.basename(os.path.dirname(run_dir))
    ply = os.path.abspath(args.ply or os.path.join(run_dir, f"{name}.ply"))
    if not os.path.exists(ply):
        sys.exit(f"splat not found: {ply} (pass --ply)")
    c2w_all, train, _ = load_cameras(run_dir)
    tag = args.tag or args.move
    out_dir = os.path.join(run_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    if args.move == "pan":
        frames = max(2, round(args.seconds_per_turn * args.turns * args.fps))
        poses, eye = pan(c2w_all[train], frames, args.at, args.height, args.pitch, args.turns)
        print(f"pan: eye {np.round(eye, 3).tolist()}, pitch {args.pitch} deg, "
              f"{360 * args.turns / (len(poses) / args.fps):.0f} deg/s")
    else:
        poses, _ = path(c2w_all, args.fps, args.speed, args.turn_rate, args.smooth, args.max_pitch, args.span)
        print(f"path: {len(c2w_all)} cameras, span {args.span}")
    print(f"{len(poses)} frames = {len(poses) / args.fps:.1f} s at {args.fps} fps")

    names = [f"{tag}_{k:04d}" for k in range(len(poses))]
    frames_dir = render_poses(ply, poses, names, args.size, out_dir, shformat=args.shformat,
                              window=args.window, vkgs=args.vkgs, no_render=args.no_render,
                              image_format=args.image_format, settle_frames=args.settle_frames)
    if args.no_render:
        return
    make_videos(frames_dir, out_dir, f"{name}_{tag}", args.fps, args.gif_fps, args.gif_width)
    refresh_links(run_dir)


if __name__ == "__main__":
    main()
