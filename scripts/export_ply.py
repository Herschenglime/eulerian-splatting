"""Export a trained PLY to the standard 3DGS format for viewers such as SuperSplat.

Usage:
    python scripts/export_ply.py output/<scene>/<run>/point_cloud/iteration_35000.ply out.ply \
        [--min_opacity 0.005] [--max_points 3000000] [--sh_degree 0] [--max_radius 10]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ply_export import convert_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--min_opacity", type=float, default=0.0,
                        help="drop Gaussians less opaque than this (0.005 is a good start)")
    parser.add_argument("--max_radius", type=float, default=None,
                        help="drop Gaussians farther than this from the origin (cameras fit in [-1,1]^3); "
                             "the far background shell otherwise reaches millions of units and SuperSplat "
                             "can't zoom in on the scene. 10 is a good start")
    parser.add_argument("--no_upright", action="store_true",
                        help="keep the training frame (scene up = -z) instead of rotating up to +y for viewers")
    parser.add_argument("--max_points", type=int, default=None,
                        help="keep only the N most opaque Gaussians")
    parser.add_argument("--sh_degree", type=int, default=None, choices=[0, 1, 2, 3],
                        help="drop higher SH bands; 0 shrinks the file ~3.5x (no view-dependent color)")
    parser.add_argument("--force", action="store_true",
                        help="convert even if the input looks already converted")
    args = parser.parse_args(argv)

    try:
        n_in, n_out = convert_file(args.input, args.output, min_opacity=args.min_opacity,
                                   max_points=args.max_points, sh_degree=args.sh_degree,
                                   max_radius=args.max_radius, upright=not args.no_upright, force=args.force)
    except ValueError as e:
        sys.exit(f"error: {e}")
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"wrote {args.output}: {n_out:,} of {n_in:,} Gaussians, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
