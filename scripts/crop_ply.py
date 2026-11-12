"""Drop Gaussians farther than a radius from the scene origin.

Works on raw training PLYs and on exported (upright, standard-format) ones. See utils/ply_crop.py
for why the far tail exists and how to pick a radius.

Usage:
    python scripts/crop_ply.py in.ply out.ply --max_radius 1000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ply_crop import crop_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--max_radius", type=float, required=True,
                        help="keep Gaussians within this distance of the origin (training cameras fit in [-1,1]^3)")
    args = parser.parse_args(argv)
    n_in, n_out = crop_file(args.input, args.output, args.max_radius)
    print(f"wrote {args.output}: {n_out:,} of {n_in:,} Gaussians within radius {args.max_radius:g}")


if __name__ == "__main__":
    main()
