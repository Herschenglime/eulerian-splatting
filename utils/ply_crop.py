"""Crop Gaussians to a sphere around the scene origin.

Training happens in a normalized frame where the training cameras fit inside [-1,1]^3, centred
on the origin. Background that the model places in the contracted outer shell un-contracts to
positions up to millions of units away (the NemeSys export spanned +-8.4M). Viewers that size
their camera to the whole bounding box (SuperSplat) then show the scene as a dot. A crop at a
generous radius (1000 kept 99.9% of NemeSys) removes only that tail; a tight one (around 10)
also removes distant background, which for a room capture may be walls.

Rotation about the origin doesn't change radii, so this works the same on raw training PLYs and
on upright exports.
"""

import numpy as np

from plyfile import PlyData, PlyElement


def within_radius(arr, max_radius):
    """Boolean mask of the Gaussians within max_radius of the origin."""
    x, y, z = (arr[c].astype(np.float64) for c in "xyz")
    return x * x + y * y + z * z <= max_radius ** 2


def crop_vertices(arr, max_radius):
    return arr[within_radius(arr, max_radius)]


def crop_file(in_path, out_path, max_radius):
    """Crop one PLY file (any Gaussian PLY with x, y, z). Returns (n_in, n_out)."""
    arr = PlyData.read(in_path)["vertex"].data
    out = crop_vertices(arr, max_radius)
    PlyData([PlyElement.describe(out, "vertex")]).write(out_path)
    return len(arr), len(out)
