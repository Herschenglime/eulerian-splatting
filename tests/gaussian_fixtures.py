"""Build PLY vertex arrays laid out exactly like GaussianModel.save_ply writes them."""

import numpy as np

SH_C0 = 0.28209479177387814

# Property order of the reference 3DGS format (and of GaussianModel.construct_list_of_attributes)
STANDARD_NAMES = (["x", "y", "z", "nx", "ny", "nz"]
                  + [f"f_dc_{i}" for i in range(3)]
                  + [f"f_rest_{i}" for i in range(45)]
                  + ["opacity"]
                  + [f"scale_{i}" for i in range(3)]
                  + [f"rot_{i}" for i in range(4)])


def rgb_to_dc(rgb):
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


def activated_vertices(xyz, rgb, opacity, scale, rot=None, f_rest=None):
    """Structured array in this repo's activated format.

    xyz [N,3], rgb [N,3] in 0..1, opacity [N] in 0..1, scale [N,3] world units,
    rot [N,4] (w,x,y,z), f_rest [N,3,15] channel-major SH bands 1..3.
    """
    n = len(xyz)
    rot = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)) if rot is None else rot
    f_rest = np.zeros((n, 3, 15)) if f_rest is None else f_rest
    arr = np.zeros(n, dtype=[(name, "f4") for name in STANDARD_NAMES])
    for i, c in enumerate("xyz"):
        arr[c] = xyz[:, i]
    dc = rgb_to_dc(rgb)
    for i in range(3):
        arr[f"f_dc_{i}"] = dc[:, i]
    flat = f_rest.reshape(n, 45)
    for i in range(45):
        arr[f"f_rest_{i}"] = flat[:, i]
    arr["opacity"] = opacity
    for i in range(3):
        arr[f"scale_{i}"] = scale[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = rot[:, i]
    return arr


def make_test_shapes():
    """RGB axes (X red, Y green, Z blue) plus a yellow wireframe cube, as small opaque dots."""
    pts, cols = [], []
    t = np.linspace(0, 1, 60)
    for axis, color in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        p = np.zeros((len(t), 3))
        p[:, axis] = t
        pts.append(p)
        cols.append(np.tile(color, (len(t), 1)))
    corners = np.array([[x, y, z] for x in (0.3, 0.8) for y in (0.3, 0.8) for z in (0.3, 0.8)])
    s = np.linspace(0, 1, 25)[:, None]
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:  # cube edge
                pts.append(corners[i] + s * (corners[j] - corners[i]))
                cols.append(np.tile((1, 0.85, 0), (len(s), 1)))
    xyz = np.concatenate(pts)
    rgb = np.concatenate(cols)
    n = len(xyz)
    return activated_vertices(xyz, rgb, opacity=np.full(n, 0.95), scale=np.full((n, 3), 0.006))
