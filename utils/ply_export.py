"""Convert this repo's Gaussian PLYs to the standard 3DGS PLY convention.

`GaussianModel.save_ply` writes *activated* values: opacity in [0, 1] and scales
in world units. Standard 3DGS viewers (SuperSplat, SIBR, ...) expect opacity as
a logit (they apply sigmoid) and scales as logs (they apply exp). Position, SH
coefficients (f_dc_*, f_rest_* channel-major) and quaternions already match.

Only numpy and plyfile are needed, so this runs on any machine.
"""

import numpy as np
from plyfile import PlyData, PlyElement

OPACITY_EPS = 1e-6
SCALE_EPS = 1e-12


def _names(arr, prefix):
    names = [n for n in arr.dtype.names if n.startswith(prefix)]
    return sorted(names, key=lambda n: int(n.split("_")[-1]))


def check_activated(arr):
    """Raise ValueError if `arr` does not look like this repo's activated format."""
    missing = [n for n in ("x", "y", "z", "opacity") if n not in arr.dtype.names]
    if missing or not _names(arr, "scale_"):
        raise ValueError(f"not a Gaussian PLY (missing {missing or 'scale_*'})")
    opacity = arr["opacity"]
    if opacity.size and (opacity.min() < 0.0 or opacity.max() > 1.0):
        raise ValueError("opacity outside [0, 1]: file is probably already in standard (logit) format")
    for name in _names(arr, "scale_"):
        if arr[name].size and arr[name].min() <= 0.0:
            raise ValueError(f"{name} has values <= 0: file is probably already in standard (log) format")


def _truncate_sh(arr, sh_degree):
    rest = _names(arr, "f_rest_")
    if not rest:
        return [n for n in arr.dtype.names], {}
    per_channel = len(rest) // 3
    keep = (sh_degree + 1) ** 2 - 1
    if keep > per_channel:
        raise ValueError(f"sh_degree {sh_degree} needs {keep} coefficients per channel, file has {per_channel}")
    # f_rest is channel-major: index = channel * per_channel + coefficient
    rename = {}
    for c in range(3):
        for k in range(keep):
            rename[f"f_rest_{c * per_channel + k}"] = f"f_rest_{c * keep + k}"
    names = [n for n in arr.dtype.names if not n.startswith("f_rest_") or n in rename]
    return names, rename


def convert_vertices(arr, min_opacity=0.0, max_points=None, sh_degree=None, max_radius=None, force=False):
    """Return a new structured array in standard 3DGS convention.

    min_opacity: drop Gaussians whose (activated) opacity is below this.
    max_points:  keep only the N most opaque Gaussians (original order kept).
    sh_degree:   drop higher-order SH coefficients (0 keeps only f_dc).
    max_radius:  drop Gaussians farther than this from the origin. Positions are in the
                 normalized scene frame (training cameras inside [-1,1]^3), and the
                 uncontracted outer shell reaches millions of units, which makes viewers
                 that frame the whole bounding box (SuperSplat) show the scene as a dot.
    force:       skip the "already converted?" sanity check.
    """
    if not force:
        check_activated(arr)

    keep = np.ones(len(arr), dtype=bool)
    if min_opacity > 0:
        keep &= arr["opacity"] >= min_opacity
    if max_radius is not None:
        keep &= arr["x"].astype(np.float64) ** 2 + arr["y"] ** 2 + arr["z"] ** 2 <= max_radius ** 2
    idx = np.nonzero(keep)[0]
    if max_points is not None and len(idx) > max_points:
        top = np.argsort(-arr["opacity"][idx], kind="stable")[:max_points]
        idx = np.sort(idx[top])
    arr = arr[idx]

    if sh_degree is None:
        names, rename = list(arr.dtype.names), {}
    else:
        names, rename = _truncate_sh(arr, sh_degree)

    out = np.empty(len(arr), dtype=[(rename.get(n, n), "f4") for n in names])
    for n in names:
        out[rename.get(n, n)] = arr[n]

    o = np.clip(arr["opacity"].astype(np.float64), OPACITY_EPS, 1 - OPACITY_EPS)
    out["opacity"] = np.log(o / (1 - o))
    for name in _names(arr, "scale_"):
        out[name] = np.log(np.maximum(arr[name].astype(np.float64), SCALE_EPS))
    return out


def activated_vertices_from_arrays(xyz, color, opacity, scaling, rotation):
    """Build a vertex array in this repo's activated format (the layout save_ply writes).

    xyz [N,3], color [N,K,3] SH coefficients (K = (deg+1)^2), opacity [N,1] or [N],
    scaling [N,3] world units, rotation [N,4]. All numpy.
    """
    n = len(xyz)
    f_dc = color[:, 0:1, :].transpose(0, 2, 1).reshape(n, -1)
    f_rest = color[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
    cols = ([("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2])]
            + [(f"n{c}", np.zeros(n)) for c in "xyz"]
            + [(f"f_dc_{i}", f_dc[:, i]) for i in range(f_dc.shape[1])]
            + [(f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1])]
            + [("opacity", np.asarray(opacity).reshape(n))]
            + [(f"scale_{i}", scaling[:, i]) for i in range(scaling.shape[1])]
            + [(f"rot_{i}", rotation[:, i]) for i in range(rotation.shape[1])])
    arr = np.empty(n, dtype=[(name, "f4") for name, _ in cols])
    for name, values in cols:
        arr[name] = values
    return arr


def read_vertices(path):
    return PlyData.read(path)["vertex"].data


def write_vertices(arr, path):
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def convert_file(in_path, out_path, **kwargs):
    """Convert one PLY file. Returns (n_in, n_out)."""
    arr = read_vertices(in_path)
    out = convert_vertices(arr, **kwargs)
    write_vertices(out, out_path)
    return len(arr), len(out)
