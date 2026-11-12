import numpy as np


def select_subset(cam_infos, n):
    """Pick `n` cameras evenly spaced through `cam_infos` (in their given order).

    Callers pass cameras sorted by image name, so for video captures this spreads
    the kept views evenly along the camera path. Deterministic; n <= 0 or
    n >= len(cam_infos) returns the list unchanged.
    """
    if n is None or n <= 0 or n >= len(cam_infos):
        return list(cam_infos)
    idx = np.unique(np.round(np.linspace(0, len(cam_infos) - 1, n)).astype(int))
    return [cam_infos[i] for i in idx]
