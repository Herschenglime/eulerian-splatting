"""GaussianModel.save_ply writes the same bytes as upstream's version, without the per-row tuples.

Upstream filled the structured array with `list(map(tuple, attributes))`, which needs ~30 GB of
host RAM for 15M Gaussians and got a 32 GB SLURM job OOM-killed at the end of a full-preset run.
"""

import sys
import types

import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Stub GPU-only modules imported by models/.
sys.modules.setdefault("tinycudann", types.ModuleType("tinycudann"))
for name in ["slang_gaussian_rasterization", "slang_gaussian_rasterization.api",
             "slang_gaussian_rasterization.api.inria_3dgs", "slang_gaussian_rasterization.internal",
             "slang_gaussian_rasterization.internal.alphablend_tiled_slang"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["slang_gaussian_rasterization.api.inria_3dgs"].common_properties_from_inria_GaussianModel = None
sys.modules["slang_gaussian_rasterization.api.inria_3dgs"].common_properties_from_inria_Camera = None
sys.modules["slang_gaussian_rasterization.internal.alphablend_tiled_slang"].render_alpha_blend_tiles_slang_raw = None

from models.gaussian_model import GaussianModel  # noqa: E402

N, SH = 777, 16


class FakeModel(GaussianModel):
    """Only what save_ply reads: get_xyz (with the 4th log-prob channel) and get_attributes."""

    def __init__(self, seed=0):  # noqa: D107 (skip GaussianModel.__init__: it builds CUDA grids)
        g = torch.Generator().manual_seed(seed)
        self._xyz4 = torch.randn(N, 4, generator=g)
        self._attrs = (torch.randn(N, SH, 3, generator=g),        # SH colour, [N, coeffs, rgb]
                       torch.rand(N, 1, generator=g),
                       torch.rand(N, 3, generator=g),
                       torch.randn(N, 4, generator=g))

    @property
    def get_xyz(self):
        return self._xyz4

    @property
    def get_attributes(self):
        return self._attrs


def upstream_save_ply(model, path):
    """The original implementation, verbatim apart from the class."""
    xyz = model.get_xyz.detach().cpu().numpy()[..., :3]
    normals = np.zeros_like(xyz)
    color, opacities, scaling, rotation = model.get_attributes
    f_dc = color[:, 0:1, :].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = color[:, 1:, :].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in model.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities.cpu().numpy(),
                                 scaling.cpu().numpy(), rotation.cpu().numpy()), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def test_save_ply_matches_upstream_bytes(tmp_path):
    model = FakeModel()
    ours, theirs = tmp_path / "ours.ply", tmp_path / "upstream.ply"
    model.save_ply(str(ours))
    upstream_save_ply(model, str(theirs))
    assert ours.read_bytes() == theirs.read_bytes()


def test_save_ply_layout(tmp_path):
    model = FakeModel(seed=1)
    path = tmp_path / "m.ply"
    model.save_ply(str(path))
    v = PlyData.read(str(path))["vertex"].data
    names = model.construct_list_of_attributes()
    assert list(v.dtype.names) == names and len(names) == 62 and len(v) == N
    color = model.get_attributes[0]
    np.testing.assert_array_equal(v["x"], model.get_xyz[:, 0].numpy())
    np.testing.assert_array_equal(v["nz"], 0)
    np.testing.assert_array_equal(v["f_dc_2"], color[:, 0, 2].numpy())
    # f_rest is channel-major: all of red's 15 coefficients, then green's, then blue's.
    np.testing.assert_array_equal(v["f_rest_15"], color[:, 1, 1].numpy())
    np.testing.assert_array_equal(v["rot_3"], model.get_attributes[3][:, 3].numpy())
