#
# Modified for Eulerian Splatting using Hashed Probability Pyramids (CVPR 2026).
# Original work Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

import torch
import math
from models.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import utils.cuda_env  # noqa: F401  (nvcc for the Slang kernels, compiled on import below)
from slang_gaussian_rasterization.api.inria_3dgs import common_properties_from_inria_GaussianModel, common_properties_from_inria_Camera
from slang_gaussian_rasterization.internal.alphablend_tiled_slang import render_alpha_blend_tiles_slang_raw

def render(viewpoint_camera, pc, pipe, bg_color,
                                scaling_modifier = 1.0,
                                override_color = None, override_opacity = None,
                                override_scaling=None, extras={"method": "default",
                                                               "record_timing": False}):

    """ Implements the Interface defined in the inria code-base."""
    assert scaling_modifier == 1.0, "scaling_modifier is not supported in the slang-gaussian-rasterization."
    assert pipe.convert_SHs_python is False, "convert_SHs_python is not supported in the slang-gaussian-rasterization."
    assert pipe.compute_cov3D_python is False, "compute_cov3D_python is not supported in the slang-gaussian-rasterization."

    active_sh = pc.active_sh_degree

    world_view_transform, proj_mat, cam_pos, fovy, fovx, height, width = common_properties_from_inria_Camera(viewpoint_camera)
    
    # cull the sampled points but not the persistent points
    xyz_ws, sh_coeffs, opacity, scales, rotations, visible_idx = pc.get_visible_xyz_and_attributes(world_view_transform, proj_mat)

    if override_color is not None:
        sh_coeffs = override_color[visible_idx]

    if override_opacity is not None:
        opacity = override_opacity[visible_idx]

    if override_scaling is not None:
        scales = override_scaling[visible_idx]

    render_pkg = render_alpha_blend_tiles_slang_raw(xyz_ws, rotations, scales,
                                                    opacity, sh_coeffs, active_sh,
                                                    world_view_transform,
                                                    proj_mat, cam_pos,
                                                    fovy, fovx, height, width,
                                                    bg_color,
                                                    render_method=extras["method"])

    return render_pkg
