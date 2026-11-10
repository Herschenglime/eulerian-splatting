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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, get_linear_lr_func, build_rotation
from torch import nn
import os
import json
from absl import logging
from functools import partial
from torchinfo import summary
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.coord_utils import contract_cube, inv_contract_cube

from models.hashgrid_modules import HashEncoding
from models.sampling_modules import HashedProbabilityPyramid

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
            
        def inv_scaling_contract(scaling, xyz):
            norm_sq = torch.sum(xyz.detach()[:, :3] ** 2, dim=-1, keepdim=True).clamp(min=1.0, max=10_000.0)
            scale_factor = 1 / ((2 * torch.sqrt(norm_sq) - 1) / norm_sq)
            return scaling / scale_factor
        
        def scaling_contract(scaling, xyz):
            norm_sq = torch.sum(xyz.detach()[:, :3] ** 2, dim=-1, keepdim=True).clamp(min=1.0, max=10_000.0)
            scale_factor = 1 / ((2 * torch.sqrt(norm_sq) - 1) / norm_sq)
            return scaling * scale_factor

        val = torch.tensor(self.scaling_init_val, device="cuda")

        self.scaling_activation = lambda s, x: scaling_contract(
            torch.nn.functional.softplus(s + self.scaling_init), x)
        self.scaling_inverse_activation = lambda s, x: (inv_scaling_contract(
            s, x) + torch.log(-torch.expm1(-inv_scaling_contract(s, x))) - self.scaling_init)
        self.opacity_activation = lambda x: torch.sigmoid(x + self.opacity_init)
        self.opacity_inverse_activation = lambda y: inverse_sigmoid(y) - self.opacity_init
        self.rotation_activation = lambda x: torch.nn.functional.normalize(x + self.rotation_init, p=2, dim=-1)
        self.rotation_inverse_activation = lambda y: y - self.rotation_init

        self.scaling_init = (val + torch.log(-torch.expm1(-val))).item()
        self.opacity_init = inverse_sigmoid(torch.tensor(self.opacity_init_val, device="cuda")).item()
        self.rotation_init = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
        self.covariance_activation = build_covariance_from_scaling_rotation

        # Use for unbounded datasets
        if self.contract_xyz:
            # contract returns values between [-2, 2], which we convert to [0, 1] 
            self.contract =  lambda x: (partial(contract_cube,
                                                a=self.acontract)(x) + 2.0) / 4.0

            # inv_contract expects values between [-2, 2]
            self.inv_contract = lambda x: partial(inv_contract_cube,
                                                      a=self.acontract)((x - .5) * 4.0)
        else: # Use for bounded (synthetic) datasets
            self.contract = lambda x: (x + (self.wrange[1] - self.wrange[0]) / 2) / (
                self.wrange[1] - self.wrange[0])
            self.inv_contract = lambda x: x * (self.wrange[1] - self.wrange[0]) - (
                self.wrange[1] - self.wrange[0]) / 2
                        
    def __init__(self, gaussian_params, sh_degree, refinement=False, optimizer_type="default", generator=None):

        self.refinement = refinement

        self.optimizer_type = optimizer_type
        self.active_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None
        self.generator = generator

        self.scaling_init_val = gaussian_params.scaling_init_val
        self.opacity_init_val = gaussian_params.opacity_init_val
        self.contract_xyz = gaussian_params.contract_xyz
        self.acontract = gaussian_params.acontract
        self.wrange = gaussian_params.wrange
        self.attrib_out_dims = [3 * (self.active_sh_degree + 1) ** 2, 1, 3, 4]

        self.setup_functions()

        if not self.refinement:
            self._attributes = self.initialize_attribute_hashgrids(gaussian_params.attribute_params)
            print(summary(self._attributes, (33, 3,)))
            
            sampling_params = gaussian_params.sampling_params
            self._density = HashedProbabilityPyramid(sampling_params.res_list,
                                                     sampling_params.ncomp_list,
                                                     sampling_params.mapping_list,
                                                     sampling_params.pxyz_init_val,
                                                     sampling_params.pxyz_activation,
                                                     sampling_params.num_samples,
                                                     sampling_params.snap_to_center,
                                                     generator=generator)
            print(summary(self._density))
        
            self.noise_samples = sampling_params.noise_samples
            self.add_noise_max_steps = sampling_params.add_noise_max_steps 
            self.num_samples_min = sampling_params.num_samples_min
            self.num_samples = sampling_params.num_samples
            self.snap_to_center = sampling_params.snap_to_center

            if self.noise_samples:
                self.get_noise = get_linear_lr_func(lr_init=sampling_params.add_noise_factor_init,
                                                    lr_final=sampling_params.add_noise_factor_final,
                                                    max_steps=sampling_params.add_noise_max_steps)
                self.num_noise = int(sampling_params.prop_noise * sampling_params.num_samples)
                self.log_prob_fn = self._density.get_log_prob
    
            if sampling_params.disable_cuda_compile:
                self.sampling_fn = torch.compile(self._density.sample)
                if self.noise_samples:
                    self.log_prob_fn = torch.compile(self._density.get_log_prob)
            else:
                self.sampling_fn = self._density.sample_cuda
                if self.noise_samples:
                    self.log_prob_fn = self._density.get_log_prob

        self.fmargin = 0.1
        self.render_extras = {'method': 'default' if self.refinement else 'reinforce'}
        self.z_threshold = gaussian_params.z_threshold
        self.max_render = 7_500_000

    def initialize_attribute_hashgrids(self, attribute_params):        
        self.hg_fres = int(attribute_params.encoding["base_resolution"]) * 2 ** (
            int(attribute_params.encoding["n_levels"]) - 1)
        return HashEncoding(attribute_params,
                            self.attrib_out_dims,
                            self.active_sh_degree).to("cuda")

    def convert2refinement(self, training_args):
        self._xyz = nn.Parameter(self.get_xyz.detach().requires_grad_(True))
        color, opacity, scaling, rotation = self.get_attributes
        
        self._features_dc = nn.Parameter(color[:, 0:1, :].detach().requires_grad_(True))
        self._features_rest = nn.Parameter(color[:, 1:, :].detach().requires_grad_(True))
        self._opacity = nn.Parameter(self.opacity_inverse_activation(opacity.detach()).requires_grad_(True))
        self._scaling = nn.Parameter(self.scaling_inverse_activation(scaling.detach(), self._xyz.detach()).requires_grad_(True))
        self._rotation = nn.Parameter(self.rotation_inverse_activation(rotation.detach()).requires_grad_(True))

        self.refinement = True
        self.render_extras = {'method': 'default'}
        self.training_setup(training_args)
        

    def capture(self):
        if self.refinement:
            return (
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._opacity,
                self._scaling,
                self._rotation,
                self.optimizer.state_dict(),
            )
        else:
            return (
                self._attributes.state_dict(),
                self._density.state_dict(),
                self.optimizer.state_dict(),
                self.xyz,
                self.xyz_norm,
            )
    
    def restore(self, model_args, training_args):
        if self.refinement:
            (self._xyz, 
             self._features_dc,
             self._features_rest,
             self._opacity,
             self._scaling,
             self._rotation,
             opt_dict) = model_args

            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
        else:
            (attributes,
             density,
             opt_dict,
             self.xyz,
             self.xyz_norm) = model_args
            
            self.training_setup(training_args)
            self._attributes.load_state_dict(attributes)
            self._density.load_state_dict(density)
            self.optimizer.load_state_dict(opt_dict)

    @torch.no_grad()
    def first_idx_and_num_dups_from_bins(self, bin_1d: torch.Tensor):
        """
        Args:
          bin_1d: [N] 1D LongTensor of packed voxel ids

        Returns:
          first_idx_sorted: [G] indices (into original rows) of first occurrences per key, sorted asc
          ndup:             number of NON-first rows (duplicates)
        """
        _, inverse, counts = torch.unique(bin_1d, return_inverse=True, return_counts=True)
        # For each unique key, find the first occurrence (smallest index)
        N = bin_1d.shape[0]
        G = counts.shape[0]
        # Scatter minimum index for each unique key
        first_idx = torch.full((G,), N, dtype=torch.long, device=bin_1d.device)
        indices = torch.arange(N, device=bin_1d.device)
        first_idx.scatter_reduce_(0, inverse, indices, reduce='amin')
        first_idx_sorted, _ = torch.sort(first_idx)
        return first_idx_sorted, N - first_idx_sorted.numel()

    def sample(self, iteration):
        xyz_norm, log_prob, bin_low = self.sampling_fn(act_layers=True)

        # Add noise to samples # todo: test this
        if self.noise_samples and (iteration < self.add_noise_max_steps):
            # create noise & add to samples
            added_noise = self.get_noise(iteration) * torch.rand((self.num_noise, 3), device="cuda")
            xyz_norm_noised = (xyz_norm[:self.num_noise] + added_noise).clamp_(1e-8, 1-1e-8)
        
            # get log prob
            log_samples_noised = self.log_prob_fn(xyz_norm_noised, act_layers=False)
            xyz_samples_noised = xyz_norm_noised * self._density.fres

            # snap to voxel center
            if self.snap_to_center:
                xyz_samples_noised = torch.floor(xyz_samples_noised) + 0.5
                xyz_norm_noised = xyz_samples_noised / self._density.fres

            with torch.no_grad():
                bin_out = torch.floor(xyz_samples_noised).long()
                bin_low_noised = bin_out[:, 0] * self._density.fres ** 2 + bin_out[:, 1] * self._density.fres + bin_out[:, 2]

            xyz_norm[:self.num_noise] = xyz_norm_noised
            bin_low[:self.num_noise] = bin_low_noised
            log_prob[:self.num_noise] = log_samples_noised

        # Next, discard all duplicates
        first_idx, ndup = self.first_idx_and_num_dups_from_bins(bin_low)
        xyz_u = xyz_norm.index_select(0, first_idx)
        bin_u = bin_low.index_select(0, first_idx)
        log_u = log_prob.index_select(0, first_idx)  # (U,1)
    
        nuniq = first_idx.numel()

        # Ealy exit condition
        if nuniq > self.num_samples_min:
            self.xyz_norm = xyz_u
            self.bin_low = bin_u
            self.log_prob = log_u  # (T,1)
            self.xyz = self.inv_contract(self.xyz_norm)
            return

        # 4. Otherwise, continue searching for more
        rounds = 0
        while rounds < 8: # hard-coded, but can increase
            rounds += 1
    
            if ndup == 0:
                break  # all unique
            if nuniq >= self.num_samples_min:
                break  # already enough uniques

            xyz_dup, log_dup, bin_dup  = self.sampling_fn(act_layers=False)
            bin_dup = bin_dup.view(-1).to(dtype=torch.long)

            # Concatenate and re-unique by 'bin_low' across the whole accumulator
            xyz_norm = torch.cat([xyz_u, xyz_dup], dim=0)
            bin_low = torch.cat([bin_u, bin_dup], dim=0)
            log_prob = torch.cat([log_u, log_dup], dim=0)  # keep (.,1)

            first_idx, ndup = self.first_idx_and_num_dups_from_bins(bin_low)

            nuniq = first_idx.numel()

            xyz_u = xyz_norm.index_select(0, first_idx)
            bin_u = bin_low.index_select(0, first_idx)
            log_u = log_prob.index_select(0, first_idx)  # (U,1)

        ntake = self.num_samples_min
        self.xyz_norm = xyz_u[:ntake]
        self.bin_low = bin_u[:ntake]
        self.log_prob = log_u[:ntake]  # (T,1)
        self.xyz = self.inv_contract(self.xyz_norm)

    def gaussians_in_frustum(self, positions, view_matrix, proj_matrix):
        """
        Clip-space frustum culling: transform to clip space with a single matmul,
        then check |x| <= w*(1+m), |y| <= w*(1+m), 0 <= z <= w  (with margin).
        """
        # Combined MVP matrix
        M = proj_matrix @ view_matrix  # [4,4]
        MT = M.T  # [4,4]

        # Transform to clip space: clip = pos @ MT[:3,:] + MT[3,:]
        # Row-vector convention: [x,y,z,1] @ M.T = pos @ MT[:3,:] + MT[3,:]
        # MT[:3,:] is [3,4], positions is [N,3], result is [N,4]
        clip = positions @ MT[:3, :] + MT[3, :]  # [N,4]

        w = clip[:, 3]  # [N]

        # Margin in clip space (approximate — scales with w like NDC margin)
        margin = self.fmargin
        w_margin = w.abs() * (1.0 + margin)

        in_frustum = (
            (clip[:, 0] >= -w_margin) &
            (clip[:, 0] <= w_margin) &
            (clip[:, 1] >= -w_margin) &
            (clip[:, 1] <= w_margin) &
            (clip[:, 2] >= -margin) &
            (clip[:, 2] <= w + margin)
        )

        return in_frustum

    def get_visible_xyz_and_attributes(self, view_matrix, proj_matrix):
        xyz = self.get_xyz

        visible_idx = torch.where(self.gaussians_in_frustum(xyz, view_matrix, proj_matrix))[0]

        if len(visible_idx) > self.max_render:
            visible_idx = visible_idx[torch.randperm(len(visible_idx), device="cuda")[:self.max_render]]

        xyz = xyz[visible_idx]        

        if self.refinement:
            color = torch.cat((self._features_dc[visible_idx],
                               self._features_rest[visible_idx]), dim=1)
            opacity = self.opacity_activation(self._opacity[visible_idx])
            scaling = self.scaling_activation(self._scaling[visible_idx], xyz)
            rotation = self.rotation_activation(self._rotation[visible_idx])

        else:
            xyz_norm = self.xyz_norm[visible_idx].detach()
            color, opacity, scaling, rotation = self.query_hashgrid_attributes(xyz, xyz_norm)
            xyz = torch.cat([xyz.detach(), self.log_prob[visible_idx]], axis=-1)

        self.scaling = scaling
        self.opacity = opacity

        return xyz, color, opacity, scaling, rotation, visible_idx

    def query_hashgrid_attributes(self, xyz, xyz_norm):
        color, opacity_raw, scaling_raw, rotation_raw = self._attributes(xyz_norm)

        # Activate the raw attributes
        opacity = self.opacity_activation(opacity_raw)
        scaling = self.scaling_activation(scaling_raw, xyz)
        rotation = self.rotation_activation(rotation_raw)

        # Store for regularization (computed with zero contraction for invariance)
        self.scaling_raw = self.scaling_activation(scaling_raw, torch.zeros_like(xyz))

        return color, opacity, scaling, rotation

    @property
    def get_attributes(self):
        xyz = self.get_xyz.detach()

        if self.refinement:
            color = torch.cat((self._features_dc, self._features_rest), dim=1)
            opacity = self.opacity_activation(self._opacity)
            scaling = self.scaling_activation(self._scaling, xyz)
            rotation = self.rotation_activation(self._rotation)
        else:
            color, opacity, scaling, rotation = self.query_hashgrid_attributes(xyz, self.xyz_norm.detach())

        self.scaling = scaling
        self.opacity = opacity

        return color, opacity, scaling, rotation

    @property
    def get_sh_weights(self):
        return self._attributes.base * (torch.exp(self._attributes._sh_weights) * self._attributes.non_dc_mask + (1 - self._attributes.non_dc_mask))

    @property
    def get_opacity(self):
        if self.refinement:
            return self.opacity_activation(self._opacity)
        else:
            return self.opacity

    @property
    def get_scaling(self):
        if self.refinement:
            return self.scaling_activation(self._scaling, self._xyz)
        else:
            return self.scaling

    @property
    def get_xyz(self):
        if self.refinement:
            return self._xyz
        else:
            return self.xyz

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        if self.refinement:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self.xyz_norm = self.contract(self._xyz)
        else:
            self.xyz = fused_point_cloud

    def training_setup(self, training_args):

        l = []

        if self.refinement:
            l = [
            {'params': [self._xyz], 'lr': training_args.position_lr, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        else:
            l.extend(self._attributes.get_optparam_groups(training_args))
            l.extend(self._density.get_optparam_groups(training_args))

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        color, opacity, scaling, rotation = self.get_attributes
        f_dc, f_rest = color[:, 0:1, :], color[:, 1:, :]
                        
        for i in range(f_dc.shape[1] * f_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1] * f_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        
        xyz = self.get_xyz.detach().cpu().numpy()[..., :3]
        normals = np.zeros_like(xyz)

        color, opacities, scaling, rotation = self.get_attributes
        f_dc = color[:, 0:1, :].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = color[:, 1:, :].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz,
                                     normals,
                                     f_dc,
                                     f_rest,
                                     opacities.cpu().numpy(),
                                     scaling.cpu().numpy(),
                                     rotation.cpu().numpy()), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, extras={}):
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.active_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.active_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self._opacity = nn.Parameter(self.opacity_inverse_activation(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)))
        scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._scaling = nn.Parameter(self.scaling_inverse_activation(scales.requires_grad_(True), self._xyz))
        self._rotation = nn.Parameter(self.rotation_inverse_activation(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)))

    def save_grids(self, path):
        if not self.refinement:
            logging.info('Saving gridded parameters to {:s}'.format(path))
            
            mkdir_p(os.path.dirname(path))
            
            grids = {}
            grids["density"] = self._density.state_dict()
            grids["attributes"] = self._attributes.state_dict()
    
            torch.save(grids, path)

    def load_grids(self, path):
        if not self.refinement:
            logging.info('Loading attribute parameters from {:s}'.format(path))

            grids = torch.load(path, weights_only=True)
    
            if "density" in grids.keys():
                state_dict_to_load = {}
                for idx, key in enumerate(grids["density"].keys()):
                    state_dict_to_load[key] = grids["density"][key]
                    logging.info("Loading {:s}".format(key))

                self._density.load_state_dict(state_dict_to_load, strict=False)
            
            if "attributes" in grids.keys():
                state_dict_to_load = {}
                for idx, key in enumerate(grids["attributes"].keys()):
                    state_dict_to_load[key] = grids["attributes"][key]
                    logging.info("Loading {:s}".format(key))

                self._attributes.load_state_dict(state_dict_to_load, strict=False)
            
            del grids