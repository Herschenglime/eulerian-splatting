import sys
import os
import torch
import time

import torch.nn as nn
import torch.nn.functional as F
import math
from absl import logging
from functools import partial

import utils.cuda_env  # noqa: F401  (nvcc for the sampling kernel below)

# JIT-compile the CUDA sampling kernel
_cuda_sampling = None
def _load_cuda_sampling():
    global _cuda_sampling
    if _cuda_sampling is not None:
        return _cuda_sampling
    try:
        from torch.utils.cpp_extension import load
        csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc')
        _cuda_sampling = load(
            name='sample_pyramid_cuda',
            sources=[os.path.join(csrc_dir, 'sample_pyramid_cuda_kernel.cu')],
            verbose=True)
        return _cuda_sampling
    except Exception as e:
        logging.warning(f'Failed to compile CUDA sampling kernel: {e}')
        return None


class _CudaLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pxyz_indices, *pxyz_layers):
        cuda_mod = _load_cuda_sampling()
        if cuda_mod is None:
            raise RuntimeError("CUDA sampling module is unavailable")

        pxyz_layers = tuple(layer.contiguous() for layer in pxyz_layers)
        pxyz_indices = pxyz_indices.contiguous()
        ctx.save_for_backward(pxyz_indices, *pxyz_layers)
        return cuda_mod.logprob_forward_cuda(list(pxyz_layers), pxyz_indices)

    @staticmethod
    def backward(ctx, grad_log_prob):
        cuda_mod = _load_cuda_sampling()
        pxyz_indices, *pxyz_layers = ctx.saved_tensors
        grad_layers = cuda_mod.logprob_backward_cuda(
            list(pxyz_layers),
            pxyz_indices,
            grad_log_prob.contiguous())
        return (None, *grad_layers)


class _CudaLogProbFromXyz(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_samples_norm, res_list, accum_list, mapping_types, fres, *pxyz_layers):
        cuda_mod = _load_cuda_sampling()
        if cuda_mod is None:
            raise RuntimeError("CUDA sampling module is unavailable")

        pxyz_layers = tuple(layer.contiguous() for layer in pxyz_layers)
        xyz_samples_norm = xyz_samples_norm.contiguous()
        log_prob, pxyz_indices = cuda_mod.logprob_xyz_forward_cuda(
            list(pxyz_layers),
            xyz_samples_norm,
            res_list,
            accum_list,
            mapping_types,
            fres)
        ctx.save_for_backward(pxyz_indices, *pxyz_layers)
        return log_prob

    @staticmethod
    def backward(ctx, grad_log_prob):
        cuda_mod = _load_cuda_sampling()
        pxyz_indices, *pxyz_layers = ctx.saved_tensors
        grad_layers = cuda_mod.logprob_backward_cuda(
            list(pxyz_layers),
            pxyz_indices,
            grad_log_prob.contiguous())
        return (None, None, None, None, None, *grad_layers)

class HashedProbabilityPyramid(nn.Module):

    def __init__(self,
                 res_list=[40, 5, 3],
                 ncomp_list=[1, 10, 2],
                 mapping_list=[None, "tiled", "tiled"],
                 pxyz_init=0.0,
                 pxyz_activation = "softmax",
                 num_samples = 100,
                 snap_to_center=True,
                 *,
                 generator=None,
                 device="cuda"):

        super().__init__()

        assert len(res_list) == len(ncomp_list)
        assert len(mapping_list) == len(res_list)
        assert ncomp_list[0] == 1

        logging.set_verbosity(2)

        self.fres = int(torch.prod(torch.tensor(res_list)))
        self.reslist = res_list
        self.accumlist = [math.prod(self.reslist[:ii]) for ii in range(len(self.reslist))]
        self.ncomplist = ncomp_list
        self.nlayers = len(res_list)
        self.num_samples = num_samples
        self.pxyz_activation = pxyz_activation
        self.snap_to_center = snap_to_center
        self.dev = device
        self.g = generator

        # Pre-allocate reusable buffers
        self._buf_xyz = torch.zeros((num_samples, 3), device=device)
        self._buf_logp = torch.zeros((num_samples, 1), device=device)
        self._buf_accum = torch.zeros((num_samples, 3), device=device, dtype=torch.long)
        self._buf_bin = torch.zeros((num_samples, 3), device=device, dtype=torch.long)

        # Initialize learnable arrays
        self.init_pxyz(pxyz_init)

        # Define activation function
        if self.pxyz_activation.lower() == "softplus":
            self.act_pxyz = lambda pin: torch.nn.functional.softplus(pin)
        elif self.pxyz_activation.lower() == "softmax":
            self.act_pxyz = lambda pin: torch.nn.functional.softmax(pin, dim=-1)
        elif self.pxyz_activation.lower() == "sigmoid":
            self.act_pxyz = lambda pin: torch.nn.functional.sigmoid(pin)
        else:
            raise ValueError(f"Invalid pxyz_activation: {activation}")

        # Initialize mappings
        maps = {"tiled": self.get_tiled_map,
                "hashed": self.get_hashed_map}

        self.mapping = [self.get_zero_map()] + [maps[mapping_list[jj]](self.ncomplist[jj], self.accumlist[jj]) for jj in range(1, len(self.reslist))]

        # Store mapping types for CUDA kernel (0=zero, 1=tiled, 2=hashed)
        _map_codes = {None: 0, "tiled": 1, "hashed": 2}
        self.mapping_types = [_map_codes.get(m, 0) for m in mapping_list]

        # Pre-compute per-layer factors
        self._factors = [self.fres / math.prod(self.reslist[:ll+1]) for ll in range(self.nlayers)]

    def forward(self):
        return self.sample()
        
    def get_zero_map(self):
        """Mapping function takes in x, y, and z and return a component to map to."""
        tiled_fn = lambda x, y, z: torch.tensor([0], device=self.dev).expand(x.shape[0])
        return tiled_fn

    def get_tiled_map(self, mcomp, res):
        """Mapping function takes in x, y, and z and return a component to map to."""
        tiled_fn = lambda x, y, z, mcomp, res: (x + y * res + z * res**2) % mcomp

        return partial(tiled_fn, mcomp=mcomp, res=res)

    def get_hashed_map(self, mcomp, res):
        """Mapping is [N, N, N] -> M, where N*N*N is in world space (or accumulated world space I guess?)
        From: https://github.com/jonbarron/camp_zipnerf/blob/main/camp_zipnerf/internal/hash_resample.py#L85 """
        pi_2 = 19349663
        pi_3 = 83492791
        hash_fn = lambda x, y, z, mcomp: torch.bitwise_xor(x, torch.bitwise_xor(y * pi_2, z * pi_3)) % mcomp

        return partial(hash_fn, mcomp=mcomp)

    def init_pxyz(self, pxyz_init=0.0):
        # First initialize pxyz as uniform
        pxyz = [pxyz_init * torch.ones((self.ncomplist[ii],
                                        self.reslist[ii],
                                        self.reslist[ii],
                                        self.reslist[ii]),
                                       device=self.dev).requires_grad_(True).contiguous() for ii in range(len(self.reslist))]

        self._pxyz = nn.ParameterList(pxyz)

    def get_optparam_groups(self, training_args):
        grad_vars = [{"params": self._pxyz.parameters(),
                      "lr": training_args.pxyz_lr, "name": "pxyz",
                      "gridded": True}]
        return grad_vars

    def activate_pxyz(self, pin):
        p = self.act_pxyz(pin).clamp(min=1e-10) # clamp to prevent division by 0
        return p / torch.sum(p, axis=-1, keepdim=True) # ensure that it's a valid distribution

    def _activated_layer(self, ll):
        pxyz = self._pxyz[ll].contiguous()
        return self.activate_pxyz(pxyz.view(pxyz.shape[0], -1)).view(pxyz.shape)

    def sample_layer(self, pxyz, ux, uy, uz, res, curr_lvl, accum_bins,
                     bin_low, mapping):

        if res == 2:
            c_x, cy_g_x, cz_g_xy = self.get_marginal_cdfs_2(pxyz)
        else:
            c_x, cy_g_x, cz_g_xy = self.get_marginal_cdfs_n(pxyz)

        accum_bins = accum_bins * self.reslist[curr_lvl] + bin_low

        mapped_bins = mapping(accum_bins[:, 0],
                              accum_bins[:, 1],
                              accum_bins[:, 2])

        if res == 2:
            x_bin_low, x_indices = self.sample1d_2(c_x, ux, res, locs=mapped_bins)
        else:
            x_bin_low, x_indices = self.sample1d_n(c_x, ux, res, locs=mapped_bins)

        cy_g_x_flattened = cy_g_x.view(-1, res + 1)
        mapped_x_bin = mapped_bins * res + x_bin_low

        if res == 2:
            y_bin_low, y_indices = self.sample1d_2(cy_g_x_flattened, uy, res, locs=mapped_x_bin)
        else:
            y_bin_low, y_indices = self.sample1d_n(cy_g_x_flattened, uy, res, locs=mapped_x_bin)

        cz_g_xy_flattened = cz_g_xy.view(-1, res + 1)
        mapped_xy_bin = mapped_bins * res ** 2 + x_bin_low * res + y_bin_low

        if res == 2:
            z_bin_low, z_indices = self.sample1d_2(cz_g_xy_flattened, uz, res, locs=mapped_xy_bin)
        else:
            z_bin_low, z_indices = self.sample1d_n(cz_g_xy_flattened, uz, res, locs=mapped_xy_bin)

        pxyz_flat = pxyz.view(-1)

        # autodiff has an easier time if you query the log probability independently 
        xyz_loc = mapped_bins * res ** 3 + x_bin_low * res ** 2 + y_bin_low * res + z_bin_low
        pxyz_log_samples = torch.log(pxyz_flat.index_select(0, xyz_loc.view(-1,)))

        return torch.stack([x_bin_low, y_bin_low, z_bin_low], axis=-1), torch.stack([x_indices, y_indices, z_indices], axis=-1), accum_bins, pxyz_log_samples

    def sample(self, uxyz=None, act_layers=True):
        """Returns points between 0 and 1."""
        
        if uxyz is not None:
            uxyz = uxyz.clamp_(1e-8, 1 - 1e-8)
            num_samples = len(uxyz)
        else:
            num_samples = self.num_samples

        # Reuse pre-allocated buffers when possible
        if uxyz is None and num_samples == self.num_samples:
            uxyz = torch.rand((self.num_samples, 3), device=self.dev).clamp_(1e-8, 1-1e-8)
            xyz_samples = self._buf_xyz.zero_()
            log_prob_samples = torch.zeros((num_samples, 1), device=self.dev)
            accum_bins = self._buf_accum.zero_()
            bin_low = self._buf_bin.zero_()
        else:
            if uxyz is None:
                uxyz = torch.rand((num_samples, 3), device=self.dev).clamp_(1e-8, 1-1e-8)
            xyz_samples = torch.zeros((num_samples, 3), device=self.dev)
            log_prob_samples = torch.zeros((num_samples, 1), device=self.dev)
            accum_bins = torch.zeros((num_samples, 3), device=self.dev, dtype=torch.long)
            bin_low = torch.zeros((num_samples, 3), device=self.dev, dtype=torch.long)

        # get uniform RV
        ux, uy, uz = torch.chunk(uxyz, 3, axis=-1)

        if act_layers:
            self.act_layers = [self._activated_layer(ll) for ll in range(self.nlayers)]

        for ll in range(self.nlayers):               
            res = self.reslist[ll]
            pxyz_layer = self.act_layers[ll]

            bin_low, xyz, accum_bins, log_prob = self.sample_layer(
                pxyz_layer, ux, uy, uz, res=res, curr_lvl=ll,
                accum_bins=accum_bins,
                bin_low=bin_low,
                mapping=self.mapping[ll])

            xyz = xyz.detach()

            factor = self.fres / math.prod(self.reslist[:ll+1])

            if ll < self.nlayers - 1:
                xyz_samples.add_(bin_low * factor)
            else:
                xyz_samples.add_(xyz * factor)

            log_prob_samples[:, 0] += log_prob

            # calculate next u
            ux, uy, uz = torch.chunk(xyz.sub_(bin_low).clamp_(1e-8, 1-1e-8), 3, axis=-1)

        if self.snap_to_center:
            xyz_samples = torch.floor(xyz_samples) + 0.5

        bin_out = torch.floor(xyz_samples).long()

        return xyz_samples.div_(self.fres).clamp_(1e-8, 1-1e-8), log_prob_samples, bin_out[:, 0] * self.fres ** 2 + bin_out[:, 1] * self.fres + bin_out[:, 2]

    def sample_cuda(self, uxyz=None, act_layers=True):
        """CUDA-accelerated sample. Falls back to Python if CUDA kernel unavailable."""
        cuda_mod = _load_cuda_sampling()
        if cuda_mod is None:
            return self.sample(uxyz=uxyz, act_layers=act_layers)

        if uxyz is not None:
            uxyz = uxyz.clamp_(1e-8, 1 - 1e-8)
            num_samples = len(uxyz)
        else:
            num_samples = self.num_samples
            uxyz = torch.rand((self.num_samples, 3), device=self.dev).clamp_(1e-8, 1-1e-8)

        ux = uxyz[:, 0].contiguous()
        uy = uxyz[:, 1].contiguous()
        uz = uxyz[:, 2].contiguous()

        # Activate pxyz layers (maintains autograd for backward)
        if act_layers:
            self.act_layers = [self._activated_layer(ll) for ll in range(self.nlayers)]

        # Precompute CDF tables (detached — only used for index finding)
        cx_list, cy_list, cz_list = [], [], []
        for ll in range(self.nlayers):
            pxyz = self.act_layers[ll]
            res = self.reslist[ll]
            if res == 2:
                c_x, cy_g_x, cz_g_xy = self.get_marginal_cdfs_2(pxyz)
            else:
                c_x, cy_g_x, cz_g_xy = self.get_marginal_cdfs_n(pxyz)
            # Flatten CDF tables for the kernel
            cx_list.append(c_x.detach().contiguous().view(-1))
            cy_list.append(cy_g_x.detach().contiguous().view(-1))
            cz_list.append(cz_g_xy.detach().contiguous().view(-1))

        # Call fused CUDA kernel
        xyz_out, bin_1d, pxyz_indices = cuda_mod.sample_pyramid_cuda(
            ux, uy, uz,
            cx_list, cy_list, cz_list,
            self.reslist, self.ncomplist, self.accumlist,
            self.mapping_types, self._factors,
            self.fres, self.snap_to_center)

        log_prob = _CudaLogProb.apply(
            pxyz_indices,
            *(layer.contiguous().view(-1) for layer in self.act_layers))

        return xyz_out, log_prob, bin_1d


    def get_marginal_cdfs_n(self, pxyz):
        pxy = torch.sum(pxyz, axis=-1) # [C, R, R]
        px = torch.sum(pxy, axis=-1) # [C, R]

        # sample from px
        pdf_x = F.pad(px, (1, 0)) # [C, R+1]
        c_x = torch.cumsum(pdf_x, dim=-1) # [C, R]

        # calculate the cdf of y|x
        py_g_x = pxy / px.unsqueeze(-1)
        py_g_x = F.pad(py_g_x, (1, 0)) # [C, W, H + 1]
        cy_g_x = torch.cumsum(py_g_x, dim=-1) # [C, W, H]

        # calculate the cdf of z|x
        pz_g_xy = pxyz / pxy.unsqueeze(-1) # [C, R, R, R]
        pz_g_xy = F.pad(pz_g_xy, (1, 0, 0, 0)) # [C, R, R, R + 1]
        cz_g_xy = torch.cumsum(pz_g_xy, dim=-1) # [C, R, R, R]

        return c_x, cy_g_x, cz_g_xy

    def get_marginal_cdfs_2(self, pxyz):
        pxy = torch.sum(pxyz, axis=-1) # [C, R, R]
        px = torch.sum(pxy, axis=-1) # [C, R]

        c_x = torch.stack((torch.zeros_like(px.select(-1, 0)),
                                            px.select(-1, 0),
                                            px.select(-1, 0) + px.select(-1, 1)), dim=-1)

        # calculate the cdf of y|x
        py_g_x = pxy / px.unsqueeze(-1)
        cy_g_x = torch.stack((torch.zeros_like(py_g_x.select(-1, 0)),
                              py_g_x.select(-1, 0),
                              py_g_x.select(-1, 0) + py_g_x.select(-1, 1)), dim=-1)
        
        # calculate the cdf of z|x
        pz_g_xy = pxyz / pxy.unsqueeze(-1)
        cz_g_xy = torch.stack((torch.zeros_like(pz_g_xy.select(-1, 0)),
                               pz_g_xy.select(-1, 0),
                               pz_g_xy.select(-1, 0) + pz_g_xy.select(-1, 1)), dim=-1)
        
        return c_x, cy_g_x, cz_g_xy

    def sample1d_n(self, cdf, u, res, locs):
        """Assumes: cdf = [R, R], u = [R, 1], res = int"""

        c_samples = cdf.index_select(0, locs.view(-1,))
        
        bin_low = (torch.searchsorted(c_samples, u, right=True) - 1).clamp_(0, res - 1)

        clow = c_samples.gather(1, bin_low).squeeze(-1)
        chigh = c_samples.gather(1, bin_low + 1).squeeze(-1)
        
        indices = (u.squeeze(1) - clow) / (chigh - clow).clamp_(min=1e-10) + bin_low.view(-1,)

        return bin_low.view(-1,), indices

    def sample1d_2(self, cdf, u, res, locs):
        """Assumes: cdf = [R, R], u = [R, 1], res = int"""

        c_samples = cdf.index_select(0, locs.view(-1,)) # [N, 2]

        bin_low = torch.where(u > c_samples.select(-1, 1).unsqueeze(-1), 1, 0).clamp_(0, res - 1)

        clow = c_samples.gather(1, bin_low).squeeze(-1)
        chigh = c_samples.gather(1, bin_low + 1).squeeze(-1)

        indices = (u.squeeze(1) - clow) / (chigh - clow).clamp_(min=1e-10) + bin_low.view(-1,)

        return bin_low.view(-1,), indices

    def _get_log_prob_torch(self, xyz_samples_norm, act_layers=False):
        """
        Log-prob of a set of xyz_samples
        """
        fres = self.fres
        i = torch.floor(xyz_samples_norm * fres).clamp_(0, fres - 1).long()  # [N,3]
    
        N = i.shape[0]
        logp = torch.zeros(N, device=self.dev)
    
        res_accum = 1
        xyz_loc = i // (fres // self.reslist[0])  # [N,3], integer

        if act_layers:
            self.act_layers = [self._activated_layer(ll) for ll in range(self.nlayers)]

        for ll in range(self.nlayers):
            res = int(self.reslist[ll])
            pxyz_layer = self.act_layers[ll]
    
            # Component ids for each sample at this layer
            comp_ids = self.mapping[ll](xyz_loc[:, 0], xyz_loc[:, 1], xyz_loc[:, 2])  # [N]
    
            # Voxel within the component at this layer
            step = fres // (res_accum * res)           # integer scale to this layer
            xyz_vox = (i // step) % res                # [N,3] in [0,res-1]
    
            # Gather probs and accumulate log
            flat = (comp_ids * (res ** 3)
                    + xyz_vox[:, 0] * (res ** 2)
                    + xyz_vox[:, 1] * res
                    + xyz_vox[:, 2])                   # [N]
            vals = pxyz_layer.view(-1).index_select(0, flat)
            logp = logp + torch.log(vals)
    
            # Prepare xyz_loc for the *next* layer’s mapping
            xyz_loc = i // (fres // (res_accum * res))
            res_accum = res_accum * res
    
        return logp.view(-1, 1)

    def get_log_prob(self, xyz_samples_norm, act_layers=False):
        cuda_mod = _load_cuda_sampling()
        if cuda_mod is None or not xyz_samples_norm.is_cuda or xyz_samples_norm.dtype != torch.float32:
            print("Falling back to _get_log_prob_torch...")
            return self._get_log_prob_torch(xyz_samples_norm, act_layers=act_layers)

        if act_layers:
            self.act_layers = [self._activated_layer(ll) for ll in range(self.nlayers)]

        return _CudaLogProbFromXyz.apply(
            xyz_samples_norm,
            self.reslist,
            self.accumlist,
            self.mapping_types,
            self.fres,
            *(layer.contiguous().view(-1) for layer in self.act_layers))
