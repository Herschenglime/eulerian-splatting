import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from absl import logging

class HashEncoding(nn.Module):

    def __init__(self, hg_config, out_dims, max_sh_degree):        
        super().__init__()

        self.out_dims = out_dims
        self.nlayers = hg_config.encoding["n_levels"]
        self.nfeats = hg_config.encoding["n_features_per_level"]
        
        self.color = tcnn.NetworkWithInputEncoding(n_input_dims=3,
                                                   n_output_dims=out_dims[0],
                                                   encoding_config=hg_config.encoding,
                                                   network_config=hg_config.color_network)
        self.opacity = tcnn.NetworkWithInputEncoding(n_input_dims=3,
                                                   n_output_dims=out_dims[1],
                                                   encoding_config={**hg_config.encoding,
                                                                    "n_features_per_level": 1},
                                                   network_config=hg_config.opacity_network)
        self.scaling_rotation = tcnn.NetworkWithInputEncoding(n_input_dims=3,
                                                              n_output_dims=sum(out_dims[2:]),
                                                              encoding_config=hg_config.encoding,
                                                              network_config= hg_config.scaling_rotation_network)

        # color weights
        n = (max_sh_degree + 1)**2
        self.base = self.make_sh_weights(max_sh_degree, hg_config.sh_weights_k)
        self._sh_weights = nn.Parameter(torch.zeros(1, n, 1, device="cuda"))
        mask = torch.ones(1, n, 1, device="cuda")
        mask[..., 0, :] = 0.0
        self.non_dc_mask = mask

        if hg_config.jit_fusion:
            self.color.jit_fusion = True
            self.opacity.jit_fusion = True
            self.scaling_rotation.jit_fusion = True

    def make_sh_weights(self, max_sh_degree, sh_weights_k=0.2):
        weights = []
        for deg in range(max_sh_degree + 1):
            n_coeffs = 2 * deg + 1   # number of coeffs at this degree
            scale = sh_weights_k ** deg
            weights.extend([scale] * n_coeffs)
        return torch.tensor (weights, dtype=torch.float32, device="cuda").view(1, -1, 1)

    @property
    def get_sh_weights(self):
        return self.base * (torch.exp(self._sh_weights) * self.non_dc_mask + (1 - self.non_dc_mask))

    def forward(self, x):            
        scaling_rotation = self.scaling_rotation(x)
        scaling, rotation = torch.split(scaling_rotation, [3, 4], dim=-1)
        return self.color(x).view(x.shape[0], -1, 3) * self.get_sh_weights, self.opacity(x), scaling, rotation

    def get_optparam_groups(self, training_args):
        l = []

        l.append({"params": self.color.parameters(),
                  "lr": training_args.hg_feature_lr,
                  "name": "color_hg", "gridded": True})
        l.append({"params": self.opacity.parameters(),
                  "lr": training_args.hg_opacity_lr,
                  "name": "opacity_hg", "gridded": True})
        l.append({"params": self.scaling_rotation.parameters(),
                  "lr": training_args.hg_scaling_rotation_lr,
                  "name": "scaling_and_rotation_hg", "gridded": True})
        l.append({"params": [self._sh_weights],
                  "lr": training_args.sh_weights_lr,
                  "name": "sh_weights", "gridded": True})
            
        return l

