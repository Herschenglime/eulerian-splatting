#
# Modified for Eulerian Splatting using Hashed Probability Pyramids (CVPR 2026).
# Original work Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class GaussianParams(ParamGroup): 
    def __init__(self, parser):
        self.scaling_init_val = 0.0006
        self.opacity_init_val = 0.05
        self.wrange = [-1.3, 1.3]
        self.contract_xyz = True
        self.acontract = 1.5
        self.z_threshold = 0.2
        self.attribute_params = AttributeParams(parser)
        self.sampling_params = SamplingParams(parser)
        super().__init__(parser, "Gaussian Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.sampling_params = self.sampling_params.extract(args)
        g.attribute_params = self.attribute_params.extract(args)
        return g

class AttributeParams(ParamGroup): 
    def __init__(self, parser):
        self.encoding = None
        self.opacity_network = None
        self.color_network = None
        self.scaling_rotation_network = None
        self.sh_weights_k = 0.2
        self.jit_fusion = False
        super().__init__(parser, "Attribute Grid Parameters")

class SamplingParams(ParamGroup): 
    def __init__(self, parser):
        self.res_list = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.ncomp_list = [1, 8, 64, 512, 4096, 32_768, 262_144, 2_097_152, 2_097_152, 2_097_152, 2_097_152, 2_097_152]
        self.mapping_list = ["none", "tiled", "tiled", "tiled", "tiled", "tiled", "tiled", "tiled", "hashed", "hashed", "hashed", "hashed"]
        self.pxyz_init_val = 1.0
        self.pxyz_activation = "softmax"
        self.num_samples = 15_000_000
        self.num_samples_min = -1
        self.noise_samples = True
        self.add_noise_max_steps = 20_000
        self.add_noise_factor_init = 0.002
        self.add_noise_factor_final = 0.0
        self.prop_noise = 0.20
        self.snap_to_center = True
        self.disable_cuda_compile = False
        super().__init__(parser, "Sampling Parameters")

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self.save_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.init_type = "random"
        self.load_ncameras = -1
        self.normalize_cameras = True
        self.refinement = False
        self.run_num = "000"
        self.run_id = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.refinement_iterations = 5000
        self.pxyz_lr = 0.012
        self.position_lr = 0.0
        self.feature_lr = 0.0025
        self.opacity_lr = 0.005
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.hg_feature_lr = 0.002
        self.hg_opacity_lr = 0.004
        self.hg_scaling_rotation_lr = 2e-4
        self.sh_weights_lr = 0.001
        self.lambda_dssim = 0.2
        self.random_background = True
        self.scale_reg = 0.02
        self.opacity_reg = 0.05
        self.sh_weights_reg = 0.001
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
