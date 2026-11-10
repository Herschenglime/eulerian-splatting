import json
import yaml
import os
import sys
from absl import logging
import torch

from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GaussianParams

from train import training
from render import render_sets
from metrics import evaluate

def load_config(path):
    with open(path, "r") as f:
        text = f.read()
    if path.lower().endswith((".yml", ".yaml")):
        if yaml is None:
            raise RuntimeError("pyyaml not installed; can't read YAML config.")
        return yaml.safe_load(text)
    return json.loads(text)

# Build the parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
gp = GaussianParams(parser)

# args for training
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--attrib_config', type=str, default=None)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action="store_true", default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000, 60_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 60_000])
parser.add_argument("--test_interval", type=int, default=-1)
parser.add_argument("--quiet", action="store_true", default=False)
parser.add_argument("--disable_visualize", action="store_true", default=False)
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--load_ply", type=str, default=None)
parser.add_argument("--dry_run", action="store_true", default=False)
parser.add_argument("--torch_cuda_arch_list", type=str, default = '8.0;8.9;10.0')

# args for timelapse snapshots (see utils/snapshot_utils.py)
parser.add_argument("--snapshot_iterations", nargs="+", type=int, default=[])
parser.add_argument("--snapshot_views", nargs="+", type=int, default=[0])
parser.add_argument("--snapshot_max_points", type=int, default=500_000)
parser.add_argument("--snapshot_center_scale", type=float, default=0.002)

# args for rendering
parser.add_argument("--eval_iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true", default=False)
parser.add_argument("--skip_test", action="store_true", default=False)

def make_args(cli_args=None, **overrides):
    args_list = cli_args if cli_args is not None else []
    
    cli_parsed = parser.parse_args(args_list)
    
    args = parser.parse_args([])

    cfg_path = overrides.get("config") or getattr(cli_parsed, "config", None)
    if cfg_path is not None:
        cfg = load_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "scenes", cfg_path + ".yaml"))
        for k, v in cfg.items():
            setattr(args, k, v)
    else:
        print("Please specify a scene for --config")
        sys.exit()

    attrib_cfg_path = overrides.get("attrib_config") or getattr(cli_parsed, "attrib_config", None)
    if attrib_cfg_path is not None:
        cfg = load_config(attrib_cfg_path)
    else:
        cfg = load_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "attribute_grids.yaml"))
        
    for k, v in cfg.items():
        setattr(args, k, v)

    explicit_cli_keys = set()
    for arg in args_list:
        if arg.startswith('-'):
            key = arg.lstrip('-').split('=')[0].replace('-', '_')
            explicit_cli_keys.add(key)

    for k, v in vars(cli_parsed).items():
        if k in explicit_cli_keys:
            setattr(args, k, v)

    for k, v in overrides.items():
        setattr(args, k, v)

    return args

def get_combined_args(parser: ArgumentParser, cli_args=None, load_checkpoint=None, **overrides):
    args_list = cli_args if cli_args is not None else []
    
    cli_parsed = parser.parse_args(args_list)

    args = parser.parse_args([])

    # First load the dataset config
    cfg_path = overrides.get("config") or getattr(cli_parsed, "config", None)
    if cfg_path is not None:
        cfg = load_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "scenes", cfg_path + ".yaml"))
        for k, v in cfg.items():
            setattr(args, k, v)
    else:
        print("Please specify a scene for --config")
        sys.exit()

    # Next load the attribute config
    attrib_cfg_path = overrides.get("attrib_config") or getattr(cli_parsed, "attrib_config", None)
    if attrib_cfg_path is not None:
        cfg = load_config(attrib_cfg_path)
    else:
        cfg = load_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", "attribute_grids.yaml"))
        
    for k, v in cfg.items():
        setattr(args, k, v)

    # Override settings with saved cfg if resuming training
    model_path = overrides.get("model_path") or getattr(cli_parsed, "model_path", None)

    load_ckpt = overrides.get("load_checkpoint") or getattr(cli_parsed, "load_checkpoint", None)
    if load_ckpt:
        model_path = os.path.dirname(os.path.dirname(load_ckpt))

    cfgfile_string = "Namespace()"
    cfgfilepath = os.path.join(model_path, "cfg_args")
    print("Looking for config file in", cfgfilepath)
    if os.path.exists(cfgfilepath):
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    else:
        print("Config file not found at", cfgfilepath)

    args_cfgfile = eval(cfgfile_string)

    for k, v in vars(args_cfgfile).items():
        setattr(args, k, v)

    explicit_cli_keys = set()
    for arg in args_list:
        if arg.startswith('-'):
            key = arg.lstrip('-').split('=')[0].replace('-', '_')
            explicit_cli_keys.add(key)

    for k, v in vars(cli_parsed).items():
        if k in explicit_cli_keys:
            setattr(args, k, v)

    for k, v in overrides.items():
        setattr(args, k, v)
    
    return args

def train_model(cli_args=None, progress_callback=None, **overrides):
    args_list = cli_args if cli_args is not None else []

    cli_parsed, _ = parser.parse_known_args(args_list)
    load_ckpt = overrides.get("load_checkpoint") or getattr(cli_parsed, "load_checkpoint", None)
    
    if load_ckpt:
        print("Loading original training configurations from cfg_args...")
        args = get_combined_args(parser, cli_args=cli_args, **overrides)
    else:
        print("Starting new training run...")
        args = make_args(cli_args=cli_args, **overrides)

    if not hasattr(args, "ip") or args.ip is None:
        args.ip = "127.0.0.1"
    if not hasattr(args, "port") or args.port is None:
        args.port = 6009
    if args.disable_visualize:
        args.visualize = False
    else:
        args.visualize = True

    torch.autograd.set_detect_anomaly(getattr(args, "detect_anomaly", False))

    os.environ['TORCH_CUDA_ARCH_LIST'] = args.torch_cuda_arch_list

    model_params = lp.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pp.extract(args)
    gaussian_params = gp.extract(args)

    return training(
        model_params,
        gaussian_params,
        optim_params,
        pipeline_params,
        args.test_iterations,
        args.test_interval,
        args.save_iterations,
        args.checkpoint_iterations,
        args.load_checkpoint,
        args.load_ply,
        args.debug_from,
        args.dry_run,
        args.visualize,
        snapshot_args={
            "iterations": args.snapshot_iterations,
            "views": args.snapshot_views,
            "max_points": args.snapshot_max_points,
            "center_scale": args.snapshot_center_scale,
        },
        progress_callback=progress_callback,
    )

def render_model(cli_args=None, **overrides):
    # Pass the cli_args down the chain
    args = get_combined_args(parser, cli_args=cli_args, **overrides)

    if not hasattr(args, "model_path") or (len(args.model_path) == 0):
        print("Need to specify a model path for rendering!")
        sys.exit()

    print("Rendering", args.model_path)
    safe_state(getattr(args, "quiet", False))

    model_params = lp.extract(args)
    pipeline_params = pp.extract(args)
    gaussian_params = gp.extract(args)

    return render_sets(
        model_params,
        gaussian_params,
        args.eval_iteration,
        pipeline_params,
        args.skip_train,
        args.skip_test,
    )

def evaluate_model(model_paths):
    return evaluate(model_paths)
    
if __name__ == "__main__":
    model_path = train_model(cli_args=sys.argv[1:])
    print("\nTraining complete.")
    
    render_model(cli_args=sys.argv[1:], model_path=model_path)
    print("\nRendering complete.")

    evaluate_model(model_paths=model_path)
    print("\nEvaluation complete.")