#
# Modified for Eulerian Splatting using Hashed Probability Pyramids (CVPR 2026).
# Original work Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

import os
import json
import yaml
import torch
import re
from random import randint
from utils.loss_utils import l1_loss, ssim, anisotropy_loss
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from models.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.system_utils import mkdir_p, searchForMaxIteration
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GaussianParams
from utils.viz_utils import visualize_test
from absl import logging

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

def training(dataset, gaussian_params, opt, pipe, test_iterations, test_interval, saving_iterations, checkpoint_iterations, checkpoint, load_ply, debug_from, dry_run=True, visualize=True, seed=42, log_level=2):
    logging.set_verbosity(log_level)

    if WANDB_FOUND:
        if not dry_run:
            wandb.login()
        
        wandb.init(
            mode="disabled" if dry_run else "online",
            project=os.path.basename(dataset.source_path) + "_" + "eulerian-splatting",
            dir=os.path.join(dataset.save_path, "output"),
            save_code="true",
            settings=wandb.Settings(code_dir=os.getcwd()),
            resume="allow",
            id=dataset.run_id if (checkpoint is not None) else None,
        )
    
    if gaussian_params.sampling_params.num_samples_min == -1:
        print("Please specify the minimum number of Gaussians using --num_samples_min.")
        sys.exit()
    first_iter = 0
    tb_writer, wandb_writer, dataset = prepare_output_and_logger(dataset, gaussian_params)

    generator = torch.Generator("cuda").manual_seed(seed)
    gaussians = GaussianModel(gaussian_params,
                              dataset.sh_degree,
                              dataset.refinement,
                              generator=generator)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        logging.info("\n[ITER {}] Loaded checkpoint at {:s}".format(first_iter, checkpoint))
    elif load_ply:
        if isinstance(load_ply, str):
            iter_load = searchForMaxIteration(os.path.join(load_ply, "point_cloud"))
        else:
            iter_load = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        scene.load(load_ply, iter_load, refinement=gaussians.refinement)
        logging.info("\n[ITER {}] Loaded gaussians from {:s}".format(iter_load, load_ply))
        gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Pre-load all training GT images to GPU to avoid CPU->GPU transfer per iteration
    for cam in scene.getTrainCameras():
        cam.original_image = cam.original_image.cuda()

    # Convert iteration lists to sets for O(1) membership lookup
    test_iterations = set(test_iterations)
    saving_iterations = set(saving_iterations)
    checkpoint_iterations = set(checkpoint_iterations)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations + opt.refinement_iterations), desc="Training progress")
    progress_bar.update(first_iter)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + opt.refinement_iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        if (iteration > opt.iterations) and (opt.refinement_iterations > 0) and (
            gaussians.refinement == False):
            gaussians.convert2refinement(opt)
            logging.info("Starting refinement...")

        if not gaussians.refinement:
            gaussians.sample(iteration)

        bg = torch.rand((3), device="cuda") / 2 if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, extras=gaussians.render_extras)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image

        if viewpoint_cam.mask is not None:
            alpha_mask = viewpoint_cam.mask.cuda()
            gt_image = gt_image + (1-alpha_mask) * bg.view(-1,1,1)
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # only apply regularization to sampling model
        if not gaussians.refinement:
            opacities = torch.abs(gaussians.get_opacity)
            opacity_reg_val = ((opacities > 0.05) * opacities).mean()
            scale_reg_val = torch.abs(gaussians.scaling_raw).mean()
            sh_weights_val = torch.abs(gaussians.get_sh_weights).sum()
            
            loss = loss + opt.opacity_reg * opacity_reg_val
            loss = loss + opt.scale_reg * scale_reg_val
            loss = loss + opt.sh_weights_reg * sh_weights_val            

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations + opt.refinement_iterations:
                progress_bar.close()

            # Log and save
            elapsed = iter_start.elapsed_time(iter_end) if iteration % 10 == 0 else None
            training_report(tb_writer, wandb_writer, iteration, Ll1, loss, l1_loss, elapsed, test_iterations, test_interval, checkpoint,
                            load_ply, first_iter, opt.iterations + opt.refinement_iterations, visualize,
                            scene, render, (pipe, background))

            if (iteration in saving_iterations):
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                save_path = os.path.join(scene.model_path, "checkpoints",
                                             "checkpoint_" + str(iteration) + ".pth")
                logging.info("\n[ITER {}] Saving Checkpoint to {:s}".format(iteration, save_path))
                mkdir_p(os.path.dirname(save_path))
                torch.save((gaussians.capture(), iteration), save_path)

            # Optimizer step
            if iteration < opt.iterations + opt.refinement_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


    if WANDB_FOUND and (not dry_run):
        wandb.finish()
    return scene.model_path

def to_namespace(obj):
    """
    Recursively converts custom objects and dictionaries into argparse.Namespace 
    objects so they can be perfectly stringified and evaluated.
    """
    if isinstance(obj, dict):
        return Namespace(**{k: to_namespace(v) for k, v in obj.items()})
    # If it's a custom object (like GroupParams) that has internal variables
    elif hasattr(obj, "__dict__") and not isinstance(obj, Namespace):
        return Namespace(**{k: to_namespace(v) for k, v in vars(obj).items()})
    elif isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    else:
        return obj
        
def prepare_output_and_logger(args, gargs):
    if WANDB_FOUND:
        if wandb.run.offline:
            wandb_writer = False
            logging.info("Wandb not available: not logging progress")
        else:
            wandb_writer = True
            logging.info("Wandb found: logging progress")
    else:
        wandb_writer = False
        logging.info("Wandb not available: not logging progress")
    
    if wandb_writer:
        unique_str = wandb.run.name        
        if not wandb.run.offline:
            args.run_num = "{:03d}".format(int(unique_str.split('-')[-1]))
    elif os.getenv('OAR_JOB_ID'):
        unique_str=os.getenv('OAR_JOB_ID')[0:10]
    else:
        unique_str = str(uuid.uuid4())[0:10]

    args.run_id = unique_str
    run_name = str(args.run_num) + "_" + unique_str

    if not args.save_path:
        args.save_path = os.path.dirname(os.path.realpath(__file__))
        
    if not args.model_path:
        args.model_path = os.path.join(args.save_path, "output", os.path.basename(args.source_path), run_name)
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    combined_dict = {**vars(args), **vars(gargs)}
    combined_namespace = to_namespace(combined_dict)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(combined_namespace))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, wandb_writer, args

def training_report(tb_writer, wandb_writer, iteration, Ll1, loss, l1_loss, elapsed, test_iterations, test_interval, checkpoint, load_ply, first_iter, last_iter, visualize, scene : Scene, renderFunc, renderArgs):

    if iteration % 10 == 0:
        if tb_writer:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
            tb_writer.add_scalar('iter_time', elapsed, iteration)
    
        if wandb_writer:
            if (iteration % 10 == 0):
                wandb.log({"train_loss_patches/l1_loss": Ll1.item(),
                          "train_loss_patches/total_loss": loss.item(),
                           "iter_time": elapsed,
                           "num_gaussians": scene.gaussians.get_xyz.shape[0]}, step=iteration)

    # Report test and samples of training set
    if (iteration in test_iterations) or ((iteration % test_interval == 0) and (test_interval != -1)) or ((checkpoint is not None) and (iteration == first_iter)) or ((load_ply is not None) and (iteration == first_iter)) or (iteration == last_iter):
        
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        g = torch.Generator("cpu").manual_seed(42)
        for config in validation_configs:
            save_idx = torch.randperm(len(config["cameras"]), generator=g)[:9]
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0

                gt_images = []
                test_images = []
                test_images_depth = []
                
                for idx, viewpoint in enumerate(config['cameras']):
                    rendered = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    
                    image = torch.clamp(rendered["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if (idx in save_idx):
                        test_images.append(image.permute(1,2,0).cpu().numpy())
                        gt_images.append(gt_image.permute(1,2,0).cpu().numpy())

                        if rendered["extra"] is not None:
                            test_images_depth.append(rendered["extra"][1,:,:].unsqueeze(-1).cpu().numpy())
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    if wandb_writer and (idx < 5):
                        write_images = {}
                        write_images[config['name'] + "_view_{}/render".format(viewpoint.image_name)] = wandb.Image(image[None])
                        write_images[config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name)] = wandb.Image(gt_image[None])
                        wandb.log(write_images, step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image * 2 - 1, gt_image * 2 - 1, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])     
                lpips_test /= len(config['cameras'])  
                ssim_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if wandb_writer:
                     wandb.log({config['name'] + '/l1_loss': l1_test,
                                config['name'] + '/ssim_loss': ssim_test,
                                config['name'] + '/psnr': psnr_test,
                                config['name'] + '/lpips': lpips_test}, step=iteration) 

                if config['name'] == 'test':
                    num_gaussians = len(scene.gaussians.get_xyz)
                    visualize_test(test_images, test_images_depth, gt_images, iteration,
                                   psnr_test, lpips_test, ssim_test, num_gaussians,
                                   scene.model_path, save=True, show=visualize)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    from run import train_model
    
    model_path = train_model(cli_args=sys.argv[1:])
    logging.info("Model saved to model_path={:s}".format(model_path))
    print("\nTraining complete.")
