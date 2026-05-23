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
from os import makedirs
import sys
from tqdm import tqdm
import torch
import torchvision

from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, GaussianParams, get_combined_args
from scene import Scene
from gaussian_renderer import render
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, "renders", name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, "renders", name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    print("Saving renders to...{}".format(render_path))
    print("Saving GT to...{}".format(gts_path))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, extras=gaussians.render_extras)["render"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, gaussian_params: GaussianParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(gaussian_params, dataset.sh_degree, refinement=True)
        scene = Scene(dataset, gaussians, shuffle=False)

        if iteration == -1:
            iter_load = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        else:
            iter_load = iteration
        scene.load(dataset.model_path, iter_load, refinement=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", iter_load, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", iter_load, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    from run import render_model
    
    render_model(cli_args=sys.argv[1:])
    print("\nRendering complete.")