"""Training-time snapshots for timelapse visuals.

At chosen iterations, renders a few fixed views twice: normally, and as
"centers" (every Gaussian shrunk to a dot at full opacity) to show where the
probability pyramid is currently placing samples. Also writes a small PLY of a
random subset of the current Gaussians in standard viewer format.
"""

import os

import torch
import torchvision

from utils.ply_export import activated_vertices_from_arrays, convert_vertices, write_vertices


def snapshot_cameras(scene, views):
    """Cameras to snapshot: indices into the test set (train set if there is no test set)."""
    cams = scene.getTestCameras() or scene.getTrainCameras()
    return [(i, cams[i]) for i in views if 0 <= i < len(cams)]


def _attributes_for(gaussians, idx):
    """Activated attributes for a subset of Gaussians, without querying all of them."""
    xyz = gaussians.get_xyz[idx].detach()
    if gaussians.refinement:
        color = torch.cat((gaussians._features_dc[idx], gaussians._features_rest[idx]), dim=1)
        opacity = gaussians.opacity_activation(gaussians._opacity[idx])
        scaling = gaussians.scaling_activation(gaussians._scaling[idx], xyz)
        rotation = gaussians.rotation_activation(gaussians._rotation[idx])
    else:
        color, opacity, scaling, rotation = gaussians.query_hashgrid_attributes(
            xyz, gaussians.xyz_norm[idx].detach())
    return [t.detach().float().cpu().numpy() for t in (xyz, color, opacity, scaling, rotation)]


@torch.no_grad()
def take_snapshot(iteration, gaussians, cameras, render_fn, pipe, out_dir,
                  max_points=500_000, center_scale=0.002, seed=0):
    """Write iter_XXXXXX_view{k}_{render,centers}.png and iter_XXXXXX_gaussians.ply into out_dir.

    cameras: list of (view_index, camera) from snapshot_cameras().
    Returns the list of written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"iter_{iteration:06d}"
    written = []

    xyz = gaussians.get_xyz
    n = xyz.shape[0]
    device = xyz.device
    background = torch.zeros(3, device=device)
    # render() indexes overrides by visible_idx, so they must cover all N Gaussians.
    dot_scale = torch.full((n, 3), center_scale, device=device)
    dot_opacity = torch.ones((n, 1), device=device)

    for k, cam in cameras:
        for kind, kwargs in (("render", {}),
                             ("centers", {"override_scaling": dot_scale, "override_opacity": dot_opacity})):
            image = render_fn(cam, gaussians, pipe, background, extras=gaussians.render_extras, **kwargs)["render"]
            path = os.path.join(out_dir, f"{tag}_view{k}_{kind}.png")
            torchvision.utils.save_image(image.clamp(0, 1), path)
            written.append(path)

    if max_points and n > 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_points].to(device)
        arr = activated_vertices_from_arrays(*_attributes_for(gaussians, idx))
        path = os.path.join(out_dir, f"{tag}_gaussians.ply")
        write_vertices(convert_vertices(arr, sh_degree=0, force=True), path)
        written.append(path)

    return written
