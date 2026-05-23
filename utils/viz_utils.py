import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def _normalize(x: np.ndarray, *, quantiles=(0.01, 0.99), range_=(0, 255), dtype=np.uint8):
    # lo, hi = np.quantile(x, quantiles[0]), np.quantile(x, quantiles[1])
    # x = np.clip(x, lo, hi)
    # x = (x - lo) / (hi - lo + 1e-6)
    x = np.clip(x, 0, 1)
    x = (x + range_[0]) * (range_[1] - range_[0])
    return x.astype(dtype)

def make_image_grid( x: np.ndarray, normalize=True, dtype=np.uint8):
    """
    Creates a grid of images from a batch of images.

    Args:
        x (np.ndarray): Batch of images of shape `[batch_size, height, width, channel]`
        ncols (int): Number of columns in the image grid.
    """
    
    (b, h, w, c) = x.shape
    ncols = math.ceil(b**0.5)
    width = w * ncols
    height = h * math.ceil(b / ncols)

    if normalize:
        x = _normalize(x)

    grid = np.zeros((height, width, c), dtype=dtype)

    for i in range(b):
        x_offset = w * (i % ncols)
        y_offset = h * (i // ncols)
        grid[y_offset : y_offset + h, x_offset : x_offset + w] = x[i]

    return grid

def visualize_test(test_images, test_depths, gt_images, iteration, test_psnr,
                   test_lpips, test_ssim, num_gaussians, save_path, save=False, show=True, token_name="test"):

    gt_grid =  make_image_grid(np.stack(gt_images, 0), normalize=True)
    test_image_grid = make_image_grid(np.stack(test_images, 0), normalize=True)

    if len(test_depths) > 0:
        test_image_depth_grid = make_image_grid(np.stack(test_depths, 0), normalize=False, dtype=np.float32)        
        nplots = 3
        plt.figure(figsize=(35,17))
    else:
        nplots = 2
        plt.figure(figsize=(20,17))
    
    plt.suptitle("{:s}: Iteration {:d} and PSNR={:.2f}, LPIPS={:.2f}, SSIM={:.2f} for image size {:d} for {:d} gaussians".format(token_name,
                                                                                                    iteration,
                                                                                                    test_psnr,
                                                                                                    test_lpips,
                                                                                                    test_ssim,
                                                                                                    gt_images[0].shape[1],
                                                                                                    num_gaussians))
    plt.subplot(1, nplots, 1)
    plt.imshow(gt_grid)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(1, nplots, 2)
    plt.imshow(test_image_grid)
    plt.title("Rendered")
    plt.axis("off")
    if len(test_depths) > 0:
        plt.subplot(1, nplots, 3)
        plt.imshow(test_image_depth_grid)
        plt.title("Depth")
        plt.axis("off")
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(save_path,
                                 "{:s}_visualization_{:d}_for_PSNR={:04f}.png".format(token_name, iteration, test_psnr)), bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()

    return


def visualize_opacity_scale_hist(gaussians, iteration, save_path, save=True, show=True):

    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    
    fig, ax = plt.subplots(3, 1, figsize=(10,14))
    hist_vals = torch.histogram(torch.linalg.norm(scaling, axis=-1).view(-1).cpu(),
                                # bins=100)
                                bins=torch.linspace(0, 1, 100))


    ax[0].stairs(hist_vals[0].numpy(), hist_vals[1].numpy())
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    ax[0].set_ylabel("Number of gaussians")
    ax[0].set_xlabel("Scaling bins")
    # ax[0].set_ylim([0, 500])
    ax[0].set_title("Histogram of norm(scale) values")

    hist_vals = torch.histogram(opacity.view(-1).cpu(),
                                bins=torch.arange(0, 1.05, 0.05))

    ax[1].stairs(hist_vals[0].numpy(), hist_vals[1].numpy())
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    ax[1].set_ylabel("Number of gaussians")
    ax[1].set_xlabel("Opacity bins")
    ax[1].set_title("Histogram of opacity values")
    ax[1].set_xlim([-0.1, 1.1])
    ax[1].plot([0.005, 0.005], [0, hist_vals[0].max()], 'r')

    hist_vals = torch.histogram(torch.linalg.norm(gaussians.get_xyz, ord=float('inf'), dim=-1).cpu().detach(),
                                bins=torch.linspace(0, 20, 100))

    # hist_vals = torch.histogram(torch.linalg.norm(gaussians.get_xyz, dim=-1).cpu().detach(),
    #                         bins=torch.linspace(0, 5, 100))
    
    ax[2].stairs(hist_vals[0].numpy(), hist_vals[1].numpy())
    ax[2].grid(axis='y', linestyle='--', alpha=0.7)
    ax[2].set_ylabel("Number of gaussians")
    ax[2].set_xlabel("x_inf bins")
    ax[2].set_title("Histogram of x_inf")

    fig.suptitle("Iteration {:d}".format(iteration))

    fig.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path,
                                 "opacity_scaling_hist_{:05d}".format(iteration)),
                    bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return