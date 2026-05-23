# **Eulerian Splatting using Hashed Probability Pyramids**

![model pipeline overview](assets/model_overview.png "model overview")

Official implementation for the paper **Eulerian Splatting using Hashed Probability Pyramids**. 

[![Paper](https://img.shields.io/badge/Paper-ArXiv-b31b1b)](https://arxiv.org/abs/0000.00000)
[![Project Page](https://img.shields.io/badge/Project%20Page-Eulerian%20Splatting-2ea44f)](https://euleriansplatting.github.io)

**Mia Gaia Polansky**, [**George Kopanas**](https://grgkopanas.github.io), [**Stephan Garbin**](http://stephangarbin.com), [**Todd Zickler**](https://zickler.seas.harvard.edu), [**Dor Verbin**](https://dorverbin.github.io)

> *We introduce a probabilistic splat-based radiance field framework that replaces the heuristic densification of 3D Gaussian Splatting with end-to-end gradient-based optimization of a volumetric probability density. By modeling primitive locations as samples from a memory-efficient hierarchical grid, our method allows probability mass to naturally flow to areas with high loss. Stabilized by a novel unbiased gradient estimator, this approach eliminates brittle priors to achieve state-of-the-art PSNR on mip-NeRF 360 while maintaining real-time rendering speeds.*

<hr style="height:3px; background-color:#bdbdbd; border:none;">

## BibTeX
<hr style="height: 1px; background-color:#bdbdbd; border: none; margin-top: -20px; margin-bottom: 5px;">

```bibtex
@inproceedings{polansky2026eulerian,
  title={Eulerian Splatting using Hashed Probability Pyramids},
  author={Polansky, Mia Gaia and Kopanas, George and Garbin, Stephan and Zickler, Todd and Verbin, Dor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

<hr style="height:3px; background-color:#bdbdbd; border:none;">

## Installation
<hr style="height: 1px; background-color:#bdbdbd; border: none; margin-top: -20px; margin-bottom: 5px;">

This codebase is tested with Python 3.9, PyTorch 2.4.1, and CUDA 12.1, with training on an Nvidia H200 GPU.

#### 1. Clone the repository
```bash
git clone https://github.com/euleriansplatting/eulerian-splatting.git --recursive
```
#### 2. Make a conda environment
```bash
conda create -n eulerian-splatting python=3.9 -y
conda activate eulerian-splatting
```
#### 3. Install dependencies
```bash
conda install -c nvidia/label/cuda-12.1.1 -c conda-forge cuda-toolkit cuda-nvcc cuda-runtime gxx_linux-64=11 ninja -y
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH

pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install wandb==0.17.5 plyfile torchinfo matplotlib absl-py tqdm slangtorch==1.3.4 setuptools==73.0.1
```
#### 3. Install submodules
```bash
cd eulerian-splatting

# Install slang rasterizer
pip install ./submodules/slang-gaussian-rasterization

# Install tiny-cuda-nn hash encoding grids
export TCNN_HALF_PRECISION=0
pip install ./submodules/tiny-cuda-nn/bindings/torch
```

<hr style="height:3px; background-color:#bdbdbd; border:none;">

#### Common Issues:

1. Detected cuda version mismatches version used to compile

```bash
conda install -c nvidia cuda-toolkit=12.1
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Training & Evaluation
<hr style="height: 1px; background-color:#bdbdbd; border: none; margin-top: -20px; margin-bottom: 5px;">

We provide instructions on how to train and evaluate in a jupyter notebook at [example.ipynb](example.ipynb).


To train and evaluate from the command line use:

```bash
python run.py --config="bicycle" --source_path="add path to bicycle dataset" --save_path="where to save the model's outputs (defaults to ./output)" --torch_cuda_arch_list="9.0" --dry_run # remove dry_run flag to enable WandB logging
```

To visualize model behavior more frequently, add `--test_interval`, for example `--test_interval=5000`.

Or alternatively split training and evaluation as:

```bash
python train.py --config="bicycle" --source_path="add path to bicycle dataset" --save_path="where to save the model's outputs (defaults to ./output)" --torch_cuda_arch_list="9.0" --dry_run

python render.py --config="bicycle" --source_path="add path to bicycle dataset" --model_path="add path to saved model" --eval_iteration="which saved iteration to load and render (defaults to latest)"

python metrics.py --model_paths="add path to saved model or list of paths"
```

To resume training from a checkpoint:
```bash
python train.py --config="bicycle" --source_path="add path to bicycle dataset" --dry_run --torch_cuda_arch_list="9.0" --load_checkpoint="path to saved .pth checkpoint file"
```
<hr style="height:3px; background-color:#bdbdbd; border:none;">
