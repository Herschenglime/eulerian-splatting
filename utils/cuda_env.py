"""Point CUDA_HOME at the venv's CUDA toolkit when no other toolkit is configured.

The Slang rasterizer and the sampling kernel are compiled at runtime by torch's cpp_extension,
which needs nvcc. scripts/setup_cuda_home.sh installs nvcc 12.8 into .venv/cuda for machines
without a system toolkit. A CUDA_HOME that is already set (e.g. `module load cuda/12.8.1` on
HiPerGator) always wins. Imported for its side effect before anything JIT-compiles.
"""

import os
import sys


def use_venv_cuda():
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    cuda = os.path.join(sys.prefix, "cuda")
    if not os.path.isfile(os.path.join(cuda, "bin", "nvcc")):
        return
    os.environ["CUDA_HOME"] = cuda
    os.environ["PATH"] = os.path.join(cuda, "bin") + os.pathsep + os.environ.get("PATH", "")
    # cpp_extension reads CUDA_HOME once, when it is first imported.
    ext = sys.modules.get("torch.utils.cpp_extension")
    if ext is not None:
        ext.CUDA_HOME = cuda


use_venv_cuda()
