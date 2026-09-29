# Installation

Use Linux and an NVIDIA GPU for training/inference. The manuscript reports OpenPCDet v0.6, PyTorch 1.13, CUDA 11.3, and an RTX 4090D. These are the reported experimental components, not a lockfile exported from the server. This source release has not been GPU-validated in the release preparation environment.

1. Create an isolated Python environment compatible with your chosen PyTorch build (Python 3.9 is a reasonable starting point for PyTorch 1.13).
2. Install a compatible PyTorch/torchvision CUDA build and a matching spconv 2.x package. Follow the [upstream installation guidance](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md). A local CUDA compiler is needed for the custom operators; the toolkit, compiler, GPU architecture and PyTorch build must be compatible.
3. From the repository root:

```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

Install PyTorch first; setup.py imports its CUDA extension builder. With the older PyTorch stack, use a compatible NumPy 1.x version instead of allowing an unconstrained upgrade to NumPy 2. Do not reuse compiled `.so` operators from another machine.

Check the installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from pcdet.ops.iou3d_nms import iou3d_nms_utils; from pcdet.models import build_network"
python tools/train.py --help
python tools/test_speed.py --help
```

The offline split/corruption-cache preparation and sampling-policy generation commands need only Python, NumPy and PyYAML. Reference prediction, rotated-IoU fusion/ranking and model execution additionally require the compiled detection environment.

CPU checks for release helpers:

```bash
pip install pytest
python -m pytest tests -q
```

Optional visualization needs Open3D or Mayavi. External weather simulation has additional dependencies described by its upstream project. The retained `OPENPCDET_*.md` files are upstream reference documentation; prefer this project's root-relative commands.
