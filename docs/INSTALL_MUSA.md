# Installation (MUSA Backend)

> This document describes how to install UniAD with MUSA backend (MooreThreads GPU).

## Prerequisites

- MUSA SDK 4.3.4 or later
- torch_musa 2.7.1 or later
- MTT S5000 GPU (or other MooreThreads GPU)

## Installation

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n uniad_musa python=3.9 -y
conda activate uniad_musa
```

**b. Torch: Install PyTorch with MUSA backend.**
```shell
# Install torch_musa following the official instructions from MooreThreads
pip install torch_musa
```

**c. Install mmcv-series packages (MUSA versions).**
```shell
# mmcv-full with MUDA ops (already musified)
pip install -v mmcv-full==1.6.1-musa -f https://download.openmmlab.com/mmcv/dist/musa/torch2.0/index.html

# Or install from source if pre-built wheels are not available
cd ~/repositories/mmcv  # Assuming mmcv-musa is already cloned
MUSA_HOME=/usr/local/musa FORCE_MUSA=1 pip install -e . -v

pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6
```

**d. Install UniAD.**
```shell
cd ~
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
git checkout musa  # Switch to MUSA branch
pip install -r requirements.txt
```

**e. Prepare pretrained weights.**

Pretrained weights are compatible between CUDA and MUSA backends. Download from HuggingFace:

```shell
mkdir ckpts & cd ckpts
# Download weights as described in INSTALL.md
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/r101_dcn_fcos3d_pretrain.pth
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/bevformer_r101_dcn_24ep.pth
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_track_map.pth
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_e2e.pth
```

## Changes from CUDA Version

The MUSA version replaces all `torch.cuda` calls with `torch_musa` equivalents:

| CUDA | MUSA |
|------|------|
| `torch.cuda.is_available()` | `torch_musa.is_available()` |
| `torch.cuda.current_device()` | `torch_musa.current_device()` |
| `torch.cuda.synchronize()` | `torch_musa.synchronize()` |
| `torch.cuda.amp.custom_bwd/custom_fwd` | `torch_musa.amp.custom_bwd/custom_fwd` |
| `.cuda()` | `.to('musa')` |
| `device='cuda'` | `device='musa'` |

## Verification

```shell
python -c "import torch_musa; print('MUSA available:', torch_musa.is_available())"
```

---
> Next Page: [Prepare The Dataset](./DATA_PREP.md)