code for FuseUNet: A Multi-Scale Feature Fusion Method for U-like Networks. 

## Abstract

Medical image segmentation is a critical task in computer vision, with UNet serving as a milestone architecture. The typical component of UNet family is the skip connection, however, their skip connections face two significant limitations: (1) they lack effective interaction between features at different scales, and (2) they rely on simple concatenation or addition operations, which constrain efficient information integration. While recent improvements to UNet have focused on enhancing encoder and decoder capabilities, these limitations remain overlooked. To overcome these challenges, we propose a novel multi-scale feature fusion method that reimagines the UNet decoding process as solving an initial value problem (IVP), treating skip connections as discrete nodes. By leveraging principles from the linear multistep method, we propose an adaptive ordinary differential equation method to enable effective multi-scale feature fusion. Our approach is independent of encoder and decoder designs, making it applicable to any U-like network. Experiments on ACDC, KiTS2023, MSD brain tumor, and ISIC2017/2018 skin lesion segmentation datasets demonstrate improved feature utilization, reduced network parameters, and maintained high performance.

## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n umamba python=3.10 -y` and `conda activate umamba `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. `cd FuseUNet/FuseUNet` and run `pip install -e .`


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

## Model Training
Download dataset and put them into the `data` folder.

[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
[KiTS](https://kits-challenge.org/kits23/)
[MSD](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train 2D models

- Train 2D `FuseUNet` model

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainer
```

### Train 3D models

- Train 3D `FuseUNet` model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainer
```

## Inference

- Predict testing cases with `FuseUNet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainer --disable_tta
```

> `CONFIGURATION` can be `2d` and `3d_fullres` for 2D and 3D models, respectively.

## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [nnU-Net](https://arxiv.org/abs/1809.10486) and [UMamba](https://arxiv.org/abs/2401.04722) for making their valuable code publicly available.

