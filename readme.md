# 暮光霭明的万能工具包

This repo is now only for **preview**, I am still working on it. I hope it will help more researchers with easy APIs one day. If you are eager to use several functions, I am happy to offer some help through [my Email](mailto:312065559@qq.com).

## Preliminary

Python >= 3.12

## Introduction

Initially, this toolkit was developed primarily for my own convenience, organizing most commonly used functions and methods. I wonder if it will be helpful to others in the future?

Currently, it includes the following modules:

- criterion: Defines some common loss functions. In most cases, existing ones from other libraries suffice, but this module addresses specific, unusual scenarios.

- dataset: Customizes datasets to support algorithms based on research needs. The goal is to organize various datasets into a consistent format, such as the OpenMIM dataset specification or other common standards.

- deploy: Contains methods used for model deployment.

- io: Defines some general read/write functions common in the medical domain.

- mm: Custom components within the OpenMIM framework.

- models: Includes some well-known neural networks.

- process: For data pre-processing and post-processing.

- utils: Other small utility tools.

## Commands

### itk_check

Check ITK image-label sample pairs whether they meet the required spacing / size.

### itk_resample

Resample ITK image-label sample pairs, according to the given spacing or size on any dimension.

### itk_orient

Orient ITK image-label sample pairs to the specified orientation, e.g., `LPI`.

### itk_patch

Extract patches from ITK image-label sample pairs. This may be helpful for training, as train-time-patching can consume a lot of CPU resources.

### itk_aug

Do augmentation on ITK image files, only supports `RandomRotate3D` now.

## OpenMIM-compatible Neural Networks

The codes are at `models`.

1. **DA_TransUnet**: DA-TransUNet: Integrating Positional and Channel Dual Attention with Transformer-Based U-Net for Enhanced Medical Image Segmentation (https://doi.org/10.3389/fbioe.2024.1398237)
2. **DconnNet**: Z. Yang and S. Farsiu, "Directional Connectivity-based Segmentation of Medical Images," in CVPR, 2023, pp. 11525-11535.
3. **LM_Net**: A Light-weight and Multi-scale Network for Medical Image Segmentation (https://doi.org/10.1016/j.compbiomed.2023.107717)
4. **MedNeXt**: Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jaeger, P.F. & Maier-Hein, K. (2023). MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2023.
5. **SegFormer**: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. NeurIPS 2021.
6. **SegFormer3D**: Perera, Shehan and Navard, Pouyan and Yilmaz, Alper. SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
7. **SwinUMamba**: Jiarun Liu, et.al., Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining.
8. **VMamba**: Liu, et.al., VMamba: Visual State Space Model.
9. **DSNet**: Guo, et.al., DSNet: A Novel Way to Use Atrous Convolutions in Semantic Segmentation, IEEE Transactions on Circuits and Systems for Video Technology.
10. **EfficientFormer**: Li, et.al., Efficientformer: Vision transformers at mobilenet speed. Advances in Neural Information Processing Systems.
11. **EfficientNet**: Mingxing Tan, Quoc V. Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019.
12. **EGE_UNet**: "EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation", which is accpeted by 26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI2023)
13. **MoCo**: Kaiming He, et.al., Momentum Contrast for Unsupervised Visual Representation Learning.
14. **SegMamba**: SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation
15. **UNet+++**: UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation
16. **UNETR**: Hatamizadeh, et.al., Unetr: Transformers for 3d medical image segmentation.

## IO toolkit

1. **SimpleITK:** `io/sitk_toolkit.py`
2. **DICOM:** `io/dcm_toolkit.py`
3. **NIfTI:** `io/nii_toolkit.py`

## Dataset

1. AdbdomenCT1K
2. BraTs2024
3. CT_ORG
4. CTSpine1K
5. FLARE_2022
6. FLARE_2023
7. ImageTBAD
8. KiTS23
9. LUNA16
10. SA_Med2D
11. TCGA
12. Totalsegmentator

## OpenMIM Plugins

1. More multi-node training method (DDP, FSDP, deepspeed).
2. 3D segmentation.
3. Visualization.
