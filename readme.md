# mgam-ITKIT: Feasible Medical Image Operation based on SimpleITK API

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

---

## ITK Preprocessing Commands

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

## OpenMMLab Extensions

### Experiment Runner

The repo provides an experiment runner based on `MMEngine`'s `Runner` class.
For use of our private runner class, the following gloval variables need to be set:

- `mm_workdir`: The working directory for the experiment, will be used to store logs, checkpoints, visualizations, and everything that the training procedure will produce.
- `mm_testdir`: The directory to store the test results. Used when `mmrun` command is called with `--test` flag.
- `mm_configdir`: The directory where the config file is located, we specify a structure for all experiment configs.

```text
mm_configdir
├── 0.1.Config1
│   ├── mgam.py (Requires exactly this name to store non-model configs)
│   ├── <model1>.py (Model config, can be multiple)
│   ├── <model2>.py
│   └── ...
│
│   # The version prefix requires every element before the final dot to be numeric,
│   # after the final dot, the suffix should not start with a number.
├── 0.2.Config2
├── 0.3.Config3
├── 0.3.1.Config3
├── 0.4.2.3.Config3
└── ...
```

- `supported_models`(Optional): A list of string that the runner will automatically search the corresponding model config file in the `mm_configdir/<version>.<config_name>/` folder when no model name is specifed during `mmrun` command call. If not set, all `py` files except `mgam.py` will be treated as model config files.

With all the above variables set, the experiment can be run with the following command:

```bash
# Single node start
mmrun $experiment_prefix$
# Multi node start example, requires specify the mmrun script path.
export mmrun=".../itkit/itkit/mm/run.py"
torchrun --nproc_per_node 4 $mmrun $experiment_prefix$
```

Please use `mmrun --help` to see more options.

**Note**

The configuration files' format aligns with the OpenMIM specification, we recommend pure-python style config. Official docs can be found [here](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#python-beta).

### Neural Networks

The codes are at `models`.

1. **[DA-TransUNet](https://doi.org/10.3389/fbioe.2024.1398237)**: Integrating Positional and Channel Dual Attention with Transformer-Based U-Net for Enhanced Medical Image Segmentation
2. **DconnNet**: Z. Yang and S. Farsiu, "Directional Connectivity-based Segmentation of Medical Images," in CVPR, 2023, pp. 11525-11535.
3. **[LM_Net](https://doi.org/10.1016/j.compbiomed.2023.107717)**: A Light-weight and Multi-scale Network for Medical Image Segmentation
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

### Dataset

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

### Other Plugins

1. More multi-node training method (DDP, FSDP, deepspeed)
2. 3D segmentation model class
3. Segmentation visualization hooks

## IO toolkit

1. **SimpleITK:** `io/sitk_toolkit.py`
2. **DICOM:** `io/dcm_toolkit.py`
3. **NIfTI:** `io/nii_toolkit.py`

## (Alpha) Lightning Extensions

The repo is transferring the developping framework from OpenMIM to PyTorch Lightning, dur to the former is no longer maintained this years. PyTorch Lightning may be more useable in the future when dealing with specific training techniques.

The codes are at `lightning/`.

## Citation

If you find this repo helpful in your research, please consider citing:

```bibtex
@misc{mgam-ITKIT,
    author = {Yiqin Zhang},
    title = {mgam-ITKIT: Feasible Medical Image Operation based on SimpleITK API},
    year = {2025},
}
```
