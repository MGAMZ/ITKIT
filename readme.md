# mgam-ITKIT: Feasible Medical Image Operation based on SimpleITK API

[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/) [![SimpleITK >= 2.5.0](https://img.shields.io/badge/SimpleITK-%3E%3D2.5-yellow)](https://github.com/SimpleITK/SimpleITK) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`mgam-ITKIT` is a user-friendly toolkit built on `SimpleITK` and `Python`, designed for common data preprocessing operations in data-driven CT medical image analysis. It assumes a straightforward data sample structure and offers intuitive functions for checking, resampling, pre-segmenting, aligning, and enhancing such data. Each operation is specified by a dedicated command-line entry with a clear parameter list.

Feel free to reach out to me for any questions or suggestions at [my Email](mailto:312065559@qq.com).

## Preliminary

The repo it tested on:

- Python >= 3.10
- numpy >= 2.2.6
- SimpleITK >= 2.5.0

Lower versions may work but are not guaranteed.

## Introduction

The repo includes the following modules:

- **criterion:** Defines some common loss functions. In most cases, existing ones from other libraries suffice, but this module addresses specific, unusual scenarios.

- **dataset:** Customizes datasets to support algorithms based on research needs. The goal is to organize various datasets into a consistent format, such as the OpenMIM dataset specification or other common standards.

- **deploy:** Contains methods used for model deployment.

- **io:** Defines some general read/write functions common in the medical domain.

- **mm:** Custom components within the OpenMIM framework.

- **models:** Includes some well-known neural networks.

- **process:** For data pre-processing and post-processing.

- **utils:** Other small utility tools.

Above all, the `process` module is the core of this repo, providing several command-line tools for common ITK image operations. Each operation is encapsulated in a separate script with a clear parameter list. This module should be easy to use for all medical image researchers.

## Installation

### From PyPI

Just run: `pip install itkit[medical_vision]`

### From GitHub Repo

First, clone the repository:

```bash
git clone https://gitee.com/MGAM/ITKIT.git
```

Then, install the package:

```bash
pip install ITKIT[medical_vision]
```

Optional dependencies:

- dev: for development and testing
- nvidia: onnx, tensorrt.
- medical_vision: pathology and dicom IO.
- pytorch: if you want to use the lightning extension. *Note that this dependency does not include `torch` itself, please install it separately according to your system and CUDA version.*
- mm: if you want to use with OpenMMLab.
- gui: if you want to use the PyQt6 GUI app.

---

## Dataset Structure

The following dataset structure is supported and assumed in all the `ITKIT` operations:

```plaintext
dataset_root
├── image
│   ├── a.mha
│   ├── b.mha
│   └── ...
├── label
│   ├── a.mha
│   ├── b.mha
│   └── ...
└── series_meta.json
```

## ITK Preprocessing

You may see `--help` to see more details for each command.

This repo also provides a PyQt6 GUI app for all the following operations. The graphical user interface is provided as an optional extra and depends on PyQt6. To install the package with the GUI support, run:

```bash
pip install "itkit[gui]"
```

Then, to launch the app:

```bash
itkit-app
```

If you find the GUI's DPI is not optimal, you may specify `QT_SCALE_FACTOR`:

```bash
QT_SCALE_FACTOR=2 itkit-app
```

![alt text](itkit-gui.png)

### itk_check

Check ITK image-label sample pairs whether they meet the required spacing / size.

```bash
itk_check <mode> <sample_folder> [--output OUT] [--min-size Z Y X] [--max-size Z Y X] [--min-spacing Z Y X] [--max-spacing Z Y X] [--same-spacing A B] [--same-size A B] [--mp]
```

Parameters

- **mode**: Operation mode: check | delete | copy | symlink
  - **check**: Validate image/label pairs against size/spacing rules and report nonconforming samples (no file changes).  
  - **delete**: Remove image and label files for samples that fail validation.  
  - **copy**: Copy valid image/label pairs to the specified output directory (creates image/ and label/ subfolders).  
  - **symlink**: Create symbolic links for valid image/label pairs in the specified output directory (creates image/ and label/ subfolders).
- **sample_folder**: Root folder containing image/ and label/ subfolders  
- **-o, --output** OUT: Output directory (required for copy and symlink)  
- **--min-size** Z Y X: Minimum size per Z Y X (three ints; -1 = ignore)  
- **--max-size** Z Y X: Maximum size per Z Y X (three ints; -1 = ignore)  
- **--min-spacing** Z Y X: Minimum spacing per Z Y X (three floats; -1 = ignore)  
- **--max-spacing** Z Y X: Maximum spacing per Z Y X (three floats; -1 = ignore)  
- **--same-spacing** A B: Two dims (X|Y|Z) that must have equal spacing  
- **--same-size** A B: Two dims (X|Y|Z) that must have equal size  
- **--mp**: Enable multiprocessing

**Note:** Triplet arguments use Z, Y, X order (Z→0, Y→1, X→2).

### itk_resample

Resample ITK image-label sample pairs, according to the given spacing or size on any dimension.

```bash
itk_resample <field> <source_folder> <dest_folder> [--spacing Z Y X] [--size Z Y X] [--target-folder PATH] [-r|--recursive] [--mp] [--workers N]
```

Parameters

- **field**: "image" or "label" or "dataset", will determine the output dtype and interpolation method.
- **source_folder**: Folder containing source image files (.mha/.nii/.nii.gz/.mhd).  
- **dest_folder**: Destination folder for resampled files (created if missing).  
- **--spacing** Z Y X: Target spacing per dimension (ZYX order). Use -1 to ignore a dimension.  
- **--size** Z Y X: Target size per dimension (ZYX order). Use -1 to ignore a dimension.  
- **--target-folder** PATH: Folder of reference images (must not combine with `--spacing/--size`).  
- **-r, --recursive**: Recursively process subdirectories, preserving layout.  
- **--mp**: Enable multiprocessing.  
- **--workers** N: Number of worker processes for multiprocessing.

Notes

- `--target-folder` is mutually exclusive with `--spacing/--size`.  
- Triplets use Z, Y, X order.  
- Outputs: resampled files in dest_folder, plus `resample_configs.json` and `series_meta.json`.

### itk_orient

Orient ITK image-label sample pairs to the specified orientation, e.g., `LPI`.

```bash
itk_orient <src_dir> <dst_dir> <orient> [--mp]
```

Parameters

- **src_dir**: Source directory containing .mha files (recursive scan).  
- **dst_dir**: Destination directory (preserves relative directory structure; must differ from `src_dir`).  
- **orient**: Target orientation string for SimpleITK.DICOMOrient (e.g., LPI).  
- **--mp**: Use multiprocessing to convert files in parallel.

Notes

- Skips files already present in `dst_dir`.  
- Preserves folder layout and writes converted .mha files to `dst_dir`.

### itk_patch

Extract patches from ITK image-label sample pairs. This may be helpful for training, as train-time-patching can consume a lot of CPU resources.

```bash
itk_patch <src_folder> <dst_folder> --patch-size PZ [PY PX] --patch-stride SZ [SY SX] [--minimum-foreground-ratio R] [--still-save-when-no-label] [--mp]
```

Parameters

- **src_folder**: Source root containing `image/` and `label/` subfolders (Path).  
- **dst_folder**: Destination root to save patches (Path).  
- **--patch-size**: Patch size as single int or three ints (Z Y X).  
- **--patch-stride**: Patch stride as single int or three ints (Z Y X).  
- **--minimum-foreground-ratio**: Minimum label foreground ratio to keep a patch (float, default 0.0).  
- **--keep-empty-label-prob**: Probability to keep patches that contain only background (0.0-1.0).
- **--still-save-when-no-label**: If set and label missing, save patches regardless.  
- **--mp**: Use multiprocessing to process cases in parallel.

Outputs

- Patches saved under `dst_folder/<case_name>/` with image and label patch files.  
- Per-dataset `crop_meta.json` summarizing extraction and available annotations.

Notes

- Triplets use Z, Y, X order.  
- Only processes cases with paired image and label files of the same name.

### itk_aug

Do augmentation on ITK image files, only supports `RandomRotate3D` now.

```bash
itk_aug <img_folder> <lbl_folder> [-oimg OUT_IMG] [-olbl OUT_LBL] [-n N] [--mp] [--random-rot Z Y X]
```

Parameters

- **img_folder**: Folder with source image .mha files.  
- **lbl_folder**: Folder with source label .mha files.  
- **-oimg, --out-img-folder** OUT_IMG: Optional folder to save augmented images.  
- **-olbl, --out-lbl-folder** OUT_LBL: Optional folder to save augmented labels.  
- **-n, --num** N: Number of augmented samples to generate per source sample (int).  
- **--mp**: Enable multiprocessing.  
- **--random-rot** Z Y X: Max random rotation degrees for Z Y X axes (three ints, order Z, Y, X).

Notes

- Only files present in both `img_folder` and `lbl_folder` are processed.  
- Augmented files are written only if corresponding output folders are provided.

### itk_extract

Extract specified classes from ITK semetic map, this is useful when you want to focus on a subset of organs.

```bash
itk_extract <source_folder> <dest_folder> <mappings...> [-r|--recursive] [--mp] [--workers N]
```

Parameters

- **source_folder**: Folder containing source images (.mha/.nii/.nii.gz/.mhd).  
- **dest_folder**: Destination folder to save extracted label files (created if missing).  
- **mappings**: One or more label mappings in "source:target" format (e.g., "1:0" "5:1").  
- **-r, --recursive**: Recursively process subdirectories and preserve relative paths.  
- **--mp**: Enable multiprocessing.  
- **--workers** N: Number of worker processes for multiprocessing (optional).

Outputs

- Remapped label files written to dest_folder (output extensions normalized to .mha).  
- Per-sample metadata saved to `dest_folder/extract_meta.json`.  
- Configuration saved to `dest_folder/extract_configs.json`.

Notes

- Mappings are parsed as integers; target labels must be unique.  
- If no matching files are found, the script exits with a message.  
- Safe to combine recursive and mp; progress shown via tqdm.

## OpenMMLab Extensions for Medical Vision

[OpenMMLab](https://github.com/open-mmlab) is an outstanding open‑source deep learning image analysis framework. ITKIT carries a set of OpenMMLab extension classes; based on mmengine and mmsegmentation, they define commonly used pipelines and computational modules for the medical imaging domain.

**Unfortunately**, the upstream OpenMMLab project has gradually fallen out of maintenance, and I have to consider abandoning this portion of the development work.

Please install `monai` package before you use functions in this section.

```bash
pip install --no-deps monai
```

### Experiment Runner

The repo provides an experiment runner based on `MMEngine`'s `Runner` class.
For use of our private runner class, the following gloval variables need to be set:

- `mm_workdir`: The working directory for the experiment, will be used to store logs, checkpoints, visualizations, and everything that the training procedure will produce.
- `mm_testdir`: The directory to store the test results. Used when `mmrun` command is called with `--test` flag.
- `mm_configdir`: The directory where the config file is located, we specify a structure for all experiment configs.

```plaintext
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

### Segmentation Framework

`ITKIT` also provides several remastered implementation based on `mmengine` BaseModel, its design is inspired by `mmsegmentation` but more lightweight.

See `itkit/mm/mgam_models.py` for details.

### Neural Networks

The codes are at `models`.

1. **[DA-TransUNet](https://doi.org/10.3389/fbioe.2024.1398237)**: Sun G, Pan Y, Kong W, Xu Z, Ma J, Racharak T, Nguyen L-M and Xin J (2024) DA-TransUNet: integrating spatial and channel dual attention with transformer U-net for medical image segmentation. Front. Bioeng. Biotechnol. 12:1398237. doi: 10.3389/fbioe.2024.1398237.
2. **[DconnNet](https://ieeexplore.ieee.org/document/10204304)**: Z. Yang and S. Farsiu, "Directional Connectivity-based Segmentation of Medical Images," in CVPR, 2023, pp. 11525-11535.
3. **[LM_Net](https://doi.org/10.1016/j.compbiomed.2023.107717)**: Zhenkun Lu, Chaoyin She, Wei Wang, Qinghua Huang. LM-Net: A light-weight and multi-scale network for medical image segmentation. Computers in Biology and Medicine. Volume 168, 2024. 107717, ISSN 0010-4825. https://doi.org/10.1016/j.compbiomed.2023.107717.
4. **[MedNeXt](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_39)**: Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jaeger, P.F. & Maier-Hein, K. (2023). MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2023.
5. **[SegFormer](https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)**: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. NeurIPS 2021.
6. **[SegFormer3D](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Perera_SegFormer3D_An_Efficient_Transformer_for_3D_Medical_Image_Segmentation_CVPRW_2024_paper.pdf)**: Perera, Shehan and Navard, Pouyan and Yilmaz, Alper. SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
7. **[SwinUMamba](https://link.springer.com/chapter/10.1007/978-3-031-72114-4_59)**: Liu, J. et al. (2024). Swin-UMamba: Mamba-Based UNet with ImageNet-Based Pretraining. In: Linguraru, M.G., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. MICCAI 2024. Lecture Notes in Computer Science, vol 15009. Springer, Cham. https://doi.org/10.1007/978-3-031-72114-4_59.
8. **[VMamba](https://proceedings.neurips.cc/paper_files/paper/2024/file/baa2da9ae4bfed26520bb61d259a3653-Paper-Conference.pdf)**: Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Jiao, Jianbin and Liu, Yunfan. VMamba: Visual State Space Model. Advances in Neural Information Processing Systems. 2024. pp. 103031-103063.
9. **[DSNet](https://arxiv.org/abs/2406.03702)**: Z. Guo, L. Bian, H. Wei, J. Li, H. Ni and X. Huang, "DSNet: A Novel Way to Use Atrous Convolutions in Semantic Segmentation," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 4, pp. 3679-3692, April 2025, doi: 10.1109/TCSVT.2024.3509504.
10. **[EfficientFormer](https://arxiv.org/pdf/2212.08059)**: Li, Yanyu and Yuan, Geng and Wen, Yang and Hu, Ju and Evangelidis, Georgios and Tulyakov, Sergey and Wang, Yanzhi and Ren, Jian. EfficientFormer: Vision Transformers at MobileNet Speed. Advances in Neural Information Processing Systems, 35, 2022.
11. **[EfficientNet](https://arxiv.org/abs/1905.11946)**: Mingxing Tan, Quoc V. Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019.
12. **[EGE_UNet](https://link.springer.com/content/pdf/10.1007/978-3-031-43901-8_46.pdf?pdf=inline%20link)**: "EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation", which is accpeted by 26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI2023)
13. **[MoCo](https://ieeexplore.ieee.org/document/9157636)**: K. He, H. Fan, Y. Wu, S. Xie and R. Girshick, "Momentum Contrast for Unsupervised Visual Representation Learning," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 9726-9735, doi: 10.1109/CVPR42600.2020.00975.
14. **[SegMamba](https://link.springer.com/chapter/10.1007/978-3-031-72111-3_54#citeas)**: Xing, Z., Ye, T., Yang, Y., Liu, G., Zhu, L. (2024). SegMamba: Long-Range Sequential Modeling Mamba for 3D Medical Image Segmentation. In: Linguraru, M.G., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. MICCAI 2024. Lecture Notes in Computer Science, vol 15008. Springer, Cham. https://doi.org/10.1007/978-3-031-72111-3_54.
15. **[UNet3+](https://ieeexplore.ieee.org/document/9053405)**: H. Huang et al., "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 1055-1059, doi: 10.1109/ICASSP40776.2020.9053405.
16. **[UNETR](https://ieeexplore.ieee.org/document/9706678)**: A. Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image Segmentation," 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, 2022, pp. 1748-1758, doi: 10.1109/WACV51458.2022.00181.

### Dataset

For the following datasets, we provide restructure scripts to convert them from the official released structure to a consistent structure, which can be used by our extensions with the same API. You may find the scripts like: `itkit/dataset/<dataset_name>/convert_<format>.py`.

1. **[AdbdomenCT1K](https://ieeexplore.ieee.org/document/9497733)**: J. Ma et al., "AbdomenCT-1K: Is Abdominal Organ Segmentation a Solved Problem?," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6695-6714, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3100536.
2. **[BraTs2024](https://arxiv.org/abs/2405.18368)**: Maria Correia de Verdier, et al., "The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI," arXiv preprint arXiv:2405.18368, 2024.
3. **[CT_ORG](https://www.nature.com/articles/s41597-020-00715-8)**: Rister, B., Yi, D., Shivakumar, K. et al. CT-ORG, a new dataset for multiple organ segmentation in computed tomography. Sci Data 7, 381 (2020). https://doi.org/10.1038/s41597-020-00715-8.
4. **[CTSpine1K](https://arxiv.org/pdf/2105.14711)**: Yang Deng, Ce Wang, Yuan Hui, et al. CtSpine1k: A large-scale dataset for spinal vertebrae segmentation in computed tomography. arXiv preprint arXiv:2105.14711 (2021).
5. **[FLARE 2022](https://arxiv.org/abs/2308.05862)**: Jun Ma, et al., Unleashing the Strengths of Unlabeled Data in Pan-cancer Abdominal Organ Quantification: the FLARE22 Challenge. arXiv preprint arXiv:2308.05862, 2023.
6. **[FLARE 2023](https://link.springer.com/book/10.1007/978-3-031-58776-4)**: Jun Ma, Bo Wang (Eds.). Fast, Low-resource, and Accurate Organ and Pan-cancer Segmentation in Abdomen CT: MICCAI Challenge, FLARE 2023, Held in Conjunction with MICCAI 2023, Vancouver, BC, Canada, October 8, 2023, Proceedings. Lecture Notes in Computer Science. Springer, Cham, 2024. DOI: https://doi.org/10.1007/978-3-031-58776-4. eBook ISBN: 978-3-031-58776-4; Softcover ISBN: 978-3-031-58775-7.
7. **[ImageTBAD](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.732711/full)**: Yao Z, Xie W, Zhang J, Dong Y, Qiu H, Yuan H, Jia Q, Wang T, Shi Y, Zhuang J, Que L, Xu X and Huang M (2021) ImageTBAD: A 3D Computed Tomography Angiography Image Dataset for Automatic Segmentation of Type-B Aortic Dissection. Front. Physiol. 12:732711. doi: 10.3389/fphys.2021.732711.
8. **[KiTS23](https://kits-challenge.org/kits23/#kits23-official-results)**:
   1. Nicholas Heller, Fabian Isensee, Dasha Trofimova, et al. The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT. arXiv:2307.01984 [cs.CV], 2023.
   2. Nicholas Heller, Fabian Isensee, Klaus H. Maier‑Hein, et al. The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge. Medical Image Analysis, Vol. 67, Article 101821, 2021. doi:10.1016/j.media.2020.101821.
9. **[LUNA16](https://www.sciencedirect.com/science/article/pii/S1361841517301020)**: Arnaud Arindra Adiyoso Setio, Alberto Traverso, Thomas de Bel, Moira S.N. Berens, Cas van den Bogaard, Piergiorgio Cerello, Hao Chen, Qi Dou, Maria Evelina Fantacci, Bram Geurts, Robbert van der Gugten, Pheng Ann Heng, Bart Jansen, Michael M.J. de Kaste, Valentin Kotov, Jack Yu‑Hung Lin, Jeroen T.M.C. Manders, Alexander Sóñora‑Mengana, Juan Carlos García‑Naranjo, Evgenia Papavasileiou, Mathias Prokop, Marco Saletta, Cornelia M. Schaefer‑Prokop, Ernst T. Scholten, Luuk Scholten, Miranda M. Snoeren, Ernesto Lopez Torres, Jef Vandemeulebroucke, Nicole Walasek, Guido C.A. Zuidhof, Bram van Ginneken, Colin Jacobs. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. Medical Image Analysis, Vol. 42, pp. 1–13, 2017. doi:10.1016/j.media.2017.06.015.
10. **[SA_Med2D](https://arxiv.org/abs/2308.16184)**: Junlong Cheng, et al. SAM-Med2D. arXiv, 2308.16184, 2023.
11. **[TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)**
12. **[Totalsegmentator](https://pubs.rsna.org/doi/10.1148/ryai.230024)**: Wasserthal Jakob, et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence, 5, 5, 2023.

### MMEngine Plugins

These plugins are located in `itkit/mm/mmeng_PlugIn.py`. Some if the designs act as fixes to the original implementation. Due to `MMEngine` is less active, there exists many unresolved issues.

1. A `TrainLoop` class that supports profiler: `IterBasedTrainLoop_SupportProfiler`
2. A test-time logger to record the quantified metrics: `LoggerJSON`
3. Remastered `DDP` and `FSDP` to inherit non-default `BaseModel` attributes: `RemasteredDDP`, `RemasteredFSDP`
4. A FSDP runtime strategy based on mmengine design: `RemasteredFSDP_Strategy`
5. A more stable runtime logger to prevent `lr` overflow and crashes the training due to display error: `RuntimeInfoHook`
6. A collate function acts on `DataLoader` to collect multi samples from multi workers: `multi_sample_collate`
7. A fixed OptimWarpper, which will no longer iterate parameters that do not require gradients, saving time in some specific senarios: `mgam_OptimWrapperConstructor`.

*I personally dislike `MMEngine`'s implementations here, it's too convoluted and difficult to maintain.*

## IO toolkit

1. **SimpleITK:** `io/sitk_toolkit.py`
2. **DICOM:** `io/dcm_toolkit.py`
3. **NIfTI:** `io/nii_toolkit.py`

## (Alpha) Lightning Extensions

The repo is transferring the developping framework from OpenMIM to PyTorch Lightning, dur to the former is no longer maintained this years. PyTorch Lightning may be more useable in the future when dealing with specific training techniques.

The codes are at `lightning/`.

Please install `monai` package before you use functions in this section.

```bash
pip install --no-deps monai
```

## Release Policy

**Release Branches:** Stable releases are managed on dedicated branches named v1, v2, v3, etc.

**Release Triggers:** Any Pull Request (PR) merged into a release branch (e.g., v3) automatically triggers a new release. These releases are typically minor version updates (e.g., v3.1, v3.2) and are tagged accordingly.

**Development:** All development occurs on other branches, but only merges to release branches trigger releases.

## Citation

If you find this repo helpful in your research, please consider citing:

```bibtex
@misc{mgam-ITKIT,
    author = {Yiqin Zhang},
    title = {mgam-ITKIT: Feasible Medical Image Operation based on SimpleITK API},
    year = {2025},
    url = {https://gitee.com/MGAM/ITKIT}
}
```
