---
title: 'ITKIT: Feasible Common Operation based on SimpleITK API'
tags:
  - Python
  - Medical Image
  - SimpleITK
  - data processing
  - OpenMMLab
authors:
  - name: Yiqin Zhang
    orcid: 0000-0003-2099-2687
    equal-contrib: false
    affiliation: 1
  - name: Meiling Chen
    equal-contrib: false
    affiliation: 2
affiliations:
 - name: University of Shanghai for Science and Technology, Shanghai, China
   index: 1
 - name: formal-tech, Shanghai, China
   index: 2
date: 29 August 2025
bibliography: paper.bib
---

# Summary

CT images are typically stored in the `DICOM` format, which provides good standardization and reproducibility. For researchers, converting them into a more storage-friendly format is a common step in data preprocessing and medical image analysis. Currently, both industry and academia tend to use the `NIFTI` format or other formats supported by `Insight Toolkit (ITK)`, which offer good cross-platform operability. In the recently popular data-driven medical image analysis research, appropriate preprocessing of the data is a necessary step. Although the research objectives vary, a large part of these preprocessing steps are the same and can be shared and utilized among different research teams, without the need to build from scratch every time.

# Statement of Need

`mgam-ITKIT` is a user-friendly toolkit built on `SimpleITK` and `Python`, designed for common data preprocessing operations in data-driven CT medical image analysis. It assumes a straightforward data sample structure and offers intuitive functions for checking, resampling, pre-segmenting, aligning, and enhancing such data. Each operation is specified by a dedicated command-line entry with a clear parameter list.

The goal of `mgam-ITKIT` is to provide data scientists with a set of easy-to-use entry functions for almost all CT image analysis tasks. After proper configuration, users can efficiently process large-scale samples with a single command, leveraging hardware capabilities and minimizing errors that may arise from incorrect parameter settings.

# Data Processing

Since `mgam-ITKIT` primarily targets basic and universal operations, we have defined an intuitive sample storage structure, and built various data processing logics on top of this structure:

```text
root/
├── dataset1/
│   ├── image/
│   │   ├── img1.mha
│   │   ├── img2.mha
│   │   └── ...
│   │
│   ├── label/
│   │   ├── img1.mha
│   │   ├── img2.mha
│   │   └── ...
│   │
│   └── ...(metas or other folders)
│
├── dataset2/
│
└── ...(Other datasets)
```

Once the user has organized the data, all the functions will be immediately available. They will automatically analyze the file structure and proceed with storage. The common commands are listed below:

- `itk_check`: Inspect all files in the structure, generate a metadata JSON file, and perform selective deletion, copying, or soft-linking based on conditions.
- `itk_orient`: Reset the orientation of the imaging data to the user's desired definition.
- `itk_resample`: Resample the imaging data in 3D to match the user's desired voxel spacing or voxel size.
- `itk_patch`: Perform three-dimensional sliding window sampling on the imaging data and generate `ITK` files with usable metadata. This is beneficial for most deep learning frameworks as it reduces the complexity of data preprocessing during training and minimizes redundant calculations.
- `itk_aug`: Augment files that conform to the `ITK` standard, and ensure that the generated images also comply with the `ITK` standard. This is also designed to serve deep learning. Some augmentation operations can be chosen to be pre-generated before training. When deep learning practitioners find that runtime preprocessing is too complex, pre-augmenting samples is likely to be beneficial.

# Analysis Framework using OpenMMLab

After conducting data processing, researchers in data-driven methods currently tend to select a deep learning framework and build models. Most of the breakthroughs in recent years have been implemented based on the `PyTorch`[@PyTorch] framework. The `mgam-ITKIT` also provides a set of medical imaging implementation components under the `OpenMMLab`[@MMEngine] training framework based on `PyTorch`[@PyTorch], including neural network architectures, dataset definitions, and preprocessing pipeline designs. However, considering that different research teams have already deviated significantly in their choices at this stage, this part of the functionality may not provide equal value to researchers. Therefore, we have only released this part of the functionality as a secondary purpose.

Some of the functions in this section rely on MONAI[@MONAI]. The supported dataset class definitions include:

- AbdomenCT_1K[@AbdomenCT1K]
- CTSpine1K[@CTSpine1K]
- FLARE 2022[@FLARE22]
- FLARE 2023[@FLARE23]
- ImageTBAD[@ImageTBAD]
- KiTS 23[@KiTS23-1; @KiTS23-2]
- Totalsegmentator[@TSD]
- BraTs 2024[@BraTs24]
- CT ORG[@CTORG]
- LUNA16[@LUNA16]

The supported neural network architectures include:

- DA_TransUnet[@DA_Trans]
- DconnNet[@Dconn]
- DSNet[@DSNet]
- EfficientFormer[@EffiFormer]
- EfficientNet[@EffiNet]
- EGE_UNet[@EGE]
- LM_Net[@LM_Net]
- MedNeXt[@MedNeXt]
- MoCo[@MoCo] (a semi-supervised method)
- SegFormer3D[@SegFormer3D]
- SwinUMamba[@SwinUM]
- UNet3+[@UN3P]
- UNETR[@UNETR]
- VMamba[@VMamba]

# Acknowledgements

We acknowledge the open source community, which made all these efforts possible.
