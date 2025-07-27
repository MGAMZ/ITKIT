# 暮光霭明的万能工具包

This repo is now only for preview, I am still working on it. I hope it will help more researchers with easy APIs one day. If you are eager to use several functions, I am happy to offer some help through [my Email](mailto:312065559@qq.com).

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
