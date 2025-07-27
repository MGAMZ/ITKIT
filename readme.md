# 暮光霭明的万能工具包

**Note:** This repo is now only for preview, I am still working on it. I hope it will help more researchers with easy APIs one day. If you are eager to use several functions, I am happy to offer some help through [my Email](mailto:312065559@qq.com).

## 简介

一开始只是为了自己开发方便，整理了大多数常用的函数方法在这个工具包中，以后不知道会不会帮到别人呢？

目前有如下几个模块：

- criterion: 定义一些常见的损失函数，其实一般情况下用别家的就可以了，这里只是应付一些特殊的情况。

- dataset: 根据研究需要，定制一些数据集支持算法。希望能够通过这个包，将各式各样的数据集组织成相同的形式。比如OpenMIM的数据集规范。也可以是一些常见的规范。

- deploy: 用于模型部署时使用的一些方法

- io: 用于定义一些通用的医学领域常见的读写函数。

- mm: OpenMIM框架下自定义组件

- models: 一些著名的神经网络

- process: 数据预处理、后处理

- utils: 其他小工具

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
