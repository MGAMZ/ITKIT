import os
import random
import json

import numpy as np
import pytest
import SimpleITK as sitk

os.environ.setdefault('MULTIPROCESSING_START_METHOD', 'spawn')


@pytest.fixture(scope="session")
def itkit_dummy_dataset(tmp_path_factory):
    """创建符合 ITKIT 标准结构的模拟数据集，包含 meta.json"""
    temp_dir = tmp_path_factory.mktemp("itkit_dataset")
    image_dir = os.path.join(temp_dir, "image")
    label_dir = os.path.join(temp_dir, "label")
    os.makedirs(image_dir)
    os.makedirs(label_dir)

    meta_content = {}

    for i in range(20):  # 创建 20 个样本以便测试 split [0.8, 0.05, 0.15]
        series_uid = f"SERIES_{i:03d}"
        size = (100, 128, 128)
        spacing = (1.0, 0.8, 0.8)

        # 生成图像
        img_arr = np.zeros(size, dtype=np.int16)
        img = sitk.GetImageFromArray(img_arr)
        img.SetSpacing(spacing)
        img.SetOrigin((0.0, 0.0, 0.0))

        # 生成标签
        lbl_arr = np.zeros(size, dtype=np.uint8)
        lbl = sitk.GetImageFromArray(lbl_arr)
        lbl.SetSpacing(spacing)

        # 保存
        sample_name = f"{series_uid}.mha"
        sitk.WriteImage(img, os.path.join(image_dir, sample_name), True)
        sitk.WriteImage(lbl, os.path.join(label_dir, sample_name), True)

        # 写入元数据
        meta_content[sample_name] = {
            "size": list(size),
            "spacing": list(spacing)
        }

    with open(os.path.join(temp_dir, "meta.json"), "w") as f:
        json.dump(meta_content, f)

    return temp_dir


@pytest.fixture(scope="session")
def shared_temp_data(tmp_path_factory):
    # Create a session-level temporary directory
    temp_dir = tmp_path_factory.mktemp("shared_data")

    # Create image and label subfolders
    image_dir = os.path.join(temp_dir, "image")
    label_dir = os.path.join(temp_dir, "label")
    os.makedirs(image_dir)
    os.makedirs(label_dir)

    for _ in range(5):
        size = tuple(np.random.randint(32, 129, size=3))
        spacing = tuple(np.random.uniform(0.5, 2.0, size=3))
        series_uid = f"1.2.3.{random.randint(1000, 9999)}"
        num_classes = np.random.randint(2, 6)

        # Generate image data
        img_arr = np.random.randint(-1024, 8192, size=size).astype(np.int16)
        img = sitk.GetImageFromArray(img_arr)
        img.SetSpacing(spacing)
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetMetaData("SeriesInstanceUID", series_uid)

        # Generate label data: Random uint8 array, classes from 0 to num_classes-1
        lbl_arr = np.random.randint(0, num_classes, size=size, dtype=np.uint8)
        lbl = sitk.GetImageFromArray(lbl_arr)
        lbl.SetSpacing(spacing)  # Same as image
        lbl.SetOrigin((0.0, 0.0, 0.0))
        lbl.SetMetaData("SeriesInstanceUID", series_uid)  # Same as image

        # Save files
        sample_name = f"{series_uid}.mha"
        img_path = os.path.join(image_dir, sample_name)
        lbl_path = os.path.join(label_dir, sample_name)
        sitk.WriteImage(img, img_path, True)
        sitk.WriteImage(lbl, lbl_path, True)

    # Return the data directory path
    return temp_dir
