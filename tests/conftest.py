import os
import random

import numpy as np
import pytest
import SimpleITK as sitk

os.environ.setdefault('MULTIPROCESSING_START_METHOD', 'spawn')


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
