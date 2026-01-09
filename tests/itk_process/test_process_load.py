import os
import numpy as np
import pytest
import cv2
from itkit.process.LoadBiomedicalData import (
    LoadImgFromOpenCV,
    LoadAnnoFromOpenCV,
    LoadImageFromMHA,
    LoadMaskFromMHA
)

@pytest.fixture
def temp_opencv_data(tmp_path):
    img_path = str(tmp_path / "test_img.png")
    mask_path = str(tmp_path / "test_mask.png")

    # Create dummy RGB image
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)

    # Create dummy mask
    mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
    cv2.imwrite(mask_path, mask)

    return img_path, mask_path

def test_opencv_load(temp_opencv_data):
    img_path, mask_path = temp_opencv_data

    # Test LoadImgFromOpenCV
    loader_img = LoadImgFromOpenCV()
    results = {"img_path": img_path}
    results = loader_img.transform(results)

    assert "img" in results
    assert results["img"].shape == (100, 100, 3)
    assert results["img_shape"] == (100, 100)

    # Test LoadAnnoFromOpenCV with label_map
    loader_anno = LoadAnnoFromOpenCV()
    results = {
        "seg_map_path": mask_path,
        "label_map": {1: 10, 2: 20},
        "seg_fields": []
    }

    # Read original mask for comparison
    original_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    results = loader_anno.transform(results)

    assert "gt_seg_map" in results
    assert results["gt_seg_map"].shape == (100, 100)
    # Check remapping
    if (original_mask == 1).any():
        assert (results["gt_seg_map"][original_mask == 1] == 10).all()
    if (original_mask == 2).any():
        assert (results["gt_seg_map"][original_mask == 2] == 20).all()

def test_mha_load(itkit_dummy_dataset):
    # itkit_dummy_dataset creates 20 samples in 'image' and 'label' folders
    sample_uid = "SERIES_000"
    img_path = os.path.join(itkit_dummy_dataset, "image", f"{sample_uid}.mha")
    mask_path = os.path.join(itkit_dummy_dataset, "label", f"{sample_uid}.mha")

    # Test LoadImageFromMHA
    loader_img = LoadImageFromMHA()
    results = {"img_path": img_path}
    results = loader_img.transform(results)

    assert "img" in results
    # itkit_dummy_dataset creates size (100, 128, 128) -> [Z, Y, X]
    assert results["img"].shape == (100, 128, 128)
    assert results["img_shape"] == (100, 128, 128)

    # Test LoadMaskFromMHA
    loader_mask = LoadMaskFromMHA()
    results = {
        "seg_map_path": mask_path,
        "seg_fields": [],
        "label_map": {1: 10}
    }
    results = loader_mask.transform(results)

    assert "gt_seg_map" in results
    assert results["gt_seg_map"].shape == (100, 128, 128)
    assert "gt_seg_map" in results["seg_fields"]

def test_mha_resample(itkit_dummy_dataset):
    sample_uid = "SERIES_001"
    img_path = os.path.join(itkit_dummy_dataset, "image", f"{sample_uid}.mha")

    # Test resampling to size
    target_size = (32, 64, 64) # Z, Y, X for itkit
    loader = LoadImageFromMHA(resample_size=target_size)
    results = {"img_path": img_path}
    results = loader.transform(results)

    # SimpleITK uses XYZ, so resample_to_size might take XYZ if not careful
    # But sitk_resample_to_size from itkit.io.sitk_toolkit should handle it.
    assert results["img"].shape == target_size
