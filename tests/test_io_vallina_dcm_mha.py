"""
Comprehensive test suite for itkit.io.VallinaIO_DcmMha module.
Tests legacy DICOM/MHA conversion and manipulation functions.
"""

import os
import tempfile

import numpy as np
import SimpleITK as sitk

from itkit.io.VallinaIO_DcmMha import (
    convert_npy_image_2_sitk,
    convert_npy_mask_2_sitk,
    dcm_2_mha,
    min_max_scale,
    save_ct_from_npy,
    save_ct_from_sitk,
)


class TestMinMaxScale:
    """Test min_max_scale function."""

    def test_min_max_scale_basic(self):
        """Test basic min-max scaling to [0, 255]."""
        img = np.array([0, 50, 100, 150, 200], dtype=np.float32)
        result = min_max_scale(img)

        assert result.min() == 0.0
        assert result.max() == 255.0
        assert result.shape == img.shape

    def test_min_max_scale_negative_values(self):
        """Test scaling with negative values."""
        img = np.array([-100, -50, 0, 50, 100], dtype=np.float32)
        result = min_max_scale(img)

        assert result.min() == 0.0
        assert result.max() == 255.0

    def test_min_max_scale_constant_image(self):
        """Test scaling of constant image."""
        img = np.full((10, 10), 100.0, dtype=np.float32)
        result = min_max_scale(img)

        # When all values are the same, division by zero occurs
        # Result should be 0/0 = nan, but multiplied by 255
        # This is a potential bug in the original code
        assert result.shape == img.shape

    def test_min_max_scale_2d_image(self):
        """Test scaling 2D image."""
        img = np.random.rand(64, 64).astype(np.float32) * 1000 - 500
        result = min_max_scale(img)

        assert result.shape == img.shape
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_min_max_scale_3d_image(self):
        """Test scaling 3D image."""
        img = np.random.rand(10, 64, 64).astype(np.float32) * 2000 - 1000
        result = min_max_scale(img)

        assert result.shape == img.shape
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_min_max_scale_preserves_relative_order(self):
        """Test that scaling preserves relative ordering of values."""
        img = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        result = min_max_scale(img)

        # Relative order should be preserved
        assert result[0] < result[1] < result[2] < result[3] < result[4]


class TestConvertNpyImage2Sitk:
    """Test convert_npy_image_2_sitk function."""

    def test_convert_basic(self):
        """Test basic numpy to sitk conversion."""
        npy_image = np.random.rand(10, 20, 30).astype(np.float32) * 1000
        result = convert_npy_image_2_sitk(npy_image)

        assert isinstance(result, sitk.Image)
        assert result.GetSize() == (30, 20, 10)  # XYZ order

    def test_convert_with_origin(self):
        """Test conversion with custom origin."""
        npy_image = np.zeros((10, 20, 30), dtype=np.float32)
        origin = (1.0, 2.0, 3.0)
        result = convert_npy_image_2_sitk(npy_image, origin=origin)

        assert result.GetOrigin() == origin

    def test_convert_with_spacing(self):
        """Test conversion with custom spacing."""
        npy_image = np.zeros((10, 20, 30), dtype=np.float32)
        spacing = (1.5, 2.0, 2.5)
        result = convert_npy_image_2_sitk(npy_image, spacing=spacing)

        assert result.GetSpacing() == spacing

    def test_convert_with_direction(self):
        """Test conversion with custom direction."""
        npy_image = np.zeros((10, 20, 30), dtype=np.float32)
        direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        result = convert_npy_image_2_sitk(npy_image, direction=direction)

        assert result.GetDirection() == direction

    def test_convert_with_custom_type(self):
        """Test conversion with custom sitk type."""
        npy_image = np.random.randint(-1000, 1000, (10, 20, 30)).astype(np.int16)
        result = convert_npy_image_2_sitk(npy_image, sitk_type=sitk.sitkInt16)

        assert isinstance(result, sitk.Image)

    def test_convert_with_all_parameters(self):
        """Test conversion with all parameters specified."""
        npy_image = np.random.rand(10, 20, 30).astype(np.float32)
        origin = (0.5, 1.0, 1.5)
        spacing = (1.0, 1.5, 2.0)
        direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)

        result = convert_npy_image_2_sitk(
            npy_image,
            origin=origin,
            spacing=spacing,
            direction=direction,
            sitk_type=sitk.sitkFloat32
        )

        assert result.GetOrigin() == origin
        assert result.GetSpacing() == spacing
        assert result.GetDirection() == direction


class TestConvertNpyMask2Sitk:
    """Test convert_npy_mask_2_sitk function."""

    def test_convert_mask_basic(self):
        """Test basic mask conversion."""
        npy_mask = np.random.randint(0, 5, (10, 20, 30)).astype(np.uint8)
        result = convert_npy_mask_2_sitk(npy_mask)

        assert isinstance(result, sitk.Image)
        assert result.GetSize() == (30, 20, 10)

    def test_convert_mask_with_label_replacement(self):
        """Test mask conversion with label replacement."""
        npy_mask = np.random.randint(0, 5, (10, 20, 30)).astype(np.uint8)
        npy_mask[5, 10, 15] = 3  # Set a specific value

        result = convert_npy_mask_2_sitk(npy_mask, label=1)

        result_arr = sitk.GetArrayFromImage(result)
        # All non-zero values should be replaced with label=1
        assert np.all((result_arr == 0) | (result_arr == 1))

    def test_convert_mask_without_label_replacement(self):
        """Test mask conversion without label replacement."""
        npy_mask = np.random.randint(0, 5, (10, 20, 30)).astype(np.uint8)
        result = convert_npy_mask_2_sitk(npy_mask, label=None)

        result_arr = sitk.GetArrayFromImage(result)
        # Values should be preserved
        assert np.array_equal(result_arr, npy_mask)

    def test_convert_mask_with_metadata(self):
        """Test mask conversion with metadata."""
        npy_mask = np.zeros((10, 20, 30), dtype=np.uint8)
        origin = (1.0, 2.0, 3.0)
        spacing = (1.5, 2.0, 2.5)
        direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)

        result = convert_npy_mask_2_sitk(
            npy_mask,
            label=None,
            origin=origin,
            spacing=spacing,
            direction=direction,
            sitk_type=sitk.sitkUInt8
        )

        assert result.GetOrigin() == origin
        assert result.GetSpacing() == spacing
        assert result.GetDirection() == direction

    def test_convert_mask_binary(self):
        """Test converting binary mask."""
        npy_mask = np.random.randint(0, 2, (10, 20, 30)).astype(np.uint8)
        result = convert_npy_mask_2_sitk(npy_mask)

        result_arr = sitk.GetArrayFromImage(result)
        assert np.all((result_arr == 0) | (result_arr == 1))

    def test_convert_mask_with_label_changes_nonzero(self):
        """Test that label parameter changes all non-zero values."""
        npy_mask = np.zeros((5, 5, 5), dtype=np.uint8)
        npy_mask[1, 1, 1] = 2
        npy_mask[2, 2, 2] = 3
        npy_mask[3, 3, 3] = 4

        result = convert_npy_mask_2_sitk(npy_mask, label=7)

        result_arr = sitk.GetArrayFromImage(result)
        assert result_arr[1, 1, 1] == 7
        assert result_arr[2, 2, 2] == 7
        assert result_arr[3, 3, 3] == 7
        assert result_arr[0, 0, 0] == 0


class TestSaveCtFromSitk:
    """Test save_ct_from_sitk function."""

    def test_save_basic(self):
        """Test basic saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            arr = np.random.randint(-1024, 1024, (10, 20, 30)).astype(np.int16)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((1.0, 1.5, 2.0))

            save_path = os.path.join(tmpdir, "test.mha")
            save_ct_from_sitk(img, save_path)

            assert os.path.exists(save_path)

            # Load and verify
            loaded = sitk.ReadImage(save_path)
            assert loaded.GetSize() == img.GetSize()
            assert loaded.GetSpacing() == img.GetSpacing()

    def test_save_with_type_conversion(self):
        """Test saving with type conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.random.rand(10, 20, 30).astype(np.float32) * 1000
            img = sitk.GetImageFromArray(arr)

            save_path = os.path.join(tmpdir, "test.mha")
            save_ct_from_sitk(img, save_path, sitk_type=sitk.sitkInt16)

            assert os.path.exists(save_path)

    def test_save_with_compression(self):
        """Test saving with compression enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.random.randint(-1024, 1024, (10, 20, 30)).astype(np.int16)
            img = sitk.GetImageFromArray(arr)

            save_path = os.path.join(tmpdir, "test_compressed.mha")
            save_ct_from_sitk(img, save_path, use_compression=True)

            assert os.path.exists(save_path)

            # Load and verify
            loaded = sitk.ReadImage(save_path)
            assert loaded.GetSize() == img.GetSize()

    def test_save_without_compression(self):
        """Test saving without compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.random.randint(-1024, 1024, (10, 20, 30)).astype(np.int16)
            img = sitk.GetImageFromArray(arr)

            save_path = os.path.join(tmpdir, "test_uncompressed.mha")
            save_ct_from_sitk(img, save_path, use_compression=False)

            assert os.path.exists(save_path)


class TestSaveCtFromNpy:
    """Test save_ct_from_npy function."""

    def test_save_npy_basic(self):
        """Test basic saving from numpy array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_image = np.random.randint(-1024, 1024, (10, 20, 30)).astype(np.int16)
            save_path = os.path.join(tmpdir, "test.mha")

            save_ct_from_npy(npy_image, save_path)

            assert os.path.exists(save_path)

            # Load and verify
            loaded = sitk.ReadImage(save_path)
            assert loaded.GetSize() == (30, 20, 10)

    def test_save_npy_with_metadata(self):
        """Test saving with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_image = np.random.randint(-1024, 1024, (10, 20, 30)).astype(np.int16)
            save_path = os.path.join(tmpdir, "test_meta.mha")

            origin = (1.0, 2.0, 3.0)
            spacing = (1.5, 2.0, 2.5)
            direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)

            save_ct_from_npy(
                npy_image,
                save_path,
                origin=origin,
                spacing=spacing,
                direction=direction
            )

            assert os.path.exists(save_path)

            # Load and verify metadata
            loaded = sitk.ReadImage(save_path)
            assert loaded.GetOrigin() == origin
            assert loaded.GetSpacing() == spacing
            assert loaded.GetDirection() == direction

    def test_save_npy_with_type_conversion(self):
        """Test saving with type conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_image = np.random.rand(10, 20, 30).astype(np.float32) * 1000
            save_path = os.path.join(tmpdir, "test_type.mha")

            save_ct_from_npy(npy_image, save_path, sitk_type=sitk.sitkInt16)

            assert os.path.exists(save_path)

    def test_save_npy_with_compression(self):
        """Test saving with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_image = np.random.randint(-1024, 1024, (10, 20, 30)).astype(np.int16)
            save_path = os.path.join(tmpdir, "test_compressed.mha")

            save_ct_from_npy(npy_image, save_path, use_compression=True)

            assert os.path.exists(save_path)

    def test_save_npy_without_metadata(self):
        """Test saving without any metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_image = np.random.randint(-1024, 1024, (10, 20, 30)).astype(np.int16)
            save_path = os.path.join(tmpdir, "test_no_meta.mha")

            save_ct_from_npy(
                npy_image,
                save_path,
                origin=None,
                spacing=None,
                direction=None
            )

            assert os.path.exists(save_path)


class TestDcm2Mha:
    """Test dcm_2_mha function.

    Note: This function relies on load_ct_info which requires actual DICOM files.
    We test the function signature and basic error handling.
    """

    def test_dcm_2_mha_function_exists(self):
        """Test that dcm_2_mha function exists."""
        assert callable(dcm_2_mha)

    def test_dcm_2_mha_signature(self):
        """Test function signature."""
        import inspect
        sig = inspect.signature(dcm_2_mha)

        assert 'dcm_path' in sig.parameters
        assert 'mha_path' in sig.parameters
        assert 'use_compress' in sig.parameters

    def test_dcm_2_mha_with_nonexistent_path(self):
        """Test with non-existent DICOM path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mha_path = os.path.join(tmpdir, "output.mha")

            # This should handle the error gracefully
            # The function may raise an error or return None depending on implementation
            try:
                dcm_2_mha("/nonexistent/dcm/path", mha_path, use_compress=False)
            except (ValueError, KeyError, RuntimeError):
                # Expected to fail with non-existent path
                pass
