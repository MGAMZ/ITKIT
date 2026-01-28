"""
Comprehensive test suite for itkit.io.sitk_toolkit module.
Tests all functions with various parameter combinations and edge cases.
"""

import os
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk

from itkit.io.sitk_toolkit import (
    INTERPOLATOR,
    PIXEL_TYPE,
    STANDARD_DIRECTION,
    STANDARD_ORIGIN,
    merge_masks,
    nii_to_sitk,
    sitk_new_blank_image,
    sitk_resample_to_image,
    sitk_resample_to_size,
    sitk_resample_to_spacing,
    split_image_label_pairs_to_2d,
)


class TestConstants:
    """Test module-level constants."""

    def test_standard_direction(self):
        """Test STANDARD_DIRECTION constant."""
        assert STANDARD_DIRECTION == [1, 0, 0, 0, 1, 0, 0, 0, 1]
        assert len(STANDARD_DIRECTION) == 9

    def test_standard_origin(self):
        """Test STANDARD_ORIGIN constant."""
        assert STANDARD_ORIGIN == [0, 0, 0]
        assert len(STANDARD_ORIGIN) == 3

    def test_pixel_type_image(self):
        """Test PIXEL_TYPE lambda for image."""
        assert PIXEL_TYPE("image") == sitk.sitkInt16

    def test_pixel_type_label(self):
        """Test PIXEL_TYPE lambda for label."""
        assert PIXEL_TYPE("label") == sitk.sitkUInt8

    def test_interpolator_image(self):
        """Test INTERPOLATOR lambda for image."""
        assert INTERPOLATOR("image") == sitk.sitkLinear

    def test_interpolator_label(self):
        """Test INTERPOLATOR lambda for label."""
        assert INTERPOLATOR("label") == sitk.sitkNearestNeighbor


class TestSitkResampleToSpacing:
    """Test sitk_resample_to_spacing function."""

    def _create_test_image(self, size=(10, 20, 30), spacing=(1.0, 1.0, 1.0), dtype=sitk.sitkInt16):
        """Helper to create a test image."""
        arr = np.random.randint(-1024, 1024, size=size).astype(np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing[::-1])  # SimpleITK uses XYZ order
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection(STANDARD_DIRECTION)
        return img

    def test_resample_to_spacing_image_field(self):
        """Test resampling image to new spacing."""
        img = self._create_test_image(size=(10, 20, 30), spacing=(2.0, 1.0, 1.0))
        result = sitk_resample_to_spacing(img, [1.0, 0.5, 0.5], "image")

        assert isinstance(result, sitk.Image)
        # ZYX order: [1.0, 0.5, 0.5] -> XYZ order: [0.5, 0.5, 1.0]
        assert result.GetSpacing() == (0.5, 0.5, 1.0)

    def test_resample_to_spacing_label_field(self):
        """Test resampling label to new spacing."""
        arr = np.random.randint(0, 5, size=(10, 20, 30)).astype(np.uint8)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 2.0))

        result = sitk_resample_to_spacing(img, [1.0, 0.5, 0.5], "label")

        assert isinstance(result, sitk.Image)
        assert result.GetSpacing() == (0.5, 0.5, 1.0)

    def test_resample_with_minus_one_preserves_spacing(self):
        """Test that -1 preserves original spacing in that dimension."""
        img = self._create_test_image(size=(10, 20, 30), spacing=(2.0, 1.5, 1.0))
        result = sitk_resample_to_spacing(img, [-1, -1, 0.5], "image")

        # Original spacing: ZYX (2.0, 1.5, 1.0) -> XYZ (1.0, 1.5, 2.0)
        # Target spacing: ZYX [-1, -1, 0.5] -> XYZ [0.5, -1, -1]
        # Result should preserve X and Y from original
        assert result.GetSpacing() == (0.5, 1.5, 2.0)

    def test_resample_same_spacing_returns_original(self):
        """Test that resampling to same spacing returns original image."""
        img = self._create_test_image(size=(10, 20, 30), spacing=(1.0, 1.0, 1.0))
        result = sitk_resample_to_spacing(img, [1.0, 1.0, 1.0], "image")

        # Should return the same image (object identity)
        # Note: The function returns the same object if spacing is identical
        assert result.GetSize() == img.GetSize()
        assert result.GetSpacing() == img.GetSpacing()

    def test_resample_invalid_field_raises_error(self):
        """Test that invalid field parameter raises assertion error."""
        img = self._create_test_image()
        with pytest.raises(AssertionError, match="field must be one of"):
            sitk_resample_to_spacing(img, [1.0, 1.0, 1.0], "invalid")

    def test_resample_invalid_spacing_length_raises_error(self):
        """Test that invalid spacing length raises assertion error."""
        img = self._create_test_image()
        with pytest.raises(AssertionError, match="Spacing must be a 3-tuple"):
            sitk_resample_to_spacing(img, [1.0, 1.0], "image")

    def test_resample_negative_spacing_raises_error(self):
        """Test that negative spacing (not -1) raises assertion error."""
        img = self._create_test_image()
        with pytest.raises(AssertionError, match="Spacing must be positive or -1"):
            sitk_resample_to_spacing(img, [1.0, -2.0, 1.0], "image")

    def test_resample_custom_interpolator(self):
        """Test resampling with custom interpolator."""
        img = self._create_test_image()
        result = sitk_resample_to_spacing(
            img, [0.5, 0.5, 0.5], "image", interp_method=sitk.sitkNearestNeighbor
        )

        assert isinstance(result, sitk.Image)


class TestSitkResampleToImage:
    """Test sitk_resample_to_image function."""

    def _create_test_image(self, size=(10, 20, 30), spacing=(1.0, 1.0, 1.0)):
        """Helper to create a test image."""
        arr = np.random.randint(-1024, 1024, size=size).astype(np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing[::-1])
        img.SetOrigin((0.0, 0.0, 0.0))
        return img

    def test_resample_to_reference_image(self):
        """Test resampling image to match reference image."""
        img = self._create_test_image(size=(10, 20, 30), spacing=(2.0, 2.0, 2.0))
        ref = self._create_test_image(size=(20, 40, 60), spacing=(1.0, 1.0, 1.0))

        result = sitk_resample_to_image(img, ref, "image")

        assert isinstance(result, sitk.Image)
        assert result.GetSize() == ref.GetSize()
        assert result.GetSpacing() == ref.GetSpacing()
        assert result.GetOrigin() == ref.GetOrigin()

    def test_resample_label_to_reference(self):
        """Test resampling label to reference image."""
        arr = np.random.randint(0, 5, size=(10, 20, 30)).astype(np.uint8)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((2.0, 2.0, 2.0))

        ref = self._create_test_image(size=(15, 30, 45), spacing=(1.5, 1.5, 1.5))

        result = sitk_resample_to_image(img, ref, "label")

        assert isinstance(result, sitk.Image)
        assert result.GetSize() == ref.GetSize()
        assert result.GetSpacing() == ref.GetSpacing()

    def test_resample_with_default_value(self):
        """Test resampling with custom default value for out-of-bounds pixels."""
        img = self._create_test_image(size=(5, 10, 15), spacing=(2.0, 2.0, 2.0))
        ref = self._create_test_image(size=(20, 40, 60), spacing=(1.0, 1.0, 1.0))

        result = sitk_resample_to_image(img, ref, "image", default_value=-1024.0)

        assert isinstance(result, sitk.Image)

    def test_resample_with_custom_interpolator(self):
        """Test resampling with custom interpolator."""
        img = self._create_test_image()
        ref = self._create_test_image(size=(15, 30, 45))

        result = sitk_resample_to_image(
            img, ref, "label", interp_method=sitk.sitkNearestNeighbor
        )

        assert isinstance(result, sitk.Image)


class TestSitkResampleToSize:
    """Test sitk_resample_to_size function."""

    def _create_test_image(self, size=(10, 20, 30), spacing=(1.0, 1.0, 1.0)):
        """Helper to create a test image."""
        arr = np.random.randint(-1024, 1024, size=size).astype(np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing[::-1])
        img.SetOrigin((0.0, 0.0, 0.0))
        return img

    def test_resample_to_new_size(self):
        """Test resampling to new size."""
        img = self._create_test_image(size=(10, 20, 30), spacing=(1.0, 1.0, 1.0))
        result = sitk_resample_to_size(img, [20, 40, 60], "image")

        assert isinstance(result, sitk.Image)
        # ZYX order: [20, 40, 60] -> XYZ order: [60, 40, 20]
        assert result.GetSize() == (60, 40, 20)

    def test_resample_with_minus_one_preserves_size(self):
        """Test that -1 preserves original size in that dimension."""
        img = self._create_test_image(size=(10, 20, 30))
        result = sitk_resample_to_size(img, [-1, 40, -1], "image")

        # Original size: ZYX (10, 20, 30) -> XYZ (30, 20, 10)
        # Target size: ZYX [-1, 40, -1] -> should preserve Z and X
        assert result.GetSize() == (30, 40, 10)

    def test_resample_same_size_returns_original(self):
        """Test that resampling to same size returns original image."""
        img = self._create_test_image(size=(10, 20, 30))
        result = sitk_resample_to_size(img, [10, 20, 30], "image")

        # Should return the same image (same size)
        assert result.GetSize() == img.GetSize()

    def test_resample_invalid_size_length_raises_error(self):
        """Test that invalid size length raises assertion error."""
        img = self._create_test_image()
        with pytest.raises(AssertionError, match="Size must be a 3-tuple"):
            sitk_resample_to_size(img, [10, 20], "image")

    def test_resample_negative_size_raises_error(self):
        """Test that negative size (not -1) raises assertion error."""
        img = self._create_test_image()
        with pytest.raises(AssertionError, match="Size must be positive or -1"):
            sitk_resample_to_size(img, [10, -5, 20], "image")

    def test_resample_label_to_size(self):
        """Test resampling label to new size."""
        arr = np.random.randint(0, 5, size=(10, 20, 30)).astype(np.uint8)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))

        result = sitk_resample_to_size(img, [15, 30, 45], "label")

        assert isinstance(result, sitk.Image)
        assert result.GetSize() == (45, 30, 15)


class TestSitkNewBlankImage:
    """Test sitk_new_blank_image function."""

    def test_create_blank_image_default_value(self):
        """Test creating blank image with default value."""
        img = sitk_new_blank_image(
            size=(10, 20, 30),
            spacing=(1.0, 1.5, 2.0),
            direction=STANDARD_DIRECTION,
            origin=(0.0, 0.0, 0.0),
            default_value=0.0
        )

        assert isinstance(img, sitk.Image)
        # The function expects size in ZYX and converts it internally
        # The result uses the .T which swaps to ZYX order
        assert img.GetSize() == (10, 20, 30)
        assert img.GetSpacing() == (1.0, 1.5, 2.0)
        assert img.GetOrigin() == (0.0, 0.0, 0.0)

        # Check that all pixels have default value
        arr = sitk.GetArrayFromImage(img)
        assert np.all(arr == 0.0)

    def test_create_blank_image_custom_value(self):
        """Test creating blank image with custom value."""
        img = sitk_new_blank_image(
            size=(5, 10, 15),
            spacing=(0.5, 1.0, 1.5),
            direction=STANDARD_DIRECTION,
            origin=(1.0, 2.0, 3.0),
            default_value=100.0
        )

        arr = sitk.GetArrayFromImage(img)
        assert np.all(arr == 100.0)
        assert img.GetOrigin() == (1.0, 2.0, 3.0)

    def test_create_blank_image_different_sizes(self):
        """Test creating blank images with different sizes."""
        for size in [(5, 5, 5), (10, 20, 30), (64, 128, 256)]:
            img = sitk_new_blank_image(
                size=size,
                spacing=(1.0, 1.0, 1.0),
                direction=STANDARD_DIRECTION,
                origin=(0.0, 0.0, 0.0)
            )
            # The function preserves the input size order
            assert img.GetSize() == size


class TestNiiToSitk:
    """Test nii_to_sitk function."""

    def test_load_nii_image_field(self):
        """Test loading NIfTI as image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test NIfTI file
            arr = np.random.randint(-1024, 1024, size=(10, 20, 30)).astype(np.int16)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((1.0, 1.5, 2.0))
            nii_path = os.path.join(tmpdir, "test.nii.gz")
            sitk.WriteImage(img, nii_path)

            # Load using nii_to_sitk
            result = nii_to_sitk(nii_path, "image")

            assert isinstance(result, sitk.Image)
            assert result.GetSize() == img.GetSize()
            assert result.GetSpacing() == img.GetSpacing()

    def test_load_nii_label_field(self):
        """Test loading NIfTI as label."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.random.randint(0, 5, size=(10, 20, 30)).astype(np.uint8)
            img = sitk.GetImageFromArray(arr)
            nii_path = os.path.join(tmpdir, "test_label.nii.gz")
            sitk.WriteImage(img, nii_path)

            result = nii_to_sitk(nii_path, "label")

            assert isinstance(result, sitk.Image)

    def test_load_nii_with_value_offset(self):
        """Test loading NIfTI with value offset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.full((10, 20, 30), 100, dtype=np.int16)
            img = sitk.GetImageFromArray(arr)
            nii_path = os.path.join(tmpdir, "test.nii.gz")
            sitk.WriteImage(img, nii_path)

            # Load with offset
            result = nii_to_sitk(nii_path, "image", value_offset=50)

            result_arr = sitk.GetArrayFromImage(result)
            # All values should be 100 + 50 = 150
            assert np.all(result_arr == 150)

    def test_load_nii_invalid_path_raises_error(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(ValueError, match="Failed to load NIfTI file"):
            nii_to_sitk("/nonexistent/path.nii.gz", "image")


class TestMergeMasks:
    """Test merge_masks function."""

    def test_merge_masks_from_file_paths(self):
        """Test merging masks from file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create binary masks
            mask1 = np.zeros((10, 20, 30), dtype=np.uint8)
            mask1[2:5, 5:10, 10:15] = 1

            mask2 = np.zeros((10, 20, 30), dtype=np.uint8)
            mask2[6:8, 12:15, 20:25] = 1

            img1 = sitk.GetImageFromArray(mask1)
            img1.SetSpacing((1.0, 1.0, 1.0))
            path1 = os.path.join(tmpdir, "mask1.mha")
            sitk.WriteImage(img1, path1)

            img2 = sitk.GetImageFromArray(mask2)
            img2.SetSpacing((1.0, 1.0, 1.0))
            path2 = os.path.join(tmpdir, "mask2.mha")
            sitk.WriteImage(img2, path2)

            # Merge masks
            result = merge_masks([path1, path2])

            assert isinstance(result, sitk.Image)
            result_arr = sitk.GetArrayFromImage(result)

            # Check merged values
            assert result_arr[3, 7, 12] == 1  # From mask1
            assert result_arr[7, 13, 22] == 2  # From mask2
            assert result_arr[0, 0, 0] == 0   # Background

    def test_merge_masks_from_sitk_images(self):
        """Test merging masks from sitk.Image objects."""
        mask1 = np.zeros((10, 20, 30), dtype=np.uint8)
        mask1[2:5, 5:10, 10:15] = 1
        img1 = sitk.GetImageFromArray(mask1)

        mask2 = np.zeros((10, 20, 30), dtype=np.uint8)
        mask2[6:8, 12:15, 20:25] = 1
        img2 = sitk.GetImageFromArray(mask2)

        result = merge_masks([img1, img2])

        assert isinstance(result, sitk.Image)
        result_arr = sitk.GetArrayFromImage(result)
        assert result_arr[3, 7, 12] == 1
        assert result_arr[7, 13, 22] == 2

    def test_merge_masks_with_overlap_warning(self, capsys):
        """Test that overlapping masks generate a warning."""
        mask1 = np.zeros((10, 20, 30), dtype=np.uint8)
        mask1[2:5, 5:10, 10:15] = 1

        # Create overlapping mask
        mask2 = np.zeros((10, 20, 30), dtype=np.uint8)
        mask2[3:6, 7:12, 12:17] = 1  # Overlaps with mask1

        img1 = sitk.GetImageFromArray(mask1)
        img2 = sitk.GetImageFromArray(mask2)

        merge_masks([img1, img2])

        captured = capsys.readouterr()
        assert "Overlapping masks detected" in captured.out

    def test_merge_masks_empty_list_raises_error(self):
        """Test that empty mask list raises error."""
        with pytest.raises(ValueError, match="No mask found"):
            merge_masks([])

    def test_merge_masks_preserves_metadata(self):
        """Test that merged mask preserves metadata from last mask."""
        mask1 = np.zeros((10, 20, 30), dtype=np.uint8)
        mask1[2:5, 5:10, 10:15] = 1
        img1 = sitk.GetImageFromArray(mask1)
        img1.SetSpacing((1.5, 2.0, 2.5))
        img1.SetOrigin((0.1, 0.2, 0.3))

        mask2 = np.zeros((10, 20, 30), dtype=np.uint8)
        mask2[6:8, 12:15, 20:25] = 1
        img2 = sitk.GetImageFromArray(mask2)
        img2.SetSpacing((1.0, 1.5, 2.0))
        img2.SetOrigin((0.0, 0.0, 0.0))

        result = merge_masks([img1, img2])

        # The function copies information from the last processed mask
        assert result.GetSpacing() == img2.GetSpacing()
        assert result.GetOrigin() == img2.GetOrigin()


class TestSplitImageLabelPairsTo2D:
    """Test split_image_label_pairs_to_2d function."""

    def test_split_matching_image_label(self):
        """Test splitting matching 3D image and label into 2D slices."""
        # Create matching 3D volumes
        img_arr = np.random.randint(-1024, 1024, size=(10, 20, 30)).astype(np.int16)
        img = sitk.GetImageFromArray(img_arr)
        img.SetSpacing((1.0, 1.5, 2.0))
        img.SetOrigin((0.0, 0.0, 0.0))

        lbl_arr = np.random.randint(0, 5, size=(10, 20, 30)).astype(np.uint8)
        lbl = sitk.GetImageFromArray(lbl_arr)
        lbl.SetSpacing((1.0, 1.5, 2.0))
        lbl.SetOrigin((0.0, 0.0, 0.0))

        # Split into 2D slices
        slices = list(split_image_label_pairs_to_2d(img, lbl))

        assert len(slices) == 10  # Z dimension
        for img_slice, lbl_slice in slices:
            assert img_slice.shape == (20, 30)
            assert lbl_slice.shape == (20, 30)

    def test_split_validates_size_mismatch(self):
        """Test that size mismatch raises assertion error."""
        img = sitk.GetImageFromArray(np.zeros((10, 20, 30), dtype=np.int16))
        lbl = sitk.GetImageFromArray(np.zeros((10, 20, 40), dtype=np.uint8))

        with pytest.raises(AssertionError, match="Image size.*!= Label size"):
            list(split_image_label_pairs_to_2d(img, lbl))

    def test_split_validates_spacing_mismatch(self):
        """Test that spacing mismatch raises assertion error."""
        img = sitk.GetImageFromArray(np.zeros((10, 20, 30), dtype=np.int16))
        img.SetSpacing((1.0, 1.0, 1.0))

        lbl = sitk.GetImageFromArray(np.zeros((10, 20, 30), dtype=np.uint8))
        lbl.SetSpacing((2.0, 2.0, 2.0))

        with pytest.raises(AssertionError, match="Image spacing.*!= Label spacing"):
            list(split_image_label_pairs_to_2d(img, lbl))

    def test_split_validates_origin_mismatch(self):
        """Test that origin mismatch raises assertion error."""
        img = sitk.GetImageFromArray(np.zeros((10, 20, 30), dtype=np.int16))
        img.SetOrigin((0.0, 0.0, 0.0))

        lbl = sitk.GetImageFromArray(np.zeros((10, 20, 30), dtype=np.uint8))
        lbl.SetOrigin((1.0, 1.0, 1.0))

        with pytest.raises(AssertionError, match="Image origin.*!= Label origin"):
            list(split_image_label_pairs_to_2d(img, lbl))

    def test_split_slice_content_correctness(self):
        """Test that slice content is correctly extracted."""
        # Create image with known pattern
        img_arr = np.arange(10 * 20 * 30).reshape((10, 20, 30)).astype(np.int16)
        img = sitk.GetImageFromArray(img_arr)

        lbl_arr = (img_arr % 5).astype(np.uint8)
        lbl = sitk.GetImageFromArray(lbl_arr)
        lbl.SetSpacing(img.GetSpacing())
        lbl.SetOrigin(img.GetOrigin())

        slices = list(split_image_label_pairs_to_2d(img, lbl))

        # Check first slice
        img_slice_0, lbl_slice_0 = slices[0]
        assert np.array_equal(img_slice_0, img_arr[0])
        assert np.array_equal(lbl_slice_0, lbl_arr[0])

        # Check last slice
        img_slice_9, lbl_slice_9 = slices[9]
        assert np.array_equal(img_slice_9, img_arr[9])
        assert np.array_equal(lbl_slice_9, lbl_arr[9])
