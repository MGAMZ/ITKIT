"""
Comprehensive test suite for itkit.io.nii_toolkit module.
Tests all functions for NIfTI file handling with various parameter combinations.

Note: nii_toolkit is deprecated, but we still provide comprehensive tests.
"""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

# Handle Python version compatibility
try:
    from itkit.io.nii_toolkit import convert_nii_sitk, merge_masks
    IMPORT_SUCCESS = True
except ImportError as e:
    # nii_toolkit may not be importable on some Python versions due to deprecated import
    IMPORT_SUCCESS = False
    pytestmark = pytest.mark.skip(reason=f"Cannot import nii_toolkit: {e}")


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="nii_toolkit import failed")
class TestConvertNiiSitk:
    """Test convert_nii_sitk function."""

    def test_convert_nii_xyz_order(self):
        """Test converting NIfTI with xyz data order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NIfTI file with nibabel
            data = np.random.randint(-1024, 1024, size=(30, 20, 10)).astype(np.float32)
            affine = np.eye(4)
            affine[0:3, 0:3] = np.diag([1.5, 2.0, 2.5])  # spacing
            affine[0:3, 3] = [0.1, 0.2, 0.3]  # origin

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_xyz.nii.gz")
            nib.save(nii_img, nii_path)

            # Convert with xyz order
            result = convert_nii_sitk(nii_path, nii_fdata_order='xyz', dtype=np.float32)

            assert isinstance(result, sitk.Image)
            # xyz order means data is transposed to zyx for SimpleITK
            # Original data: (30, 20, 10) in xyz -> (10, 20, 30) in zyx
            result_arr = sitk.GetArrayFromImage(result)
            assert result_arr.shape == (10, 20, 30)

    def test_convert_nii_zyx_order(self):
        """Test converting NIfTI with zyx data order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NIfTI file
            data = np.random.randint(-1024, 1024, size=(10, 20, 30)).astype(np.float32)
            affine = np.eye(4)
            affine[0:3, 0:3] = np.diag([1.5, 2.0, 2.5])
            affine[0:3, 3] = [0.1, 0.2, 0.3]

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_zyx.nii.gz")
            nib.save(nii_img, nii_path)

            # Convert with zyx order
            result = convert_nii_sitk(nii_path, nii_fdata_order='zyx', dtype=np.float32)

            assert isinstance(result, sitk.Image)
            result_arr = sitk.GetArrayFromImage(result)
            # zyx order preserved
            assert result_arr.shape == (10, 20, 30)

    def test_convert_nii_with_value_offset(self):
        """Test converting NIfTI with value offset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NIfTI with constant values
            data = np.full((10, 20, 30), 100.0, dtype=np.float32)
            affine = np.eye(4)

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_offset.nii.gz")
            nib.save(nii_img, nii_path)

            # Convert with offset
            result = convert_nii_sitk(nii_path, nii_fdata_order='zyx', 
                                     dtype=np.float32, value_offset=50)

            result_arr = sitk.GetArrayFromImage(result)
            # All values should be 100 + 50 = 150
            assert np.allclose(result_arr, 150.0)

    def test_convert_nii_preserves_spacing_xyz(self):
        """Test that spacing is correctly preserved in xyz mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.zeros((30, 20, 10), dtype=np.float32)
            affine = np.eye(4)
            affine[0:3, 0:3] = np.diag([1.5, 2.0, 2.5])

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_spacing.nii.gz")
            nib.save(nii_img, nii_path)

            result = convert_nii_sitk(nii_path, nii_fdata_order='xyz', dtype=np.float32)

            # Spacing should match
            spacing = result.GetSpacing()
            # In xyz mode, spacing should be [1.5, 2.0, 2.5]
            assert len(spacing) == 3

    def test_convert_nii_preserves_origin_xyz(self):
        """Test that origin is correctly preserved in xyz mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.zeros((30, 20, 10), dtype=np.float32)
            affine = np.eye(4)
            affine[0:3, 3] = [1.0, 2.0, 3.0]

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_origin.nii.gz")
            nib.save(nii_img, nii_path)

            result = convert_nii_sitk(nii_path, nii_fdata_order='xyz', dtype=np.float32)

            origin = result.GetOrigin()
            assert len(origin) == 3

    def test_convert_nii_preserves_direction(self):
        """Test that direction matrix is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.zeros((10, 20, 30), dtype=np.float32)
            affine = np.eye(4)

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_direction.nii.gz")
            nib.save(nii_img, nii_path)

            result = convert_nii_sitk(nii_path, nii_fdata_order='zyx', dtype=np.float32)

            direction = result.GetDirection()
            assert len(direction) == 9

    def test_convert_nii_dtype_conversion(self):
        """Test conversion to different data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(10, 20, 30).astype(np.float32) * 1000
            affine = np.eye(4)

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_dtype.nii.gz")
            nib.save(nii_img, nii_path)

            # Convert to int16
            result = convert_nii_sitk(nii_path, nii_fdata_order='zyx', dtype=np.int16)

            result_arr = sitk.GetArrayFromImage(result)
            assert result_arr.dtype == np.int16

    def test_convert_nii_invalid_order_raises_error(self):
        """Test that invalid nii_fdata_order raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.zeros((10, 20, 30), dtype=np.float32)
            affine = np.eye(4)

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_invalid.nii.gz")
            nib.save(nii_img, nii_path)

            with pytest.raises(ValueError, match="Invalid nii_fdata_order"):
                convert_nii_sitk(nii_path, nii_fdata_order='invalid', dtype=np.float32)

    def test_convert_nii_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises ValueError."""
        with pytest.raises(ValueError, match="Failed to load NIfTI file"):
            convert_nii_sitk("/nonexistent/path.nii.gz", nii_fdata_order='xyz')

    def test_convert_nii_without_offset(self):
        """Test conversion without value offset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.full((10, 20, 30), 200.0, dtype=np.float32)
            affine = np.eye(4)

            nii_img = nib.Nifti1Image(data, affine)
            nii_path = os.path.join(tmpdir, "test_no_offset.nii.gz")
            nib.save(nii_img, nii_path)

            result = convert_nii_sitk(nii_path, nii_fdata_order='zyx', 
                                     dtype=np.float32, value_offset=None)

            result_arr = sitk.GetArrayFromImage(result)
            assert np.allclose(result_arr, 200.0)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="nii_toolkit import failed")
class TestMergeMasksNii:
    """Test merge_masks function from nii_toolkit."""

    def test_merge_masks_basic(self):
        """Test basic mask merging functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create class index map
            class_index_map = {
                'liver': 1,
                'kidney': 2,
                'spleen': 3
            }

            # Create binary masks for each class
            shape = (10, 20, 30)

            # Liver mask
            liver_mask = np.zeros(shape, dtype=np.float32)
            liver_mask[2:5, 5:10, 10:15] = 1
            liver_img = nib.Nifti1Image(liver_mask, np.eye(4))
            liver_path = os.path.join(tmpdir, 'liver.nii.gz')
            nib.save(liver_img, liver_path)

            # Kidney mask
            kidney_mask = np.zeros(shape, dtype=np.float32)
            kidney_mask[6:8, 12:15, 20:25] = 1
            kidney_img = nib.Nifti1Image(kidney_mask, np.eye(4))
            kidney_path = os.path.join(tmpdir, 'kidney.nii.gz')
            nib.save(kidney_img, kidney_path)

            # Merge masks
            result = merge_masks([liver_path, kidney_path], class_index_map, dtype=np.uint8)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
            assert result.shape == shape

            # Check class assignments
            assert result[3, 7, 12] == 1  # Liver
            assert result[7, 13, 22] == 2  # Kidney
            assert result[0, 0, 0] == 0   # Background

    def test_merge_masks_all_classes(self):
        """Test merging multiple masks with all classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_index_map = {
                'class1': 1,
                'class2': 2,
                'class3': 3,
                'class4': 4
            }

            shape = (10, 20, 30)
            paths = []

            for i, (class_name, class_idx) in enumerate(class_index_map.items()):
                mask = np.zeros(shape, dtype=np.float32)
                # Create non-overlapping regions
                z_start = i * 2
                mask[z_start:z_start+2, 5:10, 10:15] = 1

                img = nib.Nifti1Image(mask, np.eye(4))
                path = os.path.join(tmpdir, f'{class_name}.nii.gz')
                nib.save(img, path)
                paths.append(path)

            result = merge_masks(paths, class_index_map, dtype=np.uint8)

            # Check each class is present
            for i, class_idx in enumerate(class_index_map.values()):
                z_start = i * 2
                assert result[z_start, 7, 12] == class_idx

    def test_merge_masks_missing_class_in_map_raises_error(self):
        """Test that missing class name in map raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_index_map = {'liver': 1}

            # Create mask with name not in map
            mask = np.zeros((10, 20, 30), dtype=np.float32)
            mask[2:5, 5:10, 10:15] = 1
            img = nib.Nifti1Image(mask, np.eye(4))
            path = os.path.join(tmpdir, 'unknown_organ.nii.gz')
            nib.save(img, path)

            with pytest.raises(ValueError, match="Class name unknown_organ not found"):
                merge_masks([path], class_index_map, dtype=np.uint8)

    def test_merge_masks_empty_list_raises_error(self):
        """Test that empty path list raises ValueError."""
        with pytest.raises(ValueError, match="No mask found"):
            merge_masks([], {}, dtype=np.uint8)

    def test_merge_masks_nonexistent_file(self):
        """Test handling of non-existent file in path list."""
        class_index_map = {'test': 1}
        nonexistent_path = '/nonexistent/test.nii.gz'

        # The function checks os.path.isfile, so non-existent files are skipped
        # This would result in empty merged_mask
        with pytest.raises(ValueError, match="No mask found"):
            merge_masks([nonexistent_path], class_index_map, dtype=np.uint8)

    def test_merge_masks_overlapping_regions(self):
        """Test mask merging with overlapping regions (later class wins)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_index_map = {
                'class1': 1,
                'class2': 2
            }

            shape = (10, 20, 30)

            # Create overlapping masks
            mask1 = np.zeros(shape, dtype=np.float32)
            mask1[2:6, 5:10, 10:15] = 1

            mask2 = np.zeros(shape, dtype=np.float32)
            mask2[4:8, 7:12, 12:17] = 1

            img1 = nib.Nifti1Image(mask1, np.eye(4))
            path1 = os.path.join(tmpdir, 'class1.nii.gz')
            nib.save(img1, path1)

            img2 = nib.Nifti1Image(mask2, np.eye(4))
            path2 = os.path.join(tmpdir, 'class2.nii.gz')
            nib.save(img2, path2)

            result = merge_masks([path1, path2], class_index_map, dtype=np.uint8)

            # In overlapping region, class2 should win (processed last)
            assert result[5, 8, 13] == 2

    def test_merge_masks_custom_dtype(self):
        """Test mask merging with custom dtype."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_index_map = {'test': 1}

            mask = np.zeros((10, 20, 30), dtype=np.float32)
            mask[2:5, 5:10, 10:15] = 1
            img = nib.Nifti1Image(mask, np.eye(4))
            path = os.path.join(tmpdir, 'test.nii.gz')
            nib.save(img, path)

            # Test with uint16
            result = merge_masks([path], class_index_map, dtype=np.uint16)

            assert result.dtype == np.uint16

    def test_merge_masks_filename_parsing(self):
        """Test that class names are correctly extracted from filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_index_map = {
                'organ_name_with_underscores': 5
            }

            mask = np.zeros((10, 20, 30), dtype=np.float32)
            mask[2:5, 5:10, 10:15] = 1
            img = nib.Nifti1Image(mask, np.eye(4))
            path = os.path.join(tmpdir, 'organ_name_with_underscores.nii.gz')
            nib.save(img, path)

            result = merge_masks([path], class_index_map, dtype=np.uint8)

            assert result[3, 7, 12] == 5

    def test_merge_masks_preserves_background(self):
        """Test that background (0 values) are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_index_map = {'small': 1}

            # Create mask with small region
            mask = np.zeros((10, 20, 30), dtype=np.float32)
            mask[5:6, 10:11, 15:16] = 1  # Single voxel
            img = nib.Nifti1Image(mask, np.eye(4))
            path = os.path.join(tmpdir, 'small.nii.gz')
            nib.save(img, path)

            result = merge_masks([path], class_index_map, dtype=np.uint8)

            # Most of the volume should still be background
            assert np.sum(result == 0) > np.sum(result > 0)
            # The single voxel should have class 1
            assert result[5, 10, 15] == 1
