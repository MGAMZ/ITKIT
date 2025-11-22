import os
import tempfile

import pytest
import SimpleITK as sitk

from itkit.process.itk_aug import AugProcessor


def create_test_image(path: str, size=(10, 10, 10), spacing=(1.0, 1.0, 1.0), direction=None):
    """Helper to create a test MHA image with optional direction."""
    img = sitk.Image(size, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    if direction:
        img.SetDirection(direction)
    sitk.WriteImage(img, path, True)


def create_test_label(path: str, size=(10, 10, 10), spacing=(1.0, 1.0, 1.0), direction=None):
    """Helper to create a test MHA label with optional direction."""
    img = sitk.Image(size, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    if direction:
        img.SetDirection(direction)
    # Add some label values
    arr = sitk.GetArrayFromImage(img)
    arr[2:5, 2:5, 2:5] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    if direction:
        img.SetDirection(direction)
    sitk.WriteImage(img, path, True)


@pytest.mark.itk_process
class TestAugProcessor:
    def test_successful_augmentation(self):
        """Test successful augmentation of image-label pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'img')
            lbl_dir = os.path.join(tmpdir, 'lbl')
            out_img_dir = os.path.join(tmpdir, 'out_img')
            out_lbl_dir = os.path.join(tmpdir, 'out_lbl')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            img_path = os.path.join(img_dir, 'test.mha')
            lbl_path = os.path.join(lbl_dir, 'test.mha')
            create_test_image(img_path)
            create_test_label(lbl_path)

            processor = AugProcessor(
                img_dir, lbl_dir, out_img_dir, out_lbl_dir,
                aug_num=2, random_rots=[0, 0, 0], mp=False
            )
            processor.process()

            # Check output files
            img_files = [f for f in os.listdir(out_img_dir) if f.endswith('.mha')]
            lbl_files = [f for f in os.listdir(out_lbl_dir) if f.endswith('.mha')]
            assert len(img_files) == 2  # aug_num
            assert len(lbl_files) == 2
            assert all('test_0.mha' in f or 'test_1.mha' in f for f in img_files)
            assert all('test_0.mha' in f or 'test_1.mha' in f for f in lbl_files)

    def test_no_output_folders(self):
        """Test when output folders are None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'img')
            lbl_dir = os.path.join(tmpdir, 'lbl')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            img_path = os.path.join(img_dir, 'test.mha')
            lbl_path = os.path.join(lbl_dir, 'test.mha')
            create_test_image(img_path)
            create_test_label(lbl_path)

            processor = AugProcessor(
                img_dir, lbl_dir, None, None,
                aug_num=1, random_rots=[0, 0, 0], mp=False
            )
            processor.process()

            # No output folders, so no files created
            # But process should not fail

    def test_multiprocessing(self):
        """Test with multiprocessing enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'img')
            lbl_dir = os.path.join(tmpdir, 'lbl')
            out_img_dir = os.path.join(tmpdir, 'out_img')
            out_lbl_dir = os.path.join(tmpdir, 'out_lbl')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            for i in range(3):
                create_test_image(os.path.join(img_dir, f'test{i}.mha'))
                create_test_label(os.path.join(lbl_dir, f'test{i}.mha'))

            processor = AugProcessor(
                img_dir, lbl_dir, out_img_dir, out_lbl_dir,
                aug_num=1, random_rots=[0, 0, 0], mp=True, workers=2
            )
            processor.process()

            img_files = [f for f in os.listdir(out_img_dir) if f.endswith('.mha')]
            lbl_files = [f for f in os.listdir(out_lbl_dir) if f.endswith('.mha')]
            assert len(img_files) == 3  # 3 pairs * 1 aug
            assert len(lbl_files) == 3

    def test_error_on_invalid_file(self):
        """Test error handling for invalid image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'img')
            lbl_dir = os.path.join(tmpdir, 'lbl')
            out_img_dir = os.path.join(tmpdir, 'out_img')
            out_lbl_dir = os.path.join(tmpdir, 'out_lbl')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            # Valid pair
            img_path = os.path.join(img_dir, 'valid.mha')
            lbl_path = os.path.join(lbl_dir, 'valid.mha')
            create_test_image(img_path)
            create_test_label(lbl_path)

            # Invalid file
            invalid_img = os.path.join(img_dir, 'invalid.mha')
            with open(invalid_img, 'w') as f:
                f.write("not an image")
            invalid_lbl = os.path.join(lbl_dir, 'invalid.mha')
            create_test_label(invalid_lbl)  # Valid label but invalid image

            processor = AugProcessor(
                img_dir, lbl_dir, out_img_dir, out_lbl_dir,
                aug_num=1, random_rots=[0, 0, 0], mp=False
            )
            processor.process()

            # Should process valid pair, skip invalid
            img_files = [f for f in os.listdir(out_img_dir) if f.endswith('.mha')]
            lbl_files = [f for f in os.listdir(out_lbl_dir) if f.endswith('.mha')]
            assert len(img_files) == 1  # Only valid
            assert len(lbl_files) == 1
