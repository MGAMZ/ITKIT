import os
import sys
import tempfile

import pytest
import SimpleITK as sitk

from itkit.process.itk_orient import DatasetOrientProcessor, OrientProcessor, main


def create_test_image(path: str, size=(10, 10, 10), spacing=(1.0, 1.0, 1.0), direction=None):
    """Helper to create a test MHA image with optional direction."""
    img = sitk.Image(size, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    if direction:
        img.SetDirection(direction)
    sitk.WriteImage(img, path)

@pytest.mark.itk_process
class TestOrientProcessor:
    def test_successful_orientation(self):
        """Test successful orientation of a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            os.makedirs(src_dir)

            img_path = os.path.join(src_dir, 'test.mha')
            create_test_image(img_path, direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))  # Identity

            processor = OrientProcessor(src_dir, dst_dir, 'LPI', field='image', mp=False)
            processor.process()

            dst_path = os.path.join(dst_dir, 'test.mha')
            assert os.path.exists(dst_path)
            img = sitk.ReadImage(dst_path)
            # Check if oriented to LPI (SimpleITK direction for LPI)
            expected_direction = sitk.DICOMOrient(sitk.Image((10,10,10), sitk.sitkUInt8), 'LPI').GetDirection()
            assert img.GetDirection() == expected_direction

    def test_skip_existing_file(self):
        """Test skipping when destination file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            img_path = os.path.join(src_dir, 'test.mha')
            create_test_image(img_path)

            dst_path = os.path.join(dst_dir, 'test.mha')
            create_test_image(dst_path)  # Pre-create dest

            processor = OrientProcessor(src_dir, dst_dir, 'LPI', field='image', mp=False)
            processor.process()

            # Should not overwrite, but since it's the same, check mtime or something, but for now, just ensure no error
            assert os.path.exists(dst_path)

    def test_error_on_invalid_file(self):
        """Test error handling for invalid image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            os.makedirs(src_dir)

            invalid_path = os.path.join(src_dir, 'invalid.mha')
            with open(invalid_path, 'w') as f:
                f.write("not an image")

            processor = OrientProcessor(src_dir, dst_dir, 'LPI', field='image', mp=False)
            processor.process()

            # Should not create dest file
            dst_path = os.path.join(dst_dir, 'invalid.mha')
            assert not os.path.exists(dst_path)

    def test_multiprocessing(self):
        """Test with multiprocessing enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            os.makedirs(src_dir)

            for i in range(5):
                create_test_image(os.path.join(src_dir, f'test{i}.mha'))

            processor = OrientProcessor(src_dir, dst_dir, 'LPI', field='image', mp=True, workers=2)
            processor.process()

            for i in range(5):
                dst_path = os.path.join(dst_dir, f'test{i}.mha')
                assert os.path.exists(dst_path)

    def test_main_invalid_src_dir(self, capsys):
        """Test main with non-existent source directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dst_dir = os.path.join(tmpdir, 'dst')
            original_argv = sys.argv
            sys.argv = ['itk_orient.py', '/nonexistent', dst_dir, 'LPI']
            try:
                main()
                captured = capsys.readouterr()
                assert "Source directory does not exist" in captured.out
            finally:
                sys.argv = original_argv

    def test_main_same_src_dst(self, capsys):
        """Test main with same source and destination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_argv = sys.argv
            sys.argv = ['itk_orient.py', tmpdir, tmpdir, 'LPI']
            try:
                main()
                captured = capsys.readouterr()
                assert "Source and destination directories cannot be the same!" in captured.out
            finally:
                sys.argv = original_argv

    def test_main_success(self, capsys):
        """Test main function for successful run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            os.makedirs(src_dir)
            create_test_image(os.path.join(src_dir, 'test.mha'))

            original_argv = sys.argv
            sys.argv = ['itk_orient.py', src_dir, dst_dir, 'LPI']
            try:
                main()
                captured = capsys.readouterr()
                # Assuming no error prints
                assert os.path.exists(os.path.join(dst_dir, 'test.mha'))
            finally:
                sys.argv = original_argv


@pytest.mark.itk_process
class TestDatasetOrientProcessor:
    def test_dataset_mode_successful_orientation(self):
        """Test successful orientation of a dataset with image/label structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            
            # Create standard ITKIT structure
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            # Create matching image and label files
            create_test_image(os.path.join(img_dir, 'case01.mha'), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
            create_test_image(os.path.join(lbl_dir, 'case01.mha'), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
            create_test_image(os.path.join(img_dir, 'case02.mha'), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
            create_test_image(os.path.join(lbl_dir, 'case02.mha'), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
            
            processor = DatasetOrientProcessor(src_dir, dst_dir, 'LPI', mp=False)
            processor.process()
            
            # Check output structure
            assert os.path.exists(os.path.join(dst_dir, 'image', 'case01.mha'))
            assert os.path.exists(os.path.join(dst_dir, 'label', 'case01.mha'))
            assert os.path.exists(os.path.join(dst_dir, 'image', 'case02.mha'))
            assert os.path.exists(os.path.join(dst_dir, 'label', 'case02.mha'))
            
            # Verify orientation
            img = sitk.ReadImage(os.path.join(dst_dir, 'image', 'case01.mha'))
            expected_direction = sitk.DICOMOrient(sitk.Image((10,10,10), sitk.sitkUInt8), 'LPI').GetDirection()
            assert img.GetDirection() == expected_direction

    def test_dataset_mode_skip_existing(self):
        """Test skipping when destination files already exist in dataset mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            
            # Create source structure
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'case01.mha'))
            create_test_image(os.path.join(lbl_dir, 'case01.mha'))
            
            # Pre-create destination files
            dst_img_dir = os.path.join(dst_dir, 'image')
            dst_lbl_dir = os.path.join(dst_dir, 'label')
            os.makedirs(dst_img_dir)
            os.makedirs(dst_lbl_dir)
            create_test_image(os.path.join(dst_img_dir, 'case01.mha'))
            create_test_image(os.path.join(dst_lbl_dir, 'case01.mha'))
            
            processor = DatasetOrientProcessor(src_dir, dst_dir, 'LPI', mp=False)
            processor.process()
            
            # Should not error, files should still exist
            assert os.path.exists(os.path.join(dst_dir, 'image', 'case01.mha'))
            assert os.path.exists(os.path.join(dst_dir, 'label', 'case01.mha'))

    def test_dataset_mode_multiprocessing(self):
        """Test dataset mode with multiprocessing enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            
            # Create source structure
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            # Create multiple cases
            for i in range(5):
                create_test_image(os.path.join(img_dir, f'case{i:02d}.mha'))
                create_test_image(os.path.join(lbl_dir, f'case{i:02d}.mha'))
            
            processor = DatasetOrientProcessor(src_dir, dst_dir, 'LPI', mp=True, workers=2)
            processor.process()
            
            # Check all outputs exist
            for i in range(5):
                assert os.path.exists(os.path.join(dst_dir, 'image', f'case{i:02d}.mha'))
                assert os.path.exists(os.path.join(dst_dir, 'label', f'case{i:02d}.mha'))

    def test_main_dataset_mode_success(self, capsys):
        """Test main function with dataset mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            
            # Create standard ITKIT structure
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'case01.mha'))
            create_test_image(os.path.join(lbl_dir, 'case01.mha'))
            
            original_argv = sys.argv
            sys.argv = ['itk_orient.py', src_dir, dst_dir, 'LPI', '--field', 'dataset']
            try:
                main()
                captured = capsys.readouterr()
                # Check outputs exist
                assert os.path.exists(os.path.join(dst_dir, 'image', 'case01.mha'))
                assert os.path.exists(os.path.join(dst_dir, 'label', 'case01.mha'))
            finally:
                sys.argv = original_argv

    def test_main_dataset_mode_missing_structure(self, capsys):
        """Test main function with dataset mode but missing image/label folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'src')
            dst_dir = os.path.join(tmpdir, 'dst')
            os.makedirs(src_dir)
            
            # Don't create image/label subfolders
            
            original_argv = sys.argv
            sys.argv = ['itk_orient.py', src_dir, dst_dir, 'LPI', '--field', 'dataset']
            try:
                main()
                captured = capsys.readouterr()
                assert "Error: Dataset mode requires 'image' and 'label' subfolders" in captured.out
            finally:
                sys.argv = original_argv

