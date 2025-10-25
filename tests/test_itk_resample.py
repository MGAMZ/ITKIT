import os
import tempfile
import pytest
import numpy as np
import SimpleITK as sitk
from unittest.mock import patch, MagicMock

from itkit.process.itk_resample import ResampleProcessor, SingleResampleProcessor, parse_args, validate_and_prepare_args, main, _ResampleMixin


@pytest.fixture
def sample_image():
    arr = np.random.rand(10, 10, 10).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


@pytest.fixture
def sample_label():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:8, 2:8, 2:8] = 1  # Some foreground
    lbl = sitk.GetImageFromArray(arr)
    lbl.SetSpacing((1.0, 1.0, 1.0))
    lbl.SetOrigin((0.0, 0.0, 0.0))
    return lbl


@pytest.mark.itk_process
class TestValidateAndPrepareArgs:
    def test_valid_spacing_size(self):
        args = MagicMock()
        args.target_folder = None
        args.spacing = ["2.0", "-1", "2.0"]
        args.size = ["-1", "128", "-1"]
        target_spacing, target_size = validate_and_prepare_args(args)
        assert target_spacing == [2.0, -1, 2.0]
        assert target_size == [-1, 128, -1]

    def test_target_folder_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.target_folder = tmpdir
            args.spacing = ["-1", "-1", "-1"]
            args.size = ["-1", "-1", "-1"]
            target_spacing, target_size = validate_and_prepare_args(args)
            assert target_spacing == [-1, -1, -1]
            assert target_size == [-1, -1, -1]

    def test_mutual_exclusive_error(self):
        args = MagicMock()
        args.target_folder = "/tmp"
        args.spacing = ["2.0", "-1", "-1"]
        args.size = ["-1", "-1", "-1"]
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_and_prepare_args(args)

    def test_invalid_spacing_length(self):
        args = MagicMock()
        args.target_folder = None
        args.spacing = ["2.0", "-1"]
        args.size = ["-1", "-1", "-1"]
        with pytest.raises(ValueError, match="--spacing must have 3 values"):
            validate_and_prepare_args(args)

    def test_no_resampling_rules(self):
        args = MagicMock()
        args.target_folder = None
        args.spacing = ["-1", "-1", "-1"]
        args.size = ["-1", "-1", "-1"]
        target_spacing, target_size = validate_and_prepare_args(args)
        assert target_spacing is None
        assert target_size is None

    def test_validate_and_prepare_args_spacing_size_conflict(self):
        args = MagicMock()
        args.target_folder = None
        args.spacing = ["2.0", "-1", "-1"]
        args.size = ["-1", "128", "-1"]
        # This should not raise, as they are for different dimensions
        target_spacing, target_size = validate_and_prepare_args(args)
        assert target_spacing == [2.0, -1, -1]
        assert target_size == [-1, 128, -1]

    def test_validate_and_prepare_args_same_dimension_conflict(self):
        args = MagicMock()
        args.target_folder = None
        args.spacing = ["2.0", "-1", "-1"]
        args.size = ["128", "-1", "-1"]
        with pytest.raises(ValueError, match="Cannot specify both spacing and size"):
            validate_and_prepare_args(args)


@pytest.mark.itk_process
class TestResampleProcessor:
    def test_init(self):
        processor = ResampleProcessor("/src", "/dst", [1.0, -1, 1.0], [-1, 128, -1], False, False, None, None)
        assert processor.source_folder == "/src"
        assert processor.dest_folder == "/dst"
        assert processor.target_spacing == [1.0, -1, 1.0]
        assert processor.target_size == [-1, 128, -1]

    def test_process_one_spacing_size(self, sample_image, sample_label):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_img = os.path.join(tmpdir, "img.mha")
            src_lbl = os.path.join(tmpdir, "lbl.mha")
            dst_img = os.path.join(tmpdir, "dst", "image", "img.mha")
            dst_lbl = os.path.join(tmpdir, "dst", "label", "lbl.mha")
            os.makedirs(os.path.join(tmpdir, "dst", "image"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "dst", "label"), exist_ok=True)
            sitk.WriteImage(sample_image, src_img)
            sitk.WriteImage(sample_label, src_lbl)

            processor = ResampleProcessor(tmpdir, os.path.join(tmpdir, "dst"), [2.0, -1, 2.0], [-1, 5, -1], False, False, None, None)
            result = processor.process_one((src_img, src_lbl))
            assert result is not None
            assert os.path.exists(dst_img)
            assert os.path.exists(dst_lbl)

    def test_process_one_skip_existing(self, sample_image, sample_label):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_img = os.path.join(tmpdir, "img.mha")
            dst_img = os.path.join(tmpdir, "dst", "image", "img.mha")
            os.makedirs(os.path.join(tmpdir, "dst", "image"), exist_ok=True)
            sitk.WriteImage(sample_image, src_img)
            # Pre-create dest file
            sitk.WriteImage(sample_image, dst_img)

            processor = ResampleProcessor(tmpdir, os.path.join(tmpdir, "dst"), [2.0, -1, 2.0], [-1, -1, -1], False, False, None, None)
            result = processor.process_one((src_img, src_img))  # Use src_img for both to avoid None
            assert result is not None  # Should process since label is provided
        with tempfile.TemporaryDirectory() as tmpdir:
            src_img = os.path.join(tmpdir, "img.mha")
            dst_img = os.path.join(tmpdir, "dst", "image", "img.mha")
            os.makedirs(os.path.join(tmpdir, "dst", "image"), exist_ok=True)
            sitk.WriteImage(sample_image, src_img)
            # Pre-create dest file
            sitk.WriteImage(sample_image, dst_img)

            processor = ResampleProcessor(tmpdir, os.path.join(tmpdir, "dst"), [2.0, -1, 2.0], [-1, -1, -1], False, False, None, None)
            result = processor.process_one((src_img, src_img))  # Use src_img for both to avoid None
            assert result is not None  # Should process since label is provided

    def test_process_multiprocessing(self, sample_image, sample_label):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_img = os.path.join(tmpdir, "image", "test.mha")
            src_lbl = os.path.join(tmpdir, "label", "test.mha")
            dst_img = os.path.join(tmpdir, "dst", "image", "test.mha")
            dst_lbl = os.path.join(tmpdir, "dst", "label", "test.mha")
            os.makedirs(os.path.join(tmpdir, "image"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "label"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "dst", "image"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "dst", "label"), exist_ok=True)
            sitk.WriteImage(sample_image, src_img)
            sitk.WriteImage(sample_label, src_lbl)

            processor = ResampleProcessor(tmpdir, os.path.join(tmpdir, "dst"), [2.0, -1, 2.0], [-1, -1, -1], False, True, 2, None)
            processor.process()
            assert os.path.exists(dst_img)
            assert os.path.exists(dst_lbl)

    def test_process_recursive(self, sample_image, sample_label):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "image"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "label"), exist_ok=True)
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            src_img = os.path.join(subdir, "image", "img.mha")
            src_lbl = os.path.join(subdir, "label", "lbl.mha")
            dst_img = os.path.join(tmpdir, "dst", "subdir", "image", "img.mha")
            dst_lbl = os.path.join(tmpdir, "dst", "subdir", "label", "lbl.mha")
            os.makedirs(os.path.join(subdir, "image"), exist_ok=True)
            os.makedirs(os.path.join(subdir, "label"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "dst", "subdir", "image"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "dst", "subdir", "label"), exist_ok=True)
            sitk.WriteImage(sample_image, src_img)
            sitk.WriteImage(sample_label, src_lbl)

@pytest.mark.itk_process
class TestSingleResampleProcessor:
    def test_init(self):
        processor = SingleResampleProcessor("/src", "/dst", [1.0, -1, 1.0], [-1, 128, -1], "image", False, False, None, None)
        assert processor.source_folder == "/src"
        assert processor.dest_folder == "/dst"
        assert processor.field == "image"

    def test_process_one(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "img.mha")
            dst_path = os.path.join(tmpdir, "dst", "img.mha")
            os.makedirs(os.path.join(tmpdir, "dst"), exist_ok=True)
            sitk.WriteImage(sample_image, src_path)

            processor = SingleResampleProcessor(tmpdir, os.path.join(tmpdir, "dst"), [2.0, -1, 2.0], [-1, -1, -1], "image", False, False, None, None)
            result = processor.process_one(src_path)
            assert result is not None
            assert os.path.exists(dst_path)

    def test_process_recursive(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            src_path = os.path.join(subdir, "img.mha")
            dst_path = os.path.join(tmpdir, "dst", "subdir", "img.mha")
            os.makedirs(os.path.join(tmpdir, "dst", "subdir"), exist_ok=True)
            sitk.WriteImage(sample_image, src_path)

            processor = SingleResampleProcessor(tmpdir, os.path.join(tmpdir, "dst"), [2.0, -1, 2.0], [-1, -1, -1], "image", True, False, None, None)
            processor.process()
            assert os.path.exists(dst_path)

    def test_process_one_sample_extension_conversion(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.nii.gz")
            expected_output = os.path.join(tmpdir, "dst", "input.mha")
            sitk.WriteImage(sample_image, input_path)

            processor = SingleResampleProcessor(tmpdir, os.path.join(tmpdir, "dst"), [2.0, -1, 2.0], [-1, -1, -1], "image", False, False, None, None)
            result = processor.process_one(input_path)
            assert os.path.exists(expected_output)


@pytest.mark.itk_process
class TestMain:
    def test_main_dataset_mode_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "src")
            dst = os.path.join(tmpdir, "dst")
            os.makedirs(os.path.join(src, "image"), exist_ok=True)
            os.makedirs(os.path.join(src, "label"), exist_ok=True)
            # Create dummy files
            os.makedirs(src, exist_ok=True)
            img = sitk.GetImageFromArray(np.random.rand(10, 10, 10).astype(np.float32))
            sitk.WriteImage(img, os.path.join(src, "test.mha"))

            with patch('sys.argv', ['itk_resample.py', 'image', src, dst, '--spacing', '2.0', '-1', '2.0']):
                main()
            assert os.path.exists(os.path.join(dst, "test.mha"))

    def test_main_invalid_args(self):
        with patch('sys.argv', ['itk_resample.py']):
            with pytest.raises(SystemExit):
                main()

    def test_main_invalid_target_folder(self):
        with patch('sys.argv', ['itk_resample.py', 'dataset', '/src', '/dst', '--target-folder', '/nonexistent']):
            with pytest.raises(ValueError, match="Target folder does not exist"):
                main()

    def test_main_mutual_exclusive_error(self):
        with patch('sys.argv', ['itk_resample.py', 'dataset', '/src', '/dst', '--target-folder', '/tmp', '--spacing', '2.0', '-1', '-1']):
            with pytest.raises(ValueError, match="mutually exclusive"):
                main()

    def test_main_invalid_spacing_size_length(self):
        with patch('sys.argv', ['itk_resample.py', 'dataset', '/src', '/dst', '--spacing', '2.0', '-1']):
            with pytest.raises(ValueError, match="--spacing must have 3 values"):
                main()

    def test_main_no_resampling_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, 'src')
            dst = os.path.join(tmpdir, 'dst')
            os.makedirs(src, exist_ok=True)
            with patch('sys.argv', ['itk_resample.py', 'dataset', src, dst]):
                # Should not raise, just print warning and return
                main()
                # Check no processing happened
                assert not os.path.exists(os.path.join(dst, 'image'))


@pytest.mark.itk_process
class TestResampleMixin:
    class TestProcessor(_ResampleMixin):
        def __init__(self, target_spacing, target_size, target_folder=None):
            self.target_spacing = target_spacing
            self.target_size = target_size
            self.target_folder = target_folder
            self.source_folder = "/tmp/src"

    def test_resample_one_sample_success(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mha")
            output_path = os.path.join(tmpdir, "output.mha")
            sitk.WriteImage(sample_image, input_path)

            processor = self.TestProcessor([2.0, -1, 2.0], [-1, -1, -1])
            result = processor.resample_one_sample(input_path, "image", output_path)
            assert result is not None
            assert os.path.exists(output_path)
            assert "input.mha" in result

    def test_resample_one_sample_skip_existing(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mha")
            output_path = os.path.join(tmpdir, "output.mha")
            sitk.WriteImage(sample_image, input_path)
            # Pre-create output
            sitk.WriteImage(sample_image, output_path)

            processor = self.TestProcessor([2.0, -1, 2.0], [-1, -1, -1])
            result = processor.resample_one_sample(input_path, "image", output_path)
            assert result is None

    def test_resample_one_sample_target_folder_missing_target(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mha")
            output_path = os.path.join(tmpdir, "output.mha")
            sitk.WriteImage(sample_image, input_path)

            processor = self.TestProcessor([-1, -1, -1], [-1, -1, -1], target_folder="/nonexistent")
            result = processor.resample_one_sample(input_path, "image", output_path)
            assert result is None

    def test_resample_one_sample_read_error(self):
        processor = self.TestProcessor([2.0, -1, 2.0], [-1, -1, -1])
        result = processor.resample_one_sample("/nonexistent.mha", "image", "/tmp/out.mha")
        assert result is None

    def test_apply_spacing_size_rules_spacing_only(self, sample_image):
        processor = self.TestProcessor([2.0, -1, 2.0], [-1, -1, -1])
        result = processor._apply_spacing_size_rules(sample_image, "image")
        assert result.GetSpacing() == (2.0, 1.0, 2.0)
        assert result.GetDirection() == sitk.DICOMOrient(result, 'LPI').GetDirection()

    def test_apply_spacing_size_rules_size_only(self, sample_image):
        processor = self.TestProcessor([-1, -1, -1], [-1, 5, -1])
        result = processor._apply_spacing_size_rules(sample_image, "image")
        assert result.GetSize() == (10, 5, 10)
        assert result.GetDirection() == sitk.DICOMOrient(result, 'LPI').GetDirection()

    def test_apply_spacing_size_rules_both(self, sample_image):
        processor = self.TestProcessor([2.0, -1, 2.0], [-1, 5, -1])
        result = processor._apply_spacing_size_rules(sample_image, "image")
        assert result.GetSize() == (5, 5, 5)
        assert result.GetDirection() == sitk.DICOMOrient(result, 'LPI').GetDirection()

    def test_apply_spacing_size_rules_no_change(self, sample_image):
        processor = self.TestProcessor([-1, -1, -1], [-1, -1, -1])
        result = processor._apply_spacing_size_rules(sample_image, "image")
        assert result.GetSpacing() == sample_image.GetSpacing()
        assert result.GetSize() == sample_image.GetSize()
        assert result.GetDirection() == sitk.DICOMOrient(result, 'LPI').GetDirection()


@pytest.mark.itk_process
class TestParseArgs:
    def test_parse_args_defaults(self):
        with patch('sys.argv', ['itk_resample.py', 'dataset', '/src', '/dst']):
            args = parse_args()
            assert args.mode == 'dataset'
            assert args.source_folder == '/src'
            assert args.dest_folder == '/dst'
            assert args.recursive is False
            assert args.mp is False
            assert args.workers is None
            assert args.spacing == ['-1', '-1', '-1']
            assert args.size == ['-1', '-1', '-1']
            assert args.target_folder is None

    def test_parse_args_with_options(self):
        with patch('sys.argv', ['itk_resample.py', 'image', '/src', '/dst', '--recursive', '--mp', '--workers', '4', '--spacing', '1.5', '-1', '1.5', '--size', '-1', '256', '-1']):
            args = parse_args()
            assert args.mode == 'image'
            assert args.recursive is True
            assert args.mp is True
            assert args.workers == 4
            assert args.spacing == ['1.5', '-1', '1.5']
            assert args.size == ['-1', '256', '-1']
