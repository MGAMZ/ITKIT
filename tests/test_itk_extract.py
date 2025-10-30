import os
import tempfile
import pytest
import numpy as np
import SimpleITK as sitk

from itkit.process.itk_extract import ExtractProcessor, parse_label_mappings


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_sample_image(labels, shape=(10, 10, 10), dtype=np.uint8):
    """Create a sample SimpleITK image with given labels."""
    array = np.zeros(shape, dtype=dtype)
    for i, label in enumerate(labels):
        array[i % shape[0], :, :] = label
    image = sitk.GetImageFromArray(array)
    return image

@pytest.mark.itk_process
class TestParseLabelMappings:
    def test_valid_mappings(self):
        mappings = ["1:0", "2:1", "3:2"]
        result = parse_label_mappings(mappings)
        assert result == {1: 0, 2: 1, 3: 2}

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid mapping format"):
            parse_label_mappings(["1-0"])

@pytest.mark.itk_process
class TestExtractProcessor:
    def test_init(self, temp_dir):
        label_mapping = {1: 0, 2: 1}
        processor = ExtractProcessor(temp_dir, temp_dir, label_mapping)
        assert processor.source_folder == temp_dir
        assert processor.dest_folder == temp_dir
        assert processor.label_mapping == label_mapping

    def test_process_one(self, temp_dir):
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        # Create sample input image
        image = create_sample_image([0, 1, 2])
        input_path = os.path.join(source_folder, "test.mha")
        image_arr = sitk.GetArrayFromImage(image)
        sitk.WriteImage(image, input_path, True)
        
        label_mapping = {1: 10, 2: 20}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        output_path = os.path.join(dest_folder, "test.mha")
        
        assert os.path.exists(output_path)
        output_image = sitk.ReadImage(output_path)
        output_array = sitk.GetArrayFromImage(output_image)
        
        assert np.all(output_array[image_arr==1] == 10)
        assert np.all(output_array[image_arr==2] == 20)
        
        # Check remapping: 1 -> 10, 2 -> 20, 0 stays 0
        # Verify correct number of pixels for each label
        assert np.sum(output_array == 10) == 100  # 100 pixels were originally 1
        assert np.sum(output_array == 20) == 100  # 100 pixels were originally 2
        assert np.sum(output_array == 0) == 800   # 800 pixels were originally 0
        
        # Verify specific layers were remapped correctly
        assert np.all(output_array[1, :, :] == 10)  # Layer 1 (originally 1) -> 10
        assert np.all(output_array[2, :, :] == 20)  # Layer 2 (originally 2) -> 20
        assert np.all(output_array[0, :, :] == 0)   # Layer 0 (originally 0) stays 0
        assert np.all(output_array[3, :, :] == 0)   # Layer 3 (originally 0) stays 0
        
        # Check metadata
        assert result is not None
        assert result.name == "test.mha"
        assert result.include_classes == (10, 20)
        assert result.spacing == (1.0, 1.0, 1.0)
        assert result.size == (10, 10, 10)
        assert result.origin == (0.0, 0.0, 0.0)

    def test_extract_one_sample_skip_existing(self, temp_dir):
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        input_path = os.path.join(source_folder, "test.mha")
        output_path = os.path.join(dest_folder, "test.mha")
        
        # Create empty output file to simulate existing
        with open(output_path, 'w') as f:
            f.write("")
        
        processor = ExtractProcessor(source_folder, dest_folder, {1: 0})
        result = processor._extract_one_sample(input_path, output_path)
        assert result is None  # Should skip

    def test_extract_one_sample_dtype_conversion(self, temp_dir):
        """Test automatic dtype conversion for non-uint images"""
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        # Create image with int32 dtype (signed integer)
        array = np.zeros((5, 5, 5), dtype=np.int32)
        array[0, :, :] = 1
        array[1, :, :] = 2
        image = sitk.GetImageFromArray(array)
        
        input_path = os.path.join(source_folder, "test.mha")
        sitk.WriteImage(image, input_path, True)
        
        label_mapping = {1: 10, 2: 20}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        output_path = os.path.join(dest_folder, "test.mha")
        
        assert os.path.exists(output_path)
        output_image = sitk.ReadImage(output_path)
        output_array = sitk.GetArrayFromImage(output_image)
        
        # Check dtype was converted to uint8
        assert output_array.dtype == np.uint8
        # Check remapping worked
        assert np.all(output_array[output_array == 10] == 10)
        assert np.all(output_array[output_array == 20] == 20)

    def test_extract_one_sample_no_labels_found(self, temp_dir):
        """Test when mapped labels don't exist in the image"""
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        # Create image with labels 1, 2, 3
        image = create_sample_image([1, 2, 3])
        input_path = os.path.join(source_folder, "test.mha")
        sitk.WriteImage(image, input_path, True)
        
        # Map labels that don't exist in image (4, 5)
        label_mapping = {4: 10, 5: 20}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        output_path = os.path.join(dest_folder, "test.mha")
        
        assert os.path.exists(output_path)
        output_image = sitk.ReadImage(output_path)
        output_array = sitk.GetArrayFromImage(output_image)
        
        # All pixels should be 0 since no labels were mapped
        assert np.all(output_array == 0)
        # include_classes should be None since no labels were extracted
        assert result.include_classes is None

    def test_process_one_recursive_mode(self, temp_dir):
        """Test recursive mode path handling"""
        source_folder = os.path.join(temp_dir, "source")
        sub_folder = os.path.join(source_folder, "sub")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(sub_folder)
        os.makedirs(dest_folder)
        
        # Create image in subdirectory
        image = create_sample_image([0, 1])
        input_path = os.path.join(sub_folder, "test.mha")
        sitk.WriteImage(image, input_path, True)
        
        label_mapping = {1: 10}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping, recursive=True)
        
        result = processor.process_one(input_path)
        expected_output = os.path.join(dest_folder, "sub", "test.mha")
        
        assert os.path.exists(expected_output)
        assert result is not None
        assert result.name == "test.mha"

    def test_process_one_file_extension_conversion(self, temp_dir):
        """Test file extension normalization to .mha"""
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        # Test .nii extension
        image = create_sample_image([0, 1])
        input_path = os.path.join(source_folder, "test.nii")
        sitk.WriteImage(image, input_path, True)
        
        label_mapping = {1: 10}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        output_path = os.path.join(dest_folder, "test.mha")
        
        assert os.path.exists(output_path)
        assert result.name == "test.mha"

    def test_process_one_file_extension_conversion_niigz(self, temp_dir):
        """Test file extension normalization for .nii.gz files"""
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        # Test .nii.gz extension
        image = create_sample_image([0, 1])
        input_path = os.path.join(source_folder, "test.nii.gz")
        sitk.WriteImage(image, input_path, True)
        
        label_mapping = {1: 10}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        output_path = os.path.join(dest_folder, "test.mha")
        
        assert os.path.exists(output_path)
        assert result.name == "test.mha"

    def test_process_one_complex_filename(self, temp_dir):
        """Test processing file with complex name containing multiple dots"""
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        # Test filename with multiple dots
        image = create_sample_image([0, 1])
        input_path = os.path.join(source_folder, "test.file.name.nii")
        sitk.WriteImage(image, input_path, True)
        
        label_mapping = {1: 10}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        output_path = os.path.join(dest_folder, "test.file.name.mha")
        
        assert os.path.exists(output_path)
        assert result.name == "test.file.name.mha"

    def test_extract_one_sample_different_image_properties(self, temp_dir):
        """Test processing image with different spacing and origin"""
        source_folder = os.path.join(temp_dir, "source")
        dest_folder = os.path.join(temp_dir, "dest")
        os.makedirs(source_folder)
        os.makedirs(dest_folder)
        
        # Create image with custom spacing and origin
        array = np.zeros((5, 5, 5), dtype=np.uint8)
        array[0, :, :] = 1
        image = sitk.GetImageFromArray(array)
        image.SetSpacing([2.0, 2.0, 2.0])
        image.SetOrigin([10.0, 20.0, 30.0])
        
        input_path = os.path.join(source_folder, "test.mha")
        sitk.WriteImage(image, input_path, True)
        
        label_mapping = {1: 5}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        
        # Check that spacing and origin are preserved (note: SimpleITK XYZ -> ZYX conversion)
        assert result.spacing == (2.0, 2.0, 2.0)  # spacing is not reversed
        assert result.origin == (30.0, 20.0, 10.0)  # origin is reversed from XYZ to ZYX
        assert result.size == (5, 5, 5)
        assert result.include_classes == (5,)

@pytest.mark.itk_process
class TestParseArgs:
    def test_parse_args_basic(self):
        """Test basic argument parsing"""
        from itkit.process.itk_extract import parse_args
        import sys
        from unittest.mock import patch
        
        test_args = ['script.py', 'source', 'dest', '1:0', '2:1']
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
            assert args.source_folder == 'source'
            assert args.dest_folder == 'dest'
            assert args.mappings == ['1:0', '2:1']
            assert args.recursive is False
            assert args.mp is False
            assert args.workers is None

    def test_parse_args_with_options(self):
        """Test argument parsing with all options"""
        from itkit.process.itk_extract import parse_args
        import sys
        from unittest.mock import patch
        
        test_args = ['script.py', 'source', 'dest', '1:0', '-r', '--mp', '--workers', '4']
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
            assert args.recursive is True
            assert args.mp is True
            assert args.workers == 4

@pytest.mark.itk_process
class TestMainIntegration:
    def test_main_integration_success(self, temp_dir):
        """Integration test for main function with valid inputs"""
        from itkit.process.itk_extract import main
        import sys
        from unittest.mock import patch
        
        # Create source files
        source = os.path.join(temp_dir, 'source')
        dest = os.path.join(temp_dir, 'dest')
        os.makedirs(source)
        
        # Create a sample image
        image = create_sample_image([0, 1, 2])
        img_path = os.path.join(source, "sample.mha")
        sitk.WriteImage(image, img_path, True)
        
        test_args = ['script.py', source, dest, '1:10', '2:20']
        with patch.object(sys, 'argv', test_args):
            # This should run without errors
            main()
            
            # Check output was created
            output_path = os.path.join(dest, "sample.mha")
            assert os.path.exists(output_path)
            
            # Check config file was created
            config_path = os.path.join(dest, "extract_configs.json")
            assert os.path.exists(config_path)

    def test_main_config_save_error_handling(self, temp_dir):
        """Test main function handles config save errors gracefully"""
        from itkit.process.itk_extract import main
        import sys
        from unittest.mock import patch
        
        # Create source files
        source = os.path.join(temp_dir, 'source')
        dest = os.path.join(temp_dir, 'dest')
        os.makedirs(source)
        
        # Create a sample image
        image = create_sample_image([0, 1])
        img_path = os.path.join(source, "sample.mha")
        sitk.WriteImage(image, img_path, True)
        
        test_args = ['script.py', source, dest, '1:10']
        with patch.object(sys, 'argv', test_args):
            # This should complete successfully despite any config save issues
            main()
