import os
import tempfile
import pytest
import numpy as np
import SimpleITK as sitk
from unittest.mock import patch, MagicMock

from itkit.process.itk_extract import ExtractProcessor, parse_label_mappings, main


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
        sitk.WriteImage(image, input_path)
        
        label_mapping = {1: 10, 2: 20}
        processor = ExtractProcessor(source_folder, dest_folder, label_mapping)
        
        result = processor.process_one(input_path)
        output_path = os.path.join(dest_folder, "test.mha")
        
        assert os.path.exists(output_path)
        output_image = sitk.ReadImage(output_path)
        output_array = sitk.GetArrayFromImage(output_image)
        
        # Check remapping: 1 -> 10, 2 -> 20, 0 stays 0
        assert np.all(output_array[output_array == 10] == 10)
        assert np.all(output_array[output_array == 20] == 20)
        assert np.all(output_array[output_array == 0] == 0)
        
        # Check metadata
        assert result is not None
        assert "test.mha" in result
        assert result["test.mha"]["original_labels"] == [1, 2]
        assert result["test.mha"]["extracted_labels"] == [10, 20]

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

    @patch('SimpleITK.ReadImage')
    def test_extract_one_sample_read_error(self, mock_read, temp_dir):
        mock_read.side_effect = Exception("Read error")
        processor = ExtractProcessor(temp_dir, temp_dir, {1: 0})
        result = processor._extract_one_sample("dummy", "dummy")
        assert result is None

@pytest.mark.itk_process
class TestMain:
    def test_main_success(self, temp_dir):
        source = os.path.join(temp_dir, 'source')
        dest = os.path.join(temp_dir, 'dest')
        os.makedirs(source)
        os.makedirs(dest)
        
        with patch('sys.argv', ['itk_extract.py', source, dest, '1:0', '2:1']), \
             patch('itkit.process.itk_extract.ExtractProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            with patch('os.makedirs'), patch('builtins.open', MagicMock()):
                main()
            
            mock_processor_class.assert_called_once_with(source, dest, {1: 0, 2: 1}, False, False, None)
            mock_processor.process.assert_called_once()
            mock_processor.save_meta.assert_called_once()

    def test_main_empty_mappings_error(self, temp_dir):
        source = os.path.join(temp_dir, 'source')
        dest = os.path.join(temp_dir, 'dest')
        
        with patch('sys.argv', ['itk_extract.py', source, dest]):
            with pytest.raises(SystemExit):
                main()

    def test_main_duplicate_target_error(self, temp_dir):
        source = os.path.join(temp_dir, 'source')
        dest = os.path.join(temp_dir, 'dest')
        
        with patch('sys.argv', ['itk_extract.py', source, dest, '1:0', '2:0']):
            with pytest.raises(SystemExit):
                main()
