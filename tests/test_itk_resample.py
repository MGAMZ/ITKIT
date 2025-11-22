import argparse
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from itkit.process import itk_resample
from itkit.process.metadata_models import MetadataManager


def make_args(**overrides):
    """Helper to construct argparse Namespace with defaults."""
    defaults = {
        'mode': 'image',
        'source_folder': '/src',
        'dest_folder': '/dst',
        'recursive': False,
        'mp': False,
        'workers': None,
        'spacing': ["-1", "-1", "-1"],
        'size': ["-1", "-1", "-1"],
        'target_folder': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

@pytest.mark.itk_process
class TestValidateAndPrepareArgs:
    """Grouped tests for itk_resample.validate_and_prepare_args.

    Grouping tests into a class keeps related tests together and avoids polluting
    module-level namespace while still letting pytest discover test methods.
    """

    def test_mutual_exclusive_target_and_spacing_size(self):
        args = make_args(target_folder="/tmp/tgt", spacing=["1.0", "-1", "-1"])
        with pytest.raises(ValueError, match="mutually exclusive"):
            itk_resample.validate_and_prepare_args(args)
        args = make_args(target_folder="/tmp/tgt", size=["64", "64", "-1"])
        with pytest.raises(ValueError, match="mutually exclusive"):
            itk_resample.validate_and_prepare_args(args)

    def test_spacing_size_length_validation(self):
        args = make_args(spacing=["1.0", "2.0"], size=["-1", "-1", "-1"])
        with pytest.raises(ValueError, match="--spacing must have 3 values"):
            itk_resample.validate_and_prepare_args(args)

    def test_per_dimension_exclusivity(self):
        args = make_args(spacing=["1.0", "-1", "-1"], size=["10", "-1", "-1"])
        with pytest.raises(ValueError, match="Cannot specify both spacing and size for dimension 0"):
            itk_resample.validate_and_prepare_args(args)

    def test_no_resampling_specified_prints_warning(self):
        args = make_args()
        res = itk_resample.validate_and_prepare_args(args)
        assert res == (None, None)

    def test_valid_spacing_and_size_parsing(self):
        args = make_args(spacing=["1.5", "-1", "1.5"], size=["-1", "256", "-1"], recursive=True, mp=True, workers=4)
        target_spacing, target_size = itk_resample.validate_and_prepare_args(args)
        assert isinstance(target_spacing, list) and isinstance(target_size, list)
        assert target_spacing == [1.5, -1.0, 1.5]
        assert target_size == [-1, 256, -1]

@pytest.mark.itk_process
class TestSingleResampleProcessor:
    """Test class for SingleResampleProcessor."""

    def test_full_io_processing_spacing_size(self, shared_temp_data, tmp_path):
        """Test full IO processing in SPACING_SIZE mode: process temp data folder and verify outputs."""
        dest_folder = tmp_path / "dst"
        dest_folder.mkdir()

        # Randomly generate target spacing
        target_spacing = [np.random.uniform(0.5, 3.0) for _ in range(3)]
        processor = itk_resample.SingleResampleProcessor(
            source_folder=str(shared_temp_data / "image"),
            dest_folder=str(dest_folder),
            target_spacing=target_spacing,
            target_size=[-1, -1, -1],
            field="image"
        )

        source_files = list((shared_temp_data / "image").glob("*.mha"))

        # Process all files
        processor.process()

        # Check that output files exist
        output_files = list(dest_folder.glob("*.mha"))
        assert len(output_files) == len(source_files)

        # Collect expected metadata and validate
        metadata_manager = MetadataManager()
        for output_file in output_files:
            img = sitk.ReadImage(str(output_file))
            
            # Generate expected metadata
            expected_metadata = itk_resample.SeriesMetadata.from_sitk_image(img, output_file.name)
            
            metadata_manager.update(expected_metadata)
            # Validate image properties
            expected_metadata.validate_itk_image(img)

        # Save and verify metadata JSON
        metadata_path = dest_folder / "metadata.json"
        metadata_manager.save(metadata_path)
        loaded_manager = MetadataManager(meta_file_path=metadata_path)
        for name, expected_meta in metadata_manager.meta.items():
            assert name in loaded_manager.meta
            assert loaded_manager.meta[name] == expected_meta

    def test_full_io_processing_target_image(self, shared_temp_data, tmp_path):
        """Test full IO processing in TARGET_IMAGE mode: use label as target for image."""
        dest_folder = tmp_path / "dst"
        dest_folder.mkdir()

        # Use label folder as target (same structure)
        processor = itk_resample.SingleResampleProcessor(
            source_folder=str(shared_temp_data / "image"),
            dest_folder=str(dest_folder),
            target_spacing=[-1, -1, -1],
            target_size=[-1, -1, -1],
            field="image",
            target_folder=str(shared_temp_data / "label")
        )

        source_files = list((shared_temp_data / "image").glob("*.mha"))

        # Process all files
        processor.process()

        # Check that output files exist
        output_files = list(dest_folder.glob("*.mha"))
        assert len(output_files) == len(source_files)

        # Collect expected metadata and validate
        metadata_manager = MetadataManager()
        for output_file in output_files:
            img = sitk.ReadImage(str(output_file))
            
            # Generate expected metadata
            target_file = Path(shared_temp_data / "label") / output_file.name
            target_img = sitk.ReadImage(str(target_file))
            expected_metadata = itk_resample.SeriesMetadata.from_sitk_image(target_img, output_file.name)
            
            metadata_manager.update(expected_metadata)
            # Validate image properties
            expected_metadata.validate_itk_image(img)

        # Save and verify metadata JSON
        metadata_path = dest_folder / "metadata.json"
        metadata_manager.save(metadata_path)
        loaded_manager = MetadataManager(meta_file_path=metadata_path)
        for name, expected_meta in metadata_manager.meta.items():
            assert name in loaded_manager.meta
            assert loaded_manager.meta[name] == expected_meta

    def test_full_io_processing_label_field(self, shared_temp_data, tmp_path):
        """Test full IO processing for label field: check unique classes preserved."""
        dest_folder = tmp_path / "dst"
        dest_folder.mkdir()

        # Randomly generate target size
        target_size = [np.random.randint(32, 128) for _ in range(3)]  # ZYX order
        processor = itk_resample.SingleResampleProcessor(
            source_folder=str(shared_temp_data / "label"),
            dest_folder=str(dest_folder),
            target_spacing=[-1, -1, -1],
            target_size=target_size,
            field="label"
        )

        source_files = list((shared_temp_data / "label").glob("*.mha"))

        # Process all files
        processor.process()

        # Check that output files exist
        output_files = list(dest_folder.glob("*.mha"))
        assert len(output_files) == len(source_files)

        # Collect expected metadata and validate
        metadata_manager = MetadataManager()
        for output_file in output_files:
            img = sitk.ReadImage(str(output_file))
            
            # Generate expected metadata
            expected_metadata = itk_resample.SeriesMetadata.from_sitk_image(img, output_file.name)
            
            metadata_manager.update(expected_metadata)
            # Validate image properties
            expected_metadata.validate_itk_image(img)

        # Save and verify metadata JSON
        metadata_path = dest_folder / "metadata.json"
        metadata_manager.save(metadata_path)
        loaded_manager = MetadataManager(meta_file_path=metadata_path)
        for name, expected_meta in metadata_manager.meta.items():
            assert name in loaded_manager.meta
            assert loaded_manager.meta[name] == expected_meta
