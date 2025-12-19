"""Tests for itk_convert module - ITKIT to MONAI and TorchIO format conversion."""

import csv
import json
import os
import tempfile
import importlib.util

import numpy as np
import pytest
import SimpleITK as sitk

from itkit.process import itk_convert, itk_convert_format, itk_convert_monai, itk_convert_torchio

# Check if MONAI is available for end-to-end compatibility tests
MONAI_AVAILABLE = importlib.util.find_spec("monai") is not None

if MONAI_AVAILABLE:
    import monai
    from monai.data import Dataset, load_decathlon_datalist
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

# Check if TorchIO is available for end-to-end compatibility tests
TORCHIO_AVAILABLE = importlib.util.find_spec("torchio") is not None

if TORCHIO_AVAILABLE:
    import torchio as tio

# Constants
SPACING_TOLERANCE = 0.001  # Tolerance for spacing comparison in tests


def create_test_mha_image(path: str, size: tuple, spacing: tuple, dtype=sitk.sitkInt16):
    """Helper to create test MHA images."""
    img = sitk.Image(size[::-1], dtype)  # SimpleITK uses XYZ
    img.SetSpacing(spacing[::-1])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)
    return img


def create_test_mha_label(path: str, size: tuple, spacing: tuple, num_classes: int = 3):
    """Helper to create test MHA label images with multiple classes."""
    arr = np.zeros(size, dtype=np.uint8)
    # Create some class regions
    for cls in range(1, num_classes):
        start = cls * 10
        end = start + 8
        arr[start:end, start:end, start:end] = cls

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing[::-1])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)
    return img


def setup_itkit_dataset(tmpdir, num_samples: int = 3, num_classes: int = 3):
    """Helper to setup ITKIT-style dataset with image and label directories."""
    img_dir = os.path.join(tmpdir, 'image')
    lbl_dir = os.path.join(tmpdir, 'label')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    samples = []
    for i in range(num_samples):
        size = (64, 64, 64)
        spacing = (1.0, 0.5, 0.5)
        name = f"case_{i:04d}"

        img_path = os.path.join(img_dir, f"{name}.mha")
        lbl_path = os.path.join(lbl_dir, f"{name}.mha")

        create_test_mha_image(img_path, size, spacing)
        create_test_mha_label(lbl_path, size, spacing, num_classes)

        samples.append(name)

    return samples


@pytest.mark.itk_process
class TestMonaiDatasetJson:
    """Test MonaiDatasetJson helper class."""

    def test_init_default(self):
        """Test default initialization."""
        ds = itk_convert_monai.MonaiDatasetJson()
        assert ds.name == "ITKITDataset"
        assert ds.modality == {"0": "CT"}
        assert ds.labels == {"0": "background"}
        assert len(ds.training) == 0
        assert len(ds.test) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        ds = itk_convert_monai.MonaiDatasetJson(
            name="TestDataset",
            description="Test description",
            modality={"0": "MR"},
            labels={"0": "background", "1": "tumor"},
        )
        assert ds.name == "TestDataset"
        assert ds.modality == {"0": "MR"}
        assert ds.labels == {"0": "background", "1": "tumor"}

    def test_add_training_sample(self):
        """Test adding training samples."""
        ds = itk_convert_monai.MonaiDatasetJson()
        ds.add_training_sample("./imagesTr/case_001.nii.gz", "./labelsTr/case_001.nii.gz")
        ds.add_training_sample("./imagesTr/case_002.nii.gz", "./labelsTr/case_002.nii.gz")

        assert len(ds.training) == 2
        assert ds.training[0]["image"] == "./imagesTr/case_001.nii.gz"
        assert ds.training[0]["label"] == "./labelsTr/case_001.nii.gz"

    def test_add_test_sample(self):
        """Test adding test samples."""
        ds = itk_convert_monai.MonaiDatasetJson()
        ds.add_test_sample("./imagesTs/case_001.nii.gz")

        assert len(ds.test) == 1
        assert ds.test[0]["image"] == "./imagesTs/case_001.nii.gz"
        assert "label" not in ds.test[0]

    def test_update_labels_from_classes(self):
        """Test updating labels from discovered classes."""
        ds = itk_convert_monai.MonaiDatasetJson()
        ds.update_labels_from_classes({0, 1, 2, 5})

        assert ds.labels["0"] == "background"
        assert ds.labels["1"] == "class_1"
        assert ds.labels["2"] == "class_2"
        assert ds.labels["5"] == "class_5"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ds = itk_convert_monai.MonaiDatasetJson(name="TestDataset")
        ds.add_training_sample("./imagesTr/case_001.nii.gz", "./labelsTr/case_001.nii.gz")

        data = ds.to_dict()

        assert data["name"] == "TestDataset"
        assert data["numTraining"] == 1
        assert data["numTest"] == 0
        assert len(data["training"]) == 1

    def test_save(self):
        """Test saving to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = itk_convert_monai.MonaiDatasetJson(name="TestDataset")
            ds.add_training_sample("./imagesTr/case_001.nii.gz", "./labelsTr/case_001.nii.gz")

            json_path = os.path.join(tmpdir, "dataset.json")
            ds.save(json_path)

            assert os.path.exists(json_path)

            with open(json_path) as f:
                loaded = json.load(f)

            assert loaded["name"] == "TestDataset"
            assert loaded["numTraining"] == 1


@pytest.mark.itk_process
class TestMonaiConverter:
    """Test MonaiConverter class."""

    def test_validate_source_structure_valid(self):
        """Test validation with valid source structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_itkit_dataset(tmpdir)

            converter = itk_convert_monai.MonaiConverter(
                source_folder=tmpdir,
                dest_folder=os.path.join(tmpdir, "output"),
            )
            # Should not raise
            assert converter.source_folder == tmpdir

    def test_validate_source_structure_missing_image(self):
        """Test validation with missing image folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "label"))

            with pytest.raises(ValueError, match="Missing 'image' subfolder"):
                itk_convert_monai.MonaiConverter(
                    source_folder=tmpdir,
                    dest_folder=os.path.join(tmpdir, "output"),
                )

    def test_validate_source_structure_missing_label(self):
        """Test validation with missing label folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "image"))

            with pytest.raises(ValueError, match="Missing 'label' subfolder"):
                itk_convert_monai.MonaiConverter(
                    source_folder=tmpdir,
                    dest_folder=os.path.join(tmpdir, "output"),
                )

    def test_get_items_to_process(self):
        """Test getting items to process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = setup_itkit_dataset(tmpdir, num_samples=3)

            converter = itk_convert_monai.MonaiConverter(
                source_folder=tmpdir,
                dest_folder=os.path.join(tmpdir, "output"),
            )

            items = converter.get_items_to_process()
            assert len(items) == 3

            # Check structure of first item
            img_in, img_out, lbl_in, lbl_out = items[0]
            assert img_in.endswith(".mha")
            assert img_out.endswith(".nii.gz")
            assert lbl_in.endswith(".mha")
            assert lbl_out.endswith(".nii.gz")
            assert "imagesTr" in img_out
            assert "labelsTr" in lbl_out

    def test_get_items_to_process_test_split(self):
        """Test getting items for test split (no labels)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_itkit_dataset(tmpdir, num_samples=2)

            converter = itk_convert_monai.MonaiConverter(
                source_folder=tmpdir,
                dest_folder=os.path.join(tmpdir, "output"),
                split="test",
            )

            items = converter.get_items_to_process()
            assert len(items) == 2

            img_in, img_out, lbl_in, lbl_out = items[0]
            assert "imagesTs" in img_out
            assert lbl_out == ""  # No label output for test split

    def test_full_conversion(self):
        """Test full conversion process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            samples = setup_itkit_dataset(src_dir, num_samples=3)

            converter = itk_convert_monai.convert_to_monai(
                source_folder=src_dir,
                dest_folder=dest_dir,
                dataset_name="TestDataset",
                description="Test conversion",
                modality={"0": "CT"},
            )

            # Check output structure
            assert os.path.exists(os.path.join(dest_dir, "imagesTr"))
            assert os.path.exists(os.path.join(dest_dir, "labelsTr"))
            assert os.path.exists(os.path.join(dest_dir, "dataset.json"))

            # Check converted files
            img_files = os.listdir(os.path.join(dest_dir, "imagesTr"))
            lbl_files = os.listdir(os.path.join(dest_dir, "labelsTr"))
            assert len(img_files) == 3
            assert len(lbl_files) == 3
            assert all(f.endswith(".nii.gz") for f in img_files)

            # Check dataset.json contents
            with open(os.path.join(dest_dir, "dataset.json")) as f:
                ds_json = json.load(f)

            assert ds_json["name"] == "TestDataset"
            assert ds_json["numTraining"] == 3
            assert len(ds_json["training"]) == 3

    def test_conversion_with_custom_labels(self):
        """Test conversion with custom label names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2, num_classes=3)

            labels = {"0": "background", "1": "liver", "2": "tumor"}
            converter = itk_convert_monai.convert_to_monai(
                source_folder=src_dir,
                dest_folder=dest_dir,
                labels=labels,
            )

            with open(os.path.join(dest_dir, "dataset.json")) as f:
                ds_json = json.load(f)

            assert ds_json["labels"] == labels

    def test_conversion_val_split(self):
        """Test conversion with validation split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            itk_convert_monai.convert_to_monai(
                source_folder=src_dir,
                dest_folder=dest_dir,
                split="val",
            )

            # Check output structure for validation split
            assert os.path.exists(os.path.join(dest_dir, "imagesVal"))
            assert os.path.exists(os.path.join(dest_dir, "labelsVal"))

            with open(os.path.join(dest_dir, "dataset.json")) as f:
                ds_json = json.load(f)

            assert ds_json["numValidation"] == 2
            assert "validation" in ds_json

    def test_conversion_test_split(self):
        """Test conversion with test split (no labels)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            itk_convert_monai.convert_to_monai(
                source_folder=src_dir,
                dest_folder=dest_dir,
                split="test",
            )

            # Check output structure for test split
            assert os.path.exists(os.path.join(dest_dir, "imagesTs"))
            # No labels for test split
            assert not os.path.exists(os.path.join(dest_dir, "labelsTs"))

            with open(os.path.join(dest_dir, "dataset.json")) as f:
                ds_json = json.load(f)

            assert ds_json["numTest"] == 2
            assert "test" in ds_json


@pytest.mark.itk_process
class TestConvertSingleFile:
    """Test the single file conversion function."""

    def test_convert_mha_to_nifti(self):
        """Test converting a single mha file to nii.gz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input mha
            input_path = os.path.join(tmpdir, "test.mha")
            output_path = os.path.join(tmpdir, "output", "test.nii.gz")

            create_test_mha_image(input_path, (32, 32, 32), (1.0, 1.0, 1.0))

            result = itk_convert_monai._convert_single_file((input_path, output_path))

            assert result is not None
            assert result["success"] is True
            assert os.path.exists(output_path)

            # Verify the output file
            output_img = sitk.ReadImage(output_path)
            assert output_img.GetSize() == (32, 32, 32)

    def test_convert_label_tracks_classes(self):
        """Test that label conversion tracks unique classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "label.mha")
            output_path = os.path.join(tmpdir, "output", "label.nii.gz")

            create_test_mha_label(input_path, (64, 64, 64), (1.0, 1.0, 1.0), num_classes=4)

            result = itk_convert_monai._convert_single_file((input_path, output_path))

            assert result is not None
            assert result["success"] is True
            assert result["unique_classes"] is not None
            assert 0 in result["unique_classes"]

    def test_convert_nonexistent_file(self):
        """Test converting a non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "nonexistent.mha")
            output_path = os.path.join(tmpdir, "output.nii.gz")

            result = itk_convert_monai._convert_single_file((input_path, output_path))

            assert result is not None
            assert result["success"] is False
            assert "error" in result


@pytest.mark.itk_process
class TestParseArgs:
    """Test CLI argument parsing."""

    def test_monai_subcommand_basic(self, monkeypatch):
        """Test basic monai subcommand parsing."""
        test_args = ["itk_convert", "monai", "/source", "/dest"]
        monkeypatch.setattr("sys.argv", test_args)

        args = itk_convert.parse_args()

        assert args.format == "monai"
        assert args.source_folder == "/source"
        assert args.dest_folder == "/dest"
        assert args.name == "ITKITDataset"
        assert args.split == "train"

    def test_monai_subcommand_full(self, monkeypatch):
        """Test monai subcommand with all options."""
        test_args = [
            "itk_convert", "monai", "/source", "/dest",
            "--name", "MyDataset",
            "--description", "My description",
            "--modality", "MR",
            "--split", "val",
            "--labels", "background", "liver", "tumor",
            "--mp",
            "--workers", "4",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        args = itk_convert.parse_args()

        assert args.format == "monai"
        assert args.name == "MyDataset"
        assert args.description == "My description"
        assert args.modality == "MR"
        assert args.split == "val"
        assert args.labels == ["background", "liver", "tumor"]
        assert args.mp is True
        assert args.workers == 4


@pytest.mark.itk_process
class TestMainFunction:
    """Test main CLI entry point."""

    def test_main_no_format(self, monkeypatch, capsys):
        """Test main with no format specified."""
        test_args = ["itk_convert"]
        monkeypatch.setattr("sys.argv", test_args)

        result = itk_convert.main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Please specify a format" in captured.out

    def test_main_monai_success(self, monkeypatch):
        """Test main with successful monai conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            test_args = ["itk_convert", "monai", src_dir, dest_dir]
            monkeypatch.setattr("sys.argv", test_args)

            result = itk_convert.main()

            assert result == 0
            assert os.path.exists(os.path.join(dest_dir, "dataset.json"))

    def test_main_monai_invalid_source(self, monkeypatch, capsys):
        """Test main with invalid source folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "nonexistent")
            dest_dir = os.path.join(tmpdir, "output")

            test_args = ["itk_convert", "monai", src_dir, dest_dir]
            monkeypatch.setattr("sys.argv", test_args)

            result = itk_convert.main()

            assert result == 1
            captured = capsys.readouterr()
            assert "Error during conversion" in captured.out

@pytest.mark.itk_process
@pytest.mark.skipif(not MONAI_AVAILABLE, reason="monai library is not installed")
class TestMonaiEndToEndCompatibility:
    """End-to-end test: Verify if converted folder can be correctly recognized by MONAI."""

    def test_monai_can_load_converted_data(self):
        """Test if MONAI Dataset can recognize and load converted NIfTI files and JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # 1. Setup original ITKIT format data
            num_samples = 3
            setup_itkit_dataset(src_dir, num_samples=num_samples, num_classes=3)

            # 2. Execute conversion
            itk_convert_monai.convert_to_monai(
                source_folder=src_dir,
                dest_folder=dest_dir,
                dataset_name="TestCompatibility",
                modality={"0": "CT"}
            )

            json_path = os.path.join(dest_dir, "dataset.json")
            assert os.path.exists(json_path)

            # 3. Load dataset.json using MONAI utilities
            training_data = load_decathlon_datalist(
                json_path,
                data_list_key="training",
                base_dir=dest_dir
            )

            assert len(training_data) == num_samples
            assert "image" in training_data[0]
            assert "label" in training_data[0]

            # 4. Build MONAI pipeline to verify readability
            check_ds = Dataset(
                data=training_data,
                transform=Compose([
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"])
                ])
            )

            # Try to load the first sample
            sample = check_ds[0]

            # Verify tensor dimensions and types
            assert isinstance(sample["image"], monai.data.MetaTensor)
            assert sample["image"].shape[1:] == (64, 64, 64)
            assert sample["label"].shape[1:] == (64, 64, 64)

            # Verify metadata preservation
            assert "affine" in sample["image"].meta


@pytest.mark.itk_process
class TestTorchIOConverter:
    """Test TorchIOConverter class."""

    def test_validate_source_structure_valid(self):
        """Test validation with valid source structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_itkit_dataset(tmpdir)

            converter = itk_convert_torchio.TorchIOConverter(
                source_folder=tmpdir,
                dest_folder=os.path.join(tmpdir, "output"),
            )
            # Should not raise
            assert converter.source_folder == tmpdir

    def test_validate_source_structure_missing_image(self):
        """Test validation with missing image folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "label"))

            with pytest.raises(ValueError, match="Missing 'image' subfolder"):
                itk_convert_torchio.TorchIOConverter(
                    source_folder=tmpdir,
                    dest_folder=os.path.join(tmpdir, "output"),
                )

    def test_validate_source_structure_missing_label(self):
        """Test validation with missing label folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "image"))

            with pytest.raises(ValueError, match="Missing 'label' subfolder"):
                itk_convert_torchio.TorchIOConverter(
                    source_folder=tmpdir,
                    dest_folder=os.path.join(tmpdir, "output"),
                )

    def test_get_items_to_process(self):
        """Test getting items to process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = setup_itkit_dataset(tmpdir, num_samples=3)

            converter = itk_convert_torchio.TorchIOConverter(
                source_folder=tmpdir,
                dest_folder=os.path.join(tmpdir, "output"),
            )

            items = converter.get_items_to_process()
            assert len(items) == 3

            # Check structure of first item
            img_in, img_out, lbl_in, lbl_out = items[0]
            assert img_in.endswith(".mha")
            assert img_out.endswith(".nii.gz")
            assert lbl_in.endswith(".mha")
            assert lbl_out.endswith(".nii.gz")
            assert "images" in img_out
            assert "labels" in lbl_out

    def test_full_conversion(self):
        """Test full conversion process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            samples = setup_itkit_dataset(src_dir, num_samples=3)

            converter = itk_convert_torchio.convert_to_torchio(
                source_folder=src_dir,
                dest_folder=dest_dir,
                create_csv=True,
            )

            # Check output structure
            assert os.path.exists(os.path.join(dest_dir, "images"))
            assert os.path.exists(os.path.join(dest_dir, "labels"))
            assert os.path.exists(os.path.join(dest_dir, "subjects.csv"))

            # Check converted files
            img_files = os.listdir(os.path.join(dest_dir, "images"))
            lbl_files = os.listdir(os.path.join(dest_dir, "labels"))
            assert len(img_files) == 3
            assert len(lbl_files) == 3
            assert all(f.endswith(".nii.gz") for f in img_files)

            # Check subjects.csv contents
            with open(os.path.join(dest_dir, "subjects.csv")) as f:
                reader = csv.DictReader(f)
                subjects = list(reader)

            assert len(subjects) == 3
            assert all("subject" in s for s in subjects)
            assert all("image" in s for s in subjects)
            assert all("label" in s for s in subjects)

    def test_conversion_without_csv(self):
        """Test conversion without creating CSV manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            itk_convert_torchio.convert_to_torchio(
                source_folder=src_dir,
                dest_folder=dest_dir,
                create_csv=False,
            )

            # Check that CSV was not created
            assert not os.path.exists(os.path.join(dest_dir, "subjects.csv"))

            # But images and labels should exist
            assert os.path.exists(os.path.join(dest_dir, "images"))
            assert os.path.exists(os.path.join(dest_dir, "labels"))

    def test_conversion_preserves_metadata(self):
        """Test that conversion preserves image metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # Create dataset with specific spacing
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            size = (32, 32, 32)
            spacing = (2.0, 1.5, 1.5)

            img_path = os.path.join(img_dir, "test_case.mha")
            lbl_path = os.path.join(lbl_dir, "test_case.mha")

            create_test_mha_image(img_path, size, spacing)
            create_test_mha_label(lbl_path, size, spacing)

            # Convert
            itk_convert_torchio.convert_to_torchio(
                source_folder=src_dir,
                dest_folder=dest_dir,
            )

            # Read output and verify spacing
            output_img = sitk.ReadImage(os.path.join(dest_dir, "images", "test_case.nii.gz"))
            output_spacing = output_img.GetSpacing()

            # SimpleITK uses XYZ order
            assert abs(output_spacing[0] - spacing[2]) < SPACING_TOLERANCE
            assert abs(output_spacing[1] - spacing[1]) < SPACING_TOLERANCE
            assert abs(output_spacing[2] - spacing[0]) < SPACING_TOLERANCE


@pytest.mark.itk_process
class TestConvertSingleFileTorchIO:
    """Test the single file conversion function for TorchIO."""

    def test_convert_mha_to_nifti(self):
        """Test converting a single mha file to nii.gz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input mha
            input_path = os.path.join(tmpdir, "test.mha")
            output_path = os.path.join(tmpdir, "output", "test.nii.gz")

            create_test_mha_image(input_path, (32, 32, 32), (1.0, 1.0, 1.0))

            result = itk_convert_torchio._convert_single_file((input_path, output_path))

            assert result is not None
            assert result["success"] is True
            assert os.path.exists(output_path)

            # Verify the output file
            output_img = sitk.ReadImage(output_path)
            assert output_img.GetSize() == (32, 32, 32)

    def test_convert_nonexistent_file(self):
        """Test converting a non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "nonexistent.mha")
            output_path = os.path.join(tmpdir, "output.nii.gz")

            result = itk_convert_torchio._convert_single_file((input_path, output_path))

            assert result is not None
            assert result["success"] is False
            assert "error" in result


@pytest.mark.itk_process
class TestTorchIOCLI:
    """Test TorchIO CLI integration."""

    def test_torchio_subcommand_basic(self, monkeypatch):
        """Test basic torchio subcommand parsing."""
        test_args = ["itk_convert", "torchio", "/source", "/dest"]
        monkeypatch.setattr("sys.argv", test_args)

        args = itk_convert.parse_args()

        assert args.format == "torchio"
        assert args.source_folder == "/source"
        assert args.dest_folder == "/dest"
        assert args.no_csv is False

    def test_torchio_subcommand_full(self, monkeypatch):
        """Test torchio subcommand with all options."""
        test_args = [
            "itk_convert", "torchio", "/source", "/dest",
            "--no-csv",
            "--mp",
            "--workers", "4",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        args = itk_convert.parse_args()

        assert args.format == "torchio"
        assert args.no_csv is True
        assert args.mp is True
        assert args.workers == 4

    def test_main_torchio_success(self, monkeypatch):
        """Test main with successful torchio conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            test_args = ["itk_convert", "torchio", src_dir, dest_dir]
            monkeypatch.setattr("sys.argv", test_args)

            result = itk_convert.main()

            assert result == 0
            assert os.path.exists(os.path.join(dest_dir, "subjects.csv"))

    def test_main_torchio_invalid_source(self, monkeypatch, capsys):
        """Test main with invalid source folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "nonexistent")
            dest_dir = os.path.join(tmpdir, "output")

            test_args = ["itk_convert", "torchio", src_dir, dest_dir]
            monkeypatch.setattr("sys.argv", test_args)

            result = itk_convert.main()

            assert result == 1
            captured = capsys.readouterr()
            assert "Error during conversion" in captured.out


@pytest.mark.itk_process
@pytest.mark.skipif(not TORCHIO_AVAILABLE, reason="torchio library is not installed")
class TestTorchIOEndToEndCompatibility:
    """End-to-end test: Verify if converted folder can be correctly recognized by TorchIO."""

    def test_torchio_can_load_converted_data(self):
        """Test if TorchIO SubjectsDataset can load converted NIfTI files and CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # 1. Setup original ITKIT format data
            num_samples = 3
            setup_itkit_dataset(src_dir, num_samples=num_samples, num_classes=3)

            # 2. Execute conversion
            itk_convert_torchio.convert_to_torchio(
                source_folder=src_dir,
                dest_folder=dest_dir,
                create_csv=True,
            )

            csv_path = os.path.join(dest_dir, "subjects.csv")
            assert os.path.exists(csv_path)

            # 3. Load subjects.csv
            subjects = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subject = tio.Subject(
                        image=tio.ScalarImage(os.path.join(dest_dir, row["image"])),
                        label=tio.LabelMap(os.path.join(dest_dir, row["label"])),
                        subject_id=row["subject"],
                    )
                    subjects.append(subject)

            assert len(subjects) == num_samples

            # 4. Build TorchIO dataset
            dataset = tio.SubjectsDataset(subjects)
            assert len(dataset) == num_samples

            # 5. Try to load the first sample
            sample = dataset[0]

            # Verify we can access image and label
            assert "image" in sample
            assert "label" in sample

            # Verify tensor dimensions
            image_data = sample["image"].data
            label_data = sample["label"].data

            # TorchIO adds channel dimension
            assert image_data.shape[0] == 1  # Channel dimension
            assert label_data.shape[0] == 1  # Channel dimension
            assert image_data.shape[1:] == (64, 64, 64)
            assert label_data.shape[1:] == (64, 64, 64)

    def test_torchio_can_apply_transforms(self):
        """Test that TorchIO can apply transforms to converted data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # Setup and convert
            setup_itkit_dataset(src_dir, num_samples=2, num_classes=3)
            itk_convert_torchio.convert_to_torchio(
                source_folder=src_dir,
                dest_folder=dest_dir,
                create_csv=True,
            )

            # Load subjects
            csv_path = os.path.join(dest_dir, "subjects.csv")
            subjects = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subject = tio.Subject(
                        image=tio.ScalarImage(os.path.join(dest_dir, row["image"])),
                        label=tio.LabelMap(os.path.join(dest_dir, row["label"])),
                    )
                    subjects.append(subject)

            # Create dataset with transforms
            transform = tio.Compose([
                tio.RandomFlip(axes=(0,)),
                tio.RandomAffine(),
            ])

            dataset = tio.SubjectsDataset(subjects, transform=transform)

            # Try to get a transformed sample
            sample = dataset[0]

            # Should work without errors
            assert "image" in sample
            assert "label" in sample
            assert sample["image"].data.shape[1:] == (64, 64, 64)


@pytest.mark.itk_process
class TestFormatConverter:
    """Test FormatConverter class for medical image format conversion."""

    def test_validate_source_structure_valid(self):
        """Test validation with valid source structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_itkit_dataset(tmpdir)

            from itkit.process.itk_convert_format import FormatConverter

            converter = FormatConverter(
                source_folder=tmpdir,
                dest_folder=os.path.join(tmpdir, "output"),
                target_format="nii.gz",
            )
            # Should not raise
            assert converter.source_folder == tmpdir

    def test_validate_source_structure_missing_image(self):
        """Test validation with missing image folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "label"))

            from itkit.process.itk_convert_format import FormatConverter

            with pytest.raises(ValueError, match="Missing 'image' subfolder"):
                FormatConverter(
                    source_folder=tmpdir,
                    dest_folder=os.path.join(tmpdir, "output"),
                    target_format="nii.gz",
                )

    def test_validate_source_structure_missing_label(self):
        """Test validation with missing label folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "image"))

            from itkit.process.itk_convert_format import FormatConverter

            with pytest.raises(ValueError, match="Missing 'label' subfolder"):
                FormatConverter(
                    source_folder=tmpdir,
                    dest_folder=os.path.join(tmpdir, "output"),
                    target_format="nii.gz",
                )

    def test_unsupported_format(self):
        """Test with unsupported format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_itkit_dataset(tmpdir)

            from itkit.process.itk_convert_format import FormatConverter

            with pytest.raises(ValueError, match="Unsupported target format"):
                FormatConverter(
                    source_folder=tmpdir,
                    dest_folder=os.path.join(tmpdir, "output"),
                    target_format="dcm",
                )

    def test_get_items_to_process(self):
        """Test getting items to process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = setup_itkit_dataset(tmpdir, num_samples=3)

            from itkit.process.itk_convert_format import FormatConverter

            converter = FormatConverter(
                source_folder=tmpdir,
                dest_folder=os.path.join(tmpdir, "output"),
                target_format="nii.gz",
            )

            items = converter.get_items_to_process()
            assert len(items) == 3

            # Check structure of first item
            img_in, img_out, lbl_in, lbl_out = items[0]
            assert img_in.endswith(".mha")
            assert img_out.endswith(".nii.gz")
            assert lbl_in.endswith(".mha")
            assert lbl_out.endswith(".nii.gz")
            assert "image" in img_out
            assert "label" in lbl_out

    def test_conversion_mha_to_nifti(self):
        """Test full conversion from MHA to NIfTI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            samples = setup_itkit_dataset(src_dir, num_samples=3)

            from itkit.process.itk_convert_format import convert_format

            converter = convert_format(
                source_folder=src_dir,
                dest_folder=dest_dir,
                target_format="nii.gz",
            )

            # Check output structure
            assert os.path.exists(os.path.join(dest_dir, "image"))
            assert os.path.exists(os.path.join(dest_dir, "label"))

            # Check converted files
            img_files = os.listdir(os.path.join(dest_dir, "image"))
            lbl_files = os.listdir(os.path.join(dest_dir, "label"))
            assert len(img_files) == 3
            assert len(lbl_files) == 3
            assert all(f.endswith(".nii.gz") for f in img_files)

    def test_conversion_mha_to_nrrd(self):
        """Test full conversion from MHA to NRRD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            from itkit.process.itk_convert_format import convert_format

            convert_format(
                source_folder=src_dir,
                dest_folder=dest_dir,
                target_format="nrrd",
            )

            # Check converted files
            img_files = os.listdir(os.path.join(dest_dir, "image"))
            lbl_files = os.listdir(os.path.join(dest_dir, "label"))
            assert len(img_files) == 2
            assert len(lbl_files) == 2
            assert all(f.endswith(".nrrd") for f in img_files)

    def test_conversion_mha_to_mhd(self):
        """Test full conversion from MHA to MHD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            from itkit.process.itk_convert_format import convert_format

            convert_format(
                source_folder=src_dir,
                dest_folder=dest_dir,
                target_format="mhd",
            )

            # Check converted files
            img_files = os.listdir(os.path.join(dest_dir, "image"))
            assert any(f.endswith(".mhd") for f in img_files)
            # MHD format also creates .raw files
            assert any(f.endswith(".raw") for f in img_files)

    def test_metadata_preservation_spacing(self):
        """Test that conversion preserves image spacing metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # Create dataset with specific spacing
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            size = (32, 32, 32)
            spacing = (2.5, 1.5, 1.0)

            img_path = os.path.join(img_dir, "test_case.mha")
            lbl_path = os.path.join(lbl_dir, "test_case.mha")

            original_img = create_test_mha_image(img_path, size, spacing)
            create_test_mha_label(lbl_path, size, spacing)

            # Convert to NIfTI
            from itkit.process.itk_convert_format import convert_format

            convert_format(
                source_folder=src_dir,
                dest_folder=dest_dir,
                target_format="nii.gz",
            )

            # Read output and verify spacing
            output_img = sitk.ReadImage(os.path.join(dest_dir, "image", "test_case.nii.gz"))
            output_spacing = output_img.GetSpacing()

            # SimpleITK uses XYZ order
            original_spacing = original_img.GetSpacing()
            for i in range(3):
                assert abs(output_spacing[i] - original_spacing[i]) < SPACING_TOLERANCE

    def test_metadata_preservation_origin(self):
        """Test that conversion preserves image origin metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # Create dataset with specific origin
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            size = (32, 32, 32)
            spacing = (1.0, 1.0, 1.0)
            origin = (10.0, 20.0, 30.0)

            img_path = os.path.join(img_dir, "test_case.mha")
            lbl_path = os.path.join(lbl_dir, "test_case.mha")

            img = sitk.Image(size[::-1], sitk.sitkInt16)
            img.SetSpacing(spacing[::-1])
            img.SetOrigin(origin[::-1])
            sitk.WriteImage(img, img_path)

            lbl = sitk.Image(size[::-1], sitk.sitkUInt8)
            lbl.SetSpacing(spacing[::-1])
            lbl.SetOrigin(origin[::-1])
            sitk.WriteImage(lbl, lbl_path)

            # Convert to NRRD
            from itkit.process.itk_convert_format import convert_format

            convert_format(
                source_folder=src_dir,
                dest_folder=dest_dir,
                target_format="nrrd",
            )

            # Read output and verify origin
            output_img = sitk.ReadImage(os.path.join(dest_dir, "image", "test_case.nrrd"))
            output_origin = output_img.GetOrigin()

            for i in range(3):
                assert abs(output_origin[i] - origin[::-1][i]) < SPACING_TOLERANCE

    def test_metadata_preservation_direction(self):
        """Test that conversion preserves image direction matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # Create dataset with specific direction
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            size = (32, 32, 32)
            spacing = (1.0, 1.0, 1.0)

            img_path = os.path.join(img_dir, "test_case.mha")
            lbl_path = os.path.join(lbl_dir, "test_case.mha")

            img = sitk.Image(size[::-1], sitk.sitkInt16)
            img.SetSpacing(spacing[::-1])
            # Set a non-identity direction matrix
            img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
            sitk.WriteImage(img, img_path)

            lbl = sitk.Image(size[::-1], sitk.sitkUInt8)
            lbl.SetSpacing(spacing[::-1])
            lbl.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
            sitk.WriteImage(lbl, lbl_path)

            # Convert to NIfTI
            from itkit.process.itk_convert_format import convert_format

            convert_format(
                source_folder=src_dir,
                dest_folder=dest_dir,
                target_format="nii.gz",
            )

            # Read output and verify direction
            output_img = sitk.ReadImage(os.path.join(dest_dir, "image", "test_case.nii.gz"))
            output_direction = output_img.GetDirection()

            assert len(output_direction) == 9
            # Check diagonal elements
            assert output_direction[0] == 1
            assert output_direction[4] == 1
            assert output_direction[8] == 1

    def test_data_integrity(self):
        """Test that conversion preserves actual image data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            # Create dataset with known data
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            size = (16, 16, 16)
            spacing = (1.0, 1.0, 1.0)

            # Create image with specific pattern
            arr = np.arange(16 * 16 * 16).reshape(size).astype(np.int16)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(spacing[::-1])

            img_path = os.path.join(img_dir, "test_case.mha")
            sitk.WriteImage(img, img_path)

            # Create label with specific pattern
            lbl_arr = np.zeros(size, dtype=np.uint8)
            lbl_arr[5:10, 5:10, 5:10] = 1
            lbl = sitk.GetImageFromArray(lbl_arr)
            lbl.SetSpacing(spacing[::-1])

            lbl_path = os.path.join(lbl_dir, "test_case.mha")
            sitk.WriteImage(lbl, lbl_path)

            # Convert to NIfTI
            from itkit.process.itk_convert_format import convert_format

            convert_format(
                source_folder=src_dir,
                dest_folder=dest_dir,
                target_format="nii.gz",
            )

            # Read output and verify data
            output_img = sitk.ReadImage(os.path.join(dest_dir, "image", "test_case.nii.gz"))
            output_arr = sitk.GetArrayFromImage(output_img)

            # Check that data matches
            np.testing.assert_array_equal(output_arr, arr)

            # Check label data
            output_lbl = sitk.ReadImage(os.path.join(dest_dir, "label", "test_case.nii.gz"))
            output_lbl_arr = sitk.GetArrayFromImage(output_lbl)
            np.testing.assert_array_equal(output_lbl_arr, lbl_arr)

    def test_multiple_formats_roundtrip(self):
        """Test converting through multiple formats maintains data integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original MHA dataset
            mha_dir = os.path.join(tmpdir, "mha")
            nifti_dir = os.path.join(tmpdir, "nifti")
            nrrd_dir = os.path.join(tmpdir, "nrrd")
            mha2_dir = os.path.join(tmpdir, "mha2")

            setup_itkit_dataset(mha_dir, num_samples=2)

            from itkit.process.itk_convert_format import convert_format

            # Convert MHA -> NIfTI -> NRRD -> MHA
            convert_format(mha_dir, nifti_dir, "nii.gz")
            convert_format(nifti_dir, nrrd_dir, "nrrd")
            convert_format(nrrd_dir, mha2_dir, "mha")

            # Compare original and final
            original_img = sitk.ReadImage(
                os.path.join(mha_dir, "image", "case_0000.mha")
            )
            final_img = sitk.ReadImage(os.path.join(mha2_dir, "image", "case_0000.mha"))

            # Check metadata
            assert original_img.GetSize() == final_img.GetSize()
            for i in range(3):
                assert (
                    abs(original_img.GetSpacing()[i] - final_img.GetSpacing()[i])
                    < SPACING_TOLERANCE
                )

            # Check data
            original_arr = sitk.GetArrayFromImage(original_img)
            final_arr = sitk.GetArrayFromImage(final_img)
            np.testing.assert_array_equal(original_arr, final_arr)


@pytest.mark.itk_process
class TestFormatConversionCLI:
    """Test format conversion CLI integration."""

    def test_format_subcommand_basic(self, monkeypatch):
        """Test basic format subcommand parsing."""
        test_args = ["itk_convert", "format", "nii.gz", "/source", "/dest"]
        monkeypatch.setattr("sys.argv", test_args)

        args = itk_convert.parse_args()

        assert args.format == "format"
        assert args.target_format == "nii.gz"
        assert args.source_folder == "/source"
        assert args.dest_folder == "/dest"

    def test_format_subcommand_with_options(self, monkeypatch):
        """Test format subcommand with all options."""
        test_args = [
            "itk_convert",
            "format",
            "nrrd",
            "/source",
            "/dest",
            "--mp",
            "--workers",
            "4",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        args = itk_convert.parse_args()

        assert args.format == "format"
        assert args.target_format == "nrrd"
        assert args.mp is True
        assert args.workers == 4

    def test_main_format_success(self, monkeypatch):
        """Test main with successful format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dest_dir = os.path.join(tmpdir, "output")

            setup_itkit_dataset(src_dir, num_samples=2)

            test_args = ["itk_convert", "format", "nii.gz", src_dir, dest_dir]
            monkeypatch.setattr("sys.argv", test_args)

            result = itk_convert.main()

            assert result == 0
            assert os.path.exists(os.path.join(dest_dir, "image"))
            assert os.path.exists(os.path.join(dest_dir, "label"))

    def test_main_format_invalid_source(self, monkeypatch, capsys):
        """Test main with invalid source folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "nonexistent")
            dest_dir = os.path.join(tmpdir, "output")

            test_args = ["itk_convert", "format", "nii.gz", src_dir, dest_dir]
            monkeypatch.setattr("sys.argv", test_args)

            result = itk_convert.main()

            assert result == 1
            captured = capsys.readouterr()
            assert "Error during conversion" in captured.out
