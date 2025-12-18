"""
ITKIT to other Dataset Format Converter.

ITKIT Structure:
    dataset/
    ├── image/
    │   ├── case_001.mha
    │   └── case_002.mha
    └── label/
        ├── case_001.mha
        └── case_002.mha

MONAI Decathlon Structure:
    dataset/
    ├── imagesTr/
    │   ├── case_001.nii.gz
    │   └── case_002.nii.gz
    ├── labelsTr/
    │   ├── case_001.nii.gz
    │   └── case_002.nii.gz
    ├── imagesTs/        # Optional test images
    │   └── case_003.nii.gz
    └── dataset.json     # Manifest file with metadata
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from itkit.process.base_processor import BaseITKProcessor
from itkit.process.metadata_models import SeriesMetadata


class MonaiDatasetJson:
    """Helper class to generate MONAI-compatible dataset.json manifest."""

    def __init__(
        self,
        name: str = "ITKITDataset",
        description: str = "Dataset converted from ITKIT mha format",
        reference: str = "",
        license: str = "",
        release: str = "1.0",
        tensorImageSize: str = "3D",
        modality: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
    ):
        self.name = name
        self.description = description
        self.reference = reference
        self.license = license
        self.release = release
        self.tensorImageSize = tensorImageSize
        self.modality = modality or {"0": "CT"}
        self.labels = labels or {"0": "background"}
        self.training: list[dict[str, str]] = []
        self.validation: list[dict[str, str]] = []
        self.test: list[dict[str, str]] = []

    def add_training_sample(self, image_path: str, label_path: str):
        """Add a training sample with image and label paths (relative to dataset root)."""
        self.training.append({"image": image_path, "label": label_path})

    def add_validation_sample(self, image_path: str, label_path: str):
        """Add a validation sample with image and label paths."""
        self.validation.append({"image": image_path, "label": label_path})

    def add_test_sample(self, image_path: str):
        """Add a test sample with only image path."""
        self.test.append({"image": image_path})

    def update_labels_from_classes(self, unique_classes: set[int]):
        """Update labels dictionary from discovered unique classes."""
        self.labels = {"0": "background"}
        for cls in sorted(unique_classes):
            if cls != 0:
                self.labels[str(cls)] = f"class_{cls}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "name": self.name,
            "description": self.description,
            "reference": self.reference,
            "license": self.license,
            "release": self.release,
            "tensorImageSize": self.tensorImageSize,
            "modality": self.modality,
            "labels": self.labels,
            "numTraining": len(self.training),
            "numValidation": len(self.validation),
            "numTest": len(self.test),
            "training": self.training,
        }
        if self.validation:
            data["validation"] = self.validation
        if self.test:
            data["test"] = self.test
        return data

    def save(self, path: str | Path):
        """Save dataset.json to the specified path."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


def _convert_single_file(args: tuple[str, str]) -> dict[str, Any] | None:
    """Convert a single mha file to nii.gz format.

    Args:
        args: Tuple of (input_path, output_path)

    Returns:
        Dictionary with conversion result or None if failed
    """
    input_path, output_path = args

    try:
        # Read the mha file
        image = sitk.ReadImage(input_path)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write as nii.gz
        sitk.WriteImage(image, output_path, useCompression=True)

        # Get unique classes if it's a label file
        unique_classes = None
        if image.GetPixelID() == sitk.sitkUInt8:
            arr = sitk.GetArrayFromImage(image)
            unique_classes = np.unique(arr).tolist()

        return {
            "input": input_path,
            "output": output_path,
            "success": True,
            "unique_classes": unique_classes,
        }
    except Exception as e:
        return {
            "input": input_path,
            "output": output_path,
            "success": False,
            "error": str(e),
        }


class MonaiConverter(BaseITKProcessor):
    """Processor for converting ITKIT dataset structure to MONAI format."""

    def __init__(
        self,
        source_folder: str,
        dest_folder: str,
        dataset_name: str = "ITKITDataset",
        description: str = "Dataset converted from ITKIT mha format",
        modality: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        split: Literal["train", "val", "test", "all"] = "train",
        mp: bool = False,
        workers: int | None = None,
    ):
        """Initialize the MONAI converter.

        Args:
            source_folder: Path to ITKIT dataset (with image/ and label/ subfolders)
            dest_folder: Path to output MONAI-format dataset
            dataset_name: Name for the dataset in dataset.json
            description: Description for the dataset
            modality: Modality mapping (e.g., {"0": "CT"})
            labels: Label mapping (e.g., {"0": "background", "1": "tumor"})
            split: Which split to treat the data as ("train", "val", "test", or "all")
            mp: Enable multiprocessing
            workers: Number of worker processes
        """
        super().__init__(mp=mp, workers=workers, task_description="MONAI Conversion")

        self.source_folder = source_folder
        self.dest_folder = dest_folder
        self.split = split

        # Initialize dataset.json builder
        self.dataset_json = MonaiDatasetJson(
            name=dataset_name,
            description=description,
            modality=modality,
            labels=labels,
        )

        # Track discovered classes
        self.discovered_classes: set[int] = set()

        # Validate source structure
        self._validate_source_structure()

    def _validate_source_structure(self):
        """Validate that source folder has the expected ITKIT structure."""
        img_dir = os.path.join(self.source_folder, "image")
        lbl_dir = os.path.join(self.source_folder, "label")

        if not os.path.isdir(img_dir):
            raise ValueError(f"Missing 'image' subfolder in {self.source_folder}")
        if not os.path.isdir(lbl_dir):
            raise ValueError(f"Missing 'label' subfolder in {self.source_folder}")

    def get_items_to_process(self) -> list[tuple[str, str, str, str]]:
        """Get all image-label pairs to convert.

        Returns:
            List of (img_input, img_output, lbl_input, lbl_output) tuples
        """
        img_dir = os.path.join(self.source_folder, "image")
        lbl_dir = os.path.join(self.source_folder, "label")

        # Determine output directories based on split
        if self.split == "test":
            img_out_dir = os.path.join(self.dest_folder, "imagesTs")
            lbl_out_dir = None  # Test set typically has no labels
        elif self.split == "val":
            img_out_dir = os.path.join(self.dest_folder, "imagesVal")
            lbl_out_dir = os.path.join(self.dest_folder, "labelsVal")
        else:  # "train" or "all"
            img_out_dir = os.path.join(self.dest_folder, "imagesTr")
            lbl_out_dir = os.path.join(self.dest_folder, "labelsTr")

        # Find matching pairs
        img_files = {
            self._normalize_filename(f): os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(self.SUPPORTED_EXTENSIONS)
        }
        lbl_files = {
            self._normalize_filename(f): os.path.join(lbl_dir, f)
            for f in os.listdir(lbl_dir)
            if f.endswith(self.SUPPORTED_EXTENSIONS)
        }

        items = []
        common_keys = set(img_files.keys()) & set(lbl_files.keys())

        for key in sorted(common_keys):
            img_input = img_files[key]
            lbl_input = lbl_files[key]

            # Output as .nii.gz
            output_name = f"{key}.nii.gz"
            img_output = os.path.join(img_out_dir, output_name)

            if lbl_out_dir:
                lbl_output = os.path.join(lbl_out_dir, output_name)
            else:
                lbl_output = ""  # No label output for test split

            items.append((img_input, img_output, lbl_input, lbl_output))

        return items

    def process_one(self, args: tuple[str, str, str, str]) -> SeriesMetadata | None:
        """Process a single image-label pair.

        Args:
            args: Tuple of (img_input, img_output, lbl_input, lbl_output)

        Returns:
            SeriesMetadata or None
        """
        img_input, img_output, lbl_input, lbl_output = args

        # Skip if output already exists
        if os.path.exists(img_output) and (not lbl_output or os.path.exists(lbl_output)):
            print(f"Skipping existing: {os.path.basename(img_output)}")
            return None

        # Convert image
        img_result = _convert_single_file((img_input, img_output))
        if img_result is None or not img_result.get("success"):
            error_msg = img_result.get("error", "Unknown error") if img_result else "Unknown error"
            print(f"Failed to convert image {img_input}: {error_msg}")
            return None

        # Convert label if needed
        if lbl_output:
            lbl_result = _convert_single_file((lbl_input, lbl_output))
            if lbl_result is None or not lbl_result.get("success"):
                error_msg = lbl_result.get("error", "Unknown error") if lbl_result else "Unknown error"
                print(f"Failed to convert label {lbl_input}: {error_msg}")
                return None

            # Track discovered classes
            if lbl_result.get("unique_classes"):
                self.discovered_classes.update(lbl_result["unique_classes"])

        # Read output image for metadata
        try:
            output_img = sitk.ReadImage(img_output)
            return SeriesMetadata.from_sitk_image(output_img, os.path.basename(img_output))
        except Exception:
            return None

    def process(self, desc: str | None = None):
        """Execute the conversion process."""
        items = self.get_items_to_process()

        if not items:
            print(f"No items found for conversion in {self.source_folder}")
            return

        # Create output directories
        os.makedirs(self.dest_folder, exist_ok=True)

        desc = desc or self.task_description
        if self.mp:
            # For multiprocessing, we need to handle class discovery differently
            with Pool(self.workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(self.process_one, items),
                        total=len(items),
                        desc=desc,
                        dynamic_ncols=True,
                    )
                )
        else:
            results = []
            for item in tqdm(items, desc=desc, dynamic_ncols=True):
                results.append(self.process_one(item))

        self._collect_results(results)

        # Build dataset.json entries
        self._build_dataset_json(items)

        # Update labels from discovered classes if no explicit labels were provided
        if self.dataset_json.labels == {"0": "background"} and self.discovered_classes:
            self.dataset_json.update_labels_from_classes(self.discovered_classes)

        # Save dataset.json
        self.dataset_json.save(os.path.join(self.dest_folder, "dataset.json"))
        print(f"Saved dataset.json to {self.dest_folder}")

        # Save metadata
        self.save_meta(os.path.join(self.dest_folder, "meta.json"))

    def _build_dataset_json(self, items: list[tuple[str, str, str, str]]):
        """Build dataset.json entries from processed items."""
        for img_input, img_output, lbl_input, lbl_output in items:
            # Get relative paths from dest_folder
            img_rel = os.path.relpath(img_output, self.dest_folder)
            if lbl_output:
                lbl_rel = os.path.relpath(lbl_output, self.dest_folder)

            # Add to appropriate section
            if self.split == "test":
                self.dataset_json.add_test_sample(f"./{img_rel}")
            elif self.split == "val":
                self.dataset_json.add_validation_sample(f"./{img_rel}", f"./{lbl_rel}")
            else:  # "train" or "all"
                self.dataset_json.add_training_sample(f"./{img_rel}", f"./{lbl_rel}")


def convert_to_monai(
    source_folder: str,
    dest_folder: str,
    dataset_name: str = "ITKITDataset",
    description: str = "Dataset converted from ITKIT mha format",
    modality: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    split: Literal["train", "val", "test", "all"] = "train",
    mp: bool = False,
    workers: int | None = None,
) -> MonaiConverter:
    """Convert ITKIT dataset to MONAI format.

    This is the main Python API for converting datasets.

    Args:
        source_folder: Path to ITKIT dataset (with image/ and label/ subfolders)
        dest_folder: Path to output MONAI-format dataset
        dataset_name: Name for the dataset in dataset.json
        description: Description for the dataset
        modality: Modality mapping (e.g., {"0": "CT"})
        labels: Label mapping (e.g., {"0": "background", "1": "tumor"})
        split: Which split to treat the data as ("train", "val", "test", or "all")
        mp: Enable multiprocessing
        workers: Number of worker processes

    Returns:
        The MonaiConverter instance after processing

    Example:
        >>> from itkit.process.itk_convert import convert_to_monai
        >>> converter = convert_to_monai(
        ...     source_folder="/data/itkit_dataset",
        ...     dest_folder="/data/monai_dataset",
        ...     dataset_name="MyDataset",
        ...     modality={"0": "CT"},
        ...     labels={"0": "background", "1": "liver", "2": "tumor"}
        ... )
    """
    converter = MonaiConverter(
        source_folder=source_folder,
        dest_folder=dest_folder,
        dataset_name=dataset_name,
        description=description,
        modality=modality,
        labels=labels,
        split=split,
        mp=mp,
        workers=workers,
    )
    converter.process()
    return converter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="itk_convert",
        description="Convert between different medical image dataset formats.",
    )

    subparsers = parser.add_subparsers(dest="format", help="Target format to convert to")

    # MONAI subcommand
    monai_parser = subparsers.add_parser(
        "monai",
        help="Convert ITKIT dataset to MONAI Decathlon format",
        description="Convert ITKIT's mha-based dataset structure to MONAI's recommended "
        "Decathlon format with NIfTI files and dataset.json manifest.",
    )

    monai_parser.add_argument(
        "source_folder",
        type=str,
        help="Path to ITKIT dataset (must contain 'image' and 'label' subfolders)",
    )
    monai_parser.add_argument(
        "dest_folder",
        type=str,
        help="Path to output MONAI-format dataset",
    )
    monai_parser.add_argument(
        "--name",
        type=str,
        default="ITKITDataset",
        help="Dataset name for dataset.json (default: ITKITDataset)",
    )
    monai_parser.add_argument(
        "--description",
        type=str,
        default="Dataset converted from ITKIT mha format",
        help="Dataset description for dataset.json",
    )
    monai_parser.add_argument(
        "--modality",
        type=str,
        default="CT",
        help="Primary imaging modality (default: CT)",
    )
    monai_parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split to treat the data as (default: train)",
    )
    monai_parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Label names in order (e.g., --labels background liver tumor). "
        "Index 0 is background, 1 is first class, etc.",
    )
    monai_parser.add_argument(
        "--mp",
        action="store_true",
        help="Enable multiprocessing",
    )
    monai_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: half of CPU cores)",
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    if args.format is None:
        print("Error: Please specify a format to convert to.")
        print("Available formats: monai")
        print("\nUsage: itk_convert monai <source_folder> <dest_folder> [options]")
        return 1

    if args.format == "monai":
        # Parse modality
        modality = {"0": args.modality}

        # Parse labels if provided
        labels = None
        if args.labels:
            labels = {str(i): name for i, name in enumerate(args.labels)}

        # Run conversion
        print(f"Converting ITKIT dataset to MONAI format...")
        print(f"  Source: {args.source_folder}")
        print(f"  Destination: {args.dest_folder}")
        print(f"  Dataset name: {args.name}")
        print(f"  Split: {args.split}")
        print(f"  Modality: {args.modality}")
        if labels:
            print(f"  Labels: {labels}")

        try:
            convert_to_monai(
                source_folder=args.source_folder,
                dest_folder=args.dest_folder,
                dataset_name=args.name,
                description=args.description,
                modality=modality,
                labels=labels,
                split=args.split,
                mp=args.mp,
                workers=args.workers,
            )
            print(f"\nConversion completed successfully!")
            print(f"Output saved to: {args.dest_folder}")
            return 0
        except Exception as e:
            print(f"\nError during conversion: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
