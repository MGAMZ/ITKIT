"""
ITKIT to TorchIO Format Converter.

TorchIO Structure:
    dataset/
    ├── images/
    │   ├── subject_001.nii.gz
    │   └── subject_002.nii.gz
    ├── labels/
    │   ├── subject_001.nii.gz
    │   └── subject_002.nii.gz
    └── subjects.csv     # Manifest file with paths
"""

import csv
import os
from multiprocessing import Pool
from typing import Any

import SimpleITK as sitk
from tqdm import tqdm

from itkit.process.base_processor import BaseITKProcessor
from itkit.process.metadata_models import SeriesMetadata


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

        return {
            "input": input_path,
            "output": output_path,
            "success": True,
        }
    except Exception as e:
        return {
            "input": input_path,
            "output": output_path,
            "success": False,
            "error": str(e),
        }


class TorchIOConverter(BaseITKProcessor):
    """Processor for converting ITKIT dataset structure to TorchIO format."""

    def __init__(
        self,
        source_folder: str,
        dest_folder: str,
        create_csv: bool = True,
        mp: bool = False,
        workers: int | None = None,
    ):
        """Initialize the TorchIO converter.

        Args:
            source_folder: Path to ITKIT dataset (with image/ and label/ subfolders)
            dest_folder: Path to output TorchIO-format dataset
            create_csv: Whether to create subjects.csv manifest file
            mp: Enable multiprocessing
            workers: Number of worker processes
        """
        super().__init__(mp=mp, workers=workers, task_description="TorchIO Conversion")

        self.source_folder = source_folder
        self.dest_folder = dest_folder
        self.create_csv = create_csv

        # Validate source structure
        self._validate_source_structure()

        # Store converted subjects for CSV generation
        self.subjects: list[dict[str, str]] = []

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

        # Output directories
        img_out_dir = os.path.join(self.dest_folder, "images")
        lbl_out_dir = os.path.join(self.dest_folder, "labels")

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
            lbl_output = os.path.join(lbl_out_dir, output_name)

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
        if os.path.exists(img_output) and os.path.exists(lbl_output):
            print(f"Skipping existing: {os.path.basename(img_output)}")
            return None

        # Convert image
        img_result = _convert_single_file((img_input, img_output))
        if img_result is None or not img_result.get("success"):
            error_msg = img_result.get("error", "Unknown error") if img_result else "Unknown error"
            print(f"Failed to convert image {img_input}: {error_msg}")
            return None

        # Convert label
        lbl_result = _convert_single_file((lbl_input, lbl_output))
        if lbl_result is None or not lbl_result.get("success"):
            error_msg = lbl_result.get("error", "Unknown error") if lbl_result else "Unknown error"
            print(f"Failed to convert label {lbl_input}: {error_msg}")
            return None

        # Store subject info for CSV
        subject_name = self._normalize_filename(os.path.basename(img_output))

        self.subjects.append({
            "subject": subject_name,
            "image": os.path.relpath(img_output, self.dest_folder),
            "label": os.path.relpath(lbl_output, self.dest_folder),
        })

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

        # Create subjects.csv if requested
        if self.create_csv and self.subjects:
            self._create_subjects_csv()

        # Save metadata
        self.save_meta(os.path.join(self.dest_folder, "meta.json"))

    def _create_subjects_csv(self):
        """Create subjects.csv manifest file."""
        csv_path = os.path.join(self.dest_folder, "subjects.csv")

        # Sort subjects by name for consistency
        self.subjects.sort(key=lambda x: x["subject"])

        with open(csv_path, "w", newline="") as f:
            fieldnames = ["subject", "image", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.subjects)

        print(f"Created subjects.csv manifest at {csv_path}")


def convert_to_torchio(
    source_folder: str,
    dest_folder: str,
    create_csv: bool = True,
    mp: bool = False,
    workers: int | None = None,
) -> TorchIOConverter:
    """Convert ITKIT dataset to TorchIO format.

    This is the main Python API for converting datasets.

    Args:
        source_folder: Path to ITKIT dataset (with image/ and label/ subfolders)
        dest_folder: Path to output TorchIO-format dataset
        create_csv: Whether to create subjects.csv manifest file
        mp: Enable multiprocessing
        workers: Number of worker processes

    Returns:
        The TorchIOConverter instance after processing

    Example:
        >>> from itkit.process.itk_convert_torchio import convert_to_torchio
        >>> converter = convert_to_torchio(
        ...     source_folder="/data/itkit_dataset",
        ...     dest_folder="/data/torchio_dataset",
        ...     create_csv=True
        ... )
    """
    converter = TorchIOConverter(
        source_folder=source_folder,
        dest_folder=dest_folder,
        create_csv=create_csv,
        mp=mp,
        workers=workers,
    )
    converter.process()
    return converter
