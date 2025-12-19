"""
Medical Image Format Converter.

Converts medical image files between different formats while preserving metadata
and maintaining the ITKIT dataset structure (image/ and label/ folders).

Supported formats:
    - .mha (MetaImage)
    - .mhd (MetaImage Header with separate .raw file)
    - .nii.gz (Compressed NIfTI)
    - .nii (NIfTI)
    - .nrrd (Nearly Raw Raster Data)

The converter ensures:
    1. Complete metadata preservation (spacing, origin, direction, etc.)
    2. Identical file structure (image/ and label/ folders with matching filenames)
    3. No data modification during conversion
"""

import os
from typing import Any

import SimpleITK as sitk
from tqdm import tqdm

from itkit.process.base_processor import BaseITKProcessor
from itkit.process.metadata_models import SeriesMetadata

# Supported format extensions and their characteristics
SUPPORTED_FORMATS = {
    "mha": {
        "extension": ".mha",
        "description": "MetaImage (single file)",
        "compression": True,
    },
    "mhd": {
        "extension": ".mhd",
        "description": "MetaImage Header (with separate .raw file)",
        "compression": False,
    },
    "nii.gz": {
        "extension": ".nii.gz",
        "description": "Compressed NIfTI",
        "compression": True,
    },
    "nii": {
        "extension": ".nii",
        "description": "NIfTI",
        "compression": False,
    },
    "nrrd": {
        "extension": ".nrrd",
        "description": "Nearly Raw Raster Data",
        "compression": True,
    },
}


def _convert_single_file(args: tuple[str, str, bool]) -> dict[str, Any]:
    """Convert a single medical image file to another format.

    Args:
        args: Tuple of (input_path, output_path, use_compression)

    Returns:
        Dictionary with conversion result (always includes 'success' key)
    """
    input_path, output_path, use_compression = args

    try:
        # Read the image with SimpleITK
        image = sitk.ReadImage(input_path)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write with the same metadata
        sitk.WriteImage(image, output_path, useCompression=use_compression)

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


class FormatConverter(BaseITKProcessor):
    """Processor for converting medical image files between formats."""

    def __init__(
        self,
        source_folder: str,
        dest_folder: str,
        target_format: str,
        mp: bool = False,
        workers: int | None = None,
    ):
        """Initialize the format converter.

        Args:
            source_folder: Path to source dataset (with image/ and label/ subfolders)
            dest_folder: Path to output dataset (will maintain same structure)
            target_format: Target format (e.g., 'nii.gz', 'mha', 'mhd', 'nrrd')
            mp: Enable multiprocessing
            workers: Number of worker processes
        """
        super().__init__(mp=mp, workers=workers, task_description="Format Conversion")

        self.source_folder = source_folder
        self.dest_folder = dest_folder
        self.target_format = target_format.lower()

        # Validate target format
        if self.target_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported target format: {target_format}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}"
            )

        self.format_info = SUPPORTED_FORMATS[self.target_format]
        self.target_extension = self.format_info["extension"]
        self.use_compression = self.format_info["compression"]

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

        # Output directories
        img_out_dir = os.path.join(self.dest_folder, "image")
        lbl_out_dir = os.path.join(self.dest_folder, "label")

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

            # Output with target extension
            output_name = f"{key}{self.target_extension}"
            img_output = os.path.join(img_out_dir, output_name)
            lbl_output = os.path.join(lbl_out_dir, output_name)

            items.append((img_input, img_output, lbl_input, lbl_output))

        return items

    def process_one(self, file_paths: tuple[str, str, str, str]) -> SeriesMetadata | None:
        """Process a single image-label pair.

        Args:
            file_paths: Tuple of (img_input, img_output, lbl_input, lbl_output)

        Returns:
            SeriesMetadata or None
        """
        img_input, img_output, lbl_input, lbl_output = file_paths

        # Skip if output already exists
        if os.path.exists(img_output) and os.path.exists(lbl_output):
            # Still check if output files are valid
            try:
                sitk.ReadImage(img_output)
                sitk.ReadImage(lbl_output)
                return None
            except Exception:
                # If files exist but are corrupted, re-convert them
                pass

        # Convert image
        img_result = _convert_single_file((img_input, img_output, self.use_compression))
        if not img_result.get("success"):
            error_msg = img_result.get("error", "Unknown error")
            print(f"Failed to convert image {img_input}: {error_msg}")
            return None

        # Convert label
        lbl_result = _convert_single_file((lbl_input, lbl_output, self.use_compression))
        if not lbl_result.get("success"):
            error_msg = lbl_result.get("error", "Unknown error")
            print(f"Failed to convert label {lbl_input}: {error_msg}")
            return None

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
        os.makedirs(os.path.join(self.dest_folder, "image"), exist_ok=True)
        os.makedirs(os.path.join(self.dest_folder, "label"), exist_ok=True)

        desc = desc or f"{self.task_description} to {self.target_format}"

        # Process items
        if self.mp:
            from multiprocessing import Pool

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

        # Save metadata
        self.save_meta(os.path.join(self.dest_folder, "meta.json"))
        self.save_meta(os.path.join(self.dest_folder, "image", "meta.json"))
        self.save_meta(os.path.join(self.dest_folder, "label", "meta.json"))


def convert_format(
    source_folder: str,
    dest_folder: str,
    target_format: str,
    mp: bool = False,
    workers: int | None = None,
) -> FormatConverter:
    """Convert medical image dataset between formats.

    This is the main Python API for format conversion.

    Args:
        source_folder: Path to source dataset (with image/ and label/ subfolders)
        dest_folder: Path to output dataset (will maintain same structure)
        target_format: Target format (e.g., 'nii.gz', 'mha', 'mhd', 'nrrd')
        mp: Enable multiprocessing
        workers: Number of worker processes

    Returns:
        The FormatConverter instance after processing

    Example:
        >>> from itkit.process.itk_convert_format import convert_format
        >>> converter = convert_format(
        ...     source_folder="/data/dataset_mha",
        ...     dest_folder="/data/dataset_nifti",
        ...     target_format="nii.gz",
        ...     mp=True
        ... )
    """
    converter = FormatConverter(
        source_folder=source_folder,
        dest_folder=dest_folder,
        target_format=target_format,
        mp=mp,
        workers=workers,
    )
    converter.process()
    return converter


def list_supported_formats():
    """Print all supported formats and their descriptions."""
    print("Supported medical image formats:")
    print("-" * 60)
    for fmt, info in SUPPORTED_FORMATS.items():
        compression = "Yes" if info["compression"] else "No"
        print(f"  {fmt:10s} - {info['description']:<30s} (Compression: {compression})")
    print("-" * 60)
