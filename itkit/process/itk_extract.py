import os, argparse, json, sys

import numpy as np
import SimpleITK as sitk

from itkit.process.base_processor import SingleFolderProcessor
from itkit.process.metadata_models import SeriesMetadata

DEFAULT_LABEL_DTYPE = np.uint8


class ExtractProcessor(SingleFolderProcessor):
    """Processor for extracting and remapping labels"""

    def __init__(
        self,
        source_folder: str,
        dest_folder: str,
        label_mapping: dict[int, int],
        recursive: bool = False,
        mp: bool = False,
        workers: int | None = None,
    ):
        super().__init__(
            source_folder,
            dest_folder,
            recursive,
            mp=mp,
            workers=workers,
            task_description="Extracting labels",
        )
        self.dest_folder: str
        self.label_mapping = label_mapping

    def process_one(self, file_path: str) -> SeriesMetadata | None:
        """Process one file"""
        # Determine output path
        if self.recursive:
            rel_path = os.path.relpath(file_path, self.source_folder)
            output_path = os.path.join(self.dest_folder, rel_path)
        else:
            output_path = os.path.join(self.dest_folder, os.path.basename(file_path))

        # Normalize extension to .mha
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        if base_name.endswith(".nii"):
            base_name = base_name[:-4]
        output_path = os.path.join(os.path.dirname(output_path), base_name + ".mha")

        return self._extract_one_sample(file_path, output_path)

    def _extract_one_sample(
        self, input_path: str, output_path: str
    ) -> SeriesMetadata | None:
        """Extract and remap labels from a single sample"""
        # Check if output already exists
        if os.path.exists(output_path):
            return None

        # Read image
        try:
            image_itk = sitk.ReadImage(input_path)
        except Exception as e:
            print(f"Error reading {input_path}: {e}")
            return None

        # Convert to numpy array
        try:
            image_array = sitk.GetArrayFromImage(image_itk)
            # Ensure uint dtype
            if not np.issubdtype(image_array.dtype, np.unsignedinteger):
                image_array = image_array.astype(DEFAULT_LABEL_DTYPE)
        except Exception as e:
            print(f"Error converting image to array {input_path}: {e}")
            return None

        # Create output array with same shape, initialized with background (0)
        output_array = np.zeros_like(image_array, dtype=DEFAULT_LABEL_DTYPE)

        # Apply label mappings
        original_labels = set()
        extracted_labels = set()

        for source_label, target_label in self.label_mapping.items():
            mask = image_array == source_label
            if np.any(mask):
                output_array[mask] = target_label
                original_labels.add(int(source_label))
                extracted_labels.add(int(target_label))

        # Convert back to SimpleITK image
        try:
            output_itk = sitk.GetImageFromArray(output_array)
            output_itk.CopyInformation(image_itk)
        except Exception as e:
            print(f"Error converting array to image {input_path}: {e}")
            return None

        # Write output
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(output_itk, output_path, useCompression=True)
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
            return None

        # Return metadata using SeriesMetadata
        return SeriesMetadata(
            name=os.path.basename(output_path),
            spacing=tuple(output_itk.GetSpacing()[::-1]),
            size=tuple(output_itk.GetSize()[::-1]),
            origin=tuple(output_itk.GetOrigin()[::-1]),
            include_classes=(
                tuple(sorted(extracted_labels)) if extracted_labels else None
            ),
        )


def parse_label_mappings(mapping_strings: list[str]) -> dict:
    """
    Parse label mapping strings in format "source:target" to dictionary.

    Args:
        mapping_strings: List of strings like ["1:0", "5:1", "3:2"]

    Returns:
        Dictionary mapping source labels to target labels
    """
    mapping = {}
    for mapping_str in mapping_strings:
        try:
            source, target = mapping_str.split(":")
            source_label = int(source)
            target_label = int(target)
            mapping[source_label] = target_label
        except ValueError:
            raise ValueError(
                f"Invalid mapping format: {mapping_str}. Expected 'source:target' format."
            )

    return mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and remap labels from a dataset."
    )
    parser.add_argument(
        "source_folder", type=str, help="The source folder containing .mha files."
    )
    parser.add_argument(
        "dest_folder", type=str, help="The destination folder for extracted files."
    )
    parser.add_argument(
        "mappings",
        type=str,
        nargs="+",
        help="Label mappings in format 'source:target' (e.g., '1:0' '5:1' '3:2')",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories.",
    )
    parser.add_argument(
        "--mp", action="store_true", help="Whether to use multiprocessing."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="The number of workers for multiprocessing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Parameter validation ---
    try:
        # Parse label mappings
        label_mapping = parse_label_mappings(args.mappings)

        if not label_mapping:
            raise ValueError("At least one label mapping must be specified.")

        # Check for duplicate target labels
        target_labels = list(label_mapping.values())
        if len(target_labels) != len(set(target_labels)):
            raise ValueError(
                "Duplicate target labels found. Each target label should be unique."
            )

        # Print configuration
        print(f"Extracting labels from {args.source_folder} -> {args.dest_folder}")
        print(f"  Label mappings: {label_mapping}")
        print(
            f"  Recursive: {args.recursive} | Multiprocessing: {args.mp} | Workers: {args.workers}"
        )

    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    # Save configuration
    config_data = vars(args)
    config_data["label_mapping"] = label_mapping
    try:
        os.makedirs(args.dest_folder, exist_ok=True)
        with open(os.path.join(args.dest_folder, "extract_configs.json"), "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")

    # Execute using new processor
    processor = ExtractProcessor(
        args.source_folder,
        args.dest_folder,
        label_mapping,
        args.recursive,
        args.mp,
        args.workers,
    )
    processor.process()

    print(
        f"Label extraction completed. The extracted dataset is saved in {args.dest_folder}."
    )


if __name__ == "__main__":
    main()
