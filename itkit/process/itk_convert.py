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

import argparse

from itkit.process.itk_convert_monai import convert_to_monai
from itkit.process.itk_convert_torchio import convert_to_torchio


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

    # TorchIO subcommand
    torchio_parser = subparsers.add_parser(
        "torchio",
        help="Convert ITKIT dataset to TorchIO format",
        description="Convert ITKIT's mha-based dataset structure to TorchIO format "
        "with NIfTI files and subjects.csv manifest.",
    )

    torchio_parser.add_argument(
        "source_folder",
        type=str,
        help="Path to ITKIT dataset (must contain 'image' and 'label' subfolders)",
    )
    torchio_parser.add_argument(
        "dest_folder",
        type=str,
        help="Path to output TorchIO-format dataset",
    )
    torchio_parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip creating subjects.csv manifest file",
    )
    torchio_parser.add_argument(
        "--mp",
        action="store_true",
        help="Enable multiprocessing",
    )
    torchio_parser.add_argument(
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
        print("Available formats: monai, torchio")
        print("\nUsage:")
        print("  itk_convert monai <source_folder> <dest_folder> [options]")
        print("  itk_convert torchio <source_folder> <dest_folder> [options]")
        return 1

    if args.format == "monai":
        # Parse modality
        modality = {"0": args.modality}

        # Parse labels if provided
        labels = None
        if args.labels:
            labels = {str(i): name for i, name in enumerate(args.labels)}

        # Run conversion
        print("Converting ITKIT dataset to MONAI format...")
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
            print("\nConversion completed successfully!")
            print(f"Output saved to: {args.dest_folder}")
            return 0
        except Exception as e:
            print(f"\nError during conversion: {e}")
            return 1

    elif args.format == "torchio":
        # Run conversion
        print("Converting ITKIT dataset to TorchIO format...")
        print(f"  Source: {args.source_folder}")
        print(f"  Destination: {args.dest_folder}")
        print(f"  Create CSV manifest: {not args.no_csv}")

        try:
            convert_to_torchio(
                source_folder=args.source_folder,
                dest_folder=args.dest_folder,
                create_csv=not args.no_csv,
                mp=args.mp,
                workers=args.workers,
            )
            print("\nConversion completed successfully!")
            print(f"Output saved to: {args.dest_folder}")
            return 0
        except Exception as e:
            print(f"\nError during conversion: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
