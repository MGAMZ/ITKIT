import os, pdb, argparse, json
from collections.abc import Sequence
from typing import Literal
from pathlib import Path
from enum import Enum

import numpy as np
import SimpleITK as sitk

from itkit.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size, sitk_resample_to_image
from itkit.process.metadata_models import SeriesMetadata
from itkit.process.base_processor import DatasetProcessor, SingleFolderProcessor


class ResamplingMode(Enum):
    """Enumeration for resampling modes.
    
    - SPACING_SIZE: Use explicit spacing/size rules (dimension-wise)
    - TARGET_IMAGE: Use target image as reference
    """
    SPACING_SIZE = "spacing_size"
    TARGET_IMAGE = "target_image"


class _ResampleMixin:
    """Mixin class for shared resampling logic.
    
    This mixin supports two resampling modes:
    1. SPACING_SIZE: Resample using dimension-wise spacing/size rules
    2. TARGET_IMAGE: Resample to match a target reference image
    
    Subclasses must implement:
    - _get_target_path(input_path, field): Return the path to target reference image,
                                           or None if target image mode is not used.
    """
    
    def resample_one_sample(self, input_path: str, field: Literal['image', 'label'], output_path: str) -> None | SeriesMetadata:
        """Resample a single sample image/label file.
        
        Args:
            input_path: Path to input image
            field: 'image' or 'label' to indicate field type
            output_path: Path to save output (will be converted to .mha)
            
        Returns:
            SeriesMetadata if successful, None if skipped or failed
        """
        self.resampling_mode: ResamplingMode  # Should be set by subclass
        
        output_path = output_path.replace(".nii.gz", ".mha").replace(".nii", ".mha").replace(".mhd", ".mha")
        
        try:
            image_itk = sitk.ReadImage(input_path)
        except Exception as e:
            print(f"Error reading {input_path}: {e}")
            return None
        
        # Branch based on resampling mode
        if self.resampling_mode == ResamplingMode.TARGET_IMAGE:
            image_resampled = self._resample_to_target_image(image_itk, input_path, field)
            if image_resampled is None:
                return None
        else:  # ResamplingMode.SPACING_SIZE
            image_resampled = self._apply_spacing_size_rules(image_itk, field)
        
        # Save output
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(image_resampled, output_path, useCompression=True)
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
            return None

        return SeriesMetadata(
            name=Path(output_path).name,
            spacing=image_resampled.GetSpacing()[::-1],
            size=image_resampled.GetSize()[::-1],
            origin=image_resampled.GetOrigin()[::-1],
            include_classes=np.unique(sitk.GetArrayFromImage(image_resampled)).tolist() 
                           if field == "label" else None
        )

    def _get_target_path(self, input_path: str, field: Literal['image', 'label']) -> str | None:
        """Get the path to target reference image.
        
        This method should be implemented by subclasses to provide the mapping
        from input file to target file based on the specific folder structure.
        
        Args:
            input_path: Path to input file
            field: 'image' or 'label'
            
        Returns:
            Path to target image, or None if no mapping exists
        """
        raise NotImplementedError("Subclass must implement _get_target_path()")

    def _resample_to_target_image(self, image_itk: sitk.Image, input_path: str, field: Literal['image', 'label']) -> sitk.Image | None:
        """Resample using a target reference image.
        
        Args:
            image_itk: Input SimpleITK image
            input_path: Path to input file
            field: 'image' or 'label'
            
        Returns:
            Resampled image, or None if target not found or loading failed
        """
        target_path = self._get_target_path(input_path, field)
        if target_path is None:
            print(f"Warning: No target path for {input_path}. Skipping.")
            return None
        
        if not os.path.exists(target_path):
            print(f"Warning: Target file not found: {target_path}. Skipping.")
            return None
        
        try:
            target_image = sitk.ReadImage(target_path)
            return sitk_resample_to_image(image_itk, target_image, field)
        except Exception as e:
            print(f"Error reading or resampling with target {target_path}: {e}")
            return None

    def _apply_spacing_size_rules(self, image_itk: sitk.Image, field: Literal['image', 'label']) -> sitk.Image:
        """Resample using dimension-wise spacing/size rules.
        
        Args:
            image_itk: Input SimpleITK image
            field: 'image' or 'label' (used for interpolation method selection)
            
        Returns:
            Resampled image with LPI orientation
        """
        self.target_spacing: Sequence[float]
        self.target_size: Sequence[int]
        
        # Stage 1: Spacing resample
        orig_spacing = image_itk.GetSpacing()[::-1]
        effective_spacing = list(orig_spacing)
        needs_spacing_resample = False
        
        for i in range(3):
            if self.target_spacing[i] != -1:
                effective_spacing[i] = self.target_spacing[i]
                needs_spacing_resample = True
        
        image_after_spacing = image_itk
        if needs_spacing_resample and not np.allclose(effective_spacing, orig_spacing):
            image_after_spacing = sitk_resample_to_spacing(image_itk, effective_spacing, field)
        assert isinstance(image_after_spacing, sitk.Image), "Resampling failed, result is not a SimpleITK image."
        
        # Stage 2: Size resample
        current_size = image_after_spacing.GetSize()[::-1]
        effective_size = list(current_size)
        needs_size_resample = False
        
        for i in range(3):
            if self.target_size[i] != -1:
                effective_size[i] = self.target_size[i]
                needs_size_resample = True
        
        image_resampled = image_after_spacing
        if needs_size_resample and effective_size != list(current_size):
            image_resampled = sitk_resample_to_size(image_after_spacing, effective_size, field)
        
        # Stage 3: Orientation adjustment
        image_resampled = sitk.DICOMOrient(image_resampled, 'LPI')
        
        return image_resampled


class ResampleProcessor(DatasetProcessor, _ResampleMixin):
    """Processor for resampling datasets with image/label structure.
    
    Supports both SPACING_SIZE and TARGET_IMAGE resampling modes.
    For TARGET_IMAGE mode, expects target_folder to have the same structure
    as source_folder (with image/ and label/ subfolders).
    """
    
    def __init__(self,
                 source_folder: str,
                 dest_folder: str,
                 target_spacing: Sequence[float] | None,
                 target_size: Sequence[int] | None,
                 mp: bool = False,
                 workers: int | None = None,
                 target_folder: str | None = None):
        super().__init__(source_folder, dest_folder, mp=mp, workers=workers)
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.target_folder = target_folder
        
        # Determine resampling mode based on parameters
        if target_folder is not None:
            self.resampling_mode = ResamplingMode.TARGET_IMAGE
        else:
            self.resampling_mode = ResamplingMode.SPACING_SIZE

    def _get_target_path(self, input_path: str, field: Literal['image', 'label']) -> str | None:
        assert self.resampling_mode == ResamplingMode.TARGET_IMAGE, "Target path requested in non-target-image mode."
        assert self.target_folder is not None, "Target folder must be specified for target image mode."
        
        # Compute relative path from the field-specific subfolder
        source_base_folder = os.path.join(self.source_folder, field)
        target_rel = os.path.relpath(input_path, source_base_folder)
        
        # Target is in target_folder/field/relative_path
        target_path = os.path.join(self.target_folder, field, target_rel)
        return target_path

    def process_one(self, args) -> None | SeriesMetadata:
        """Process one image-label pair.
        
        Skips if both output files already exist.
        """
        assert self.dest_folder is not None, "Destination folder must be specified."
        img_path, lbl_path = args
        img_out_path = os.path.join(self.dest_folder, "image", os.path.basename(img_path))
        lbl_out_path = os.path.join(self.dest_folder, "label", os.path.basename(lbl_path))
        
        if Path(img_out_path).exists() and Path(lbl_out_path).exists():
            print(f"Output files already exist, skipping: {img_out_path}, {lbl_out_path}")
            return None
        
        img_meta = self.resample_one_sample(
            input_path=img_path,
            field="image",
            output_path=img_out_path
        )
        lbl_meta = self.resample_one_sample(
            input_path=lbl_path,
            field="label",
            output_path=lbl_out_path
        )
        return lbl_meta


class SingleResampleProcessor(SingleFolderProcessor, _ResampleMixin):
    """Processor for resampling single folders (image or label mode).
    
    Supports both SPACING_SIZE and TARGET_IMAGE resampling modes.
    For TARGET_IMAGE mode, target_folder should be a flat folder containing
    reference images matching those in source_folder.
    """
    
    def __init__(self,
                 source_folder: str,
                 dest_folder: str,
                 target_spacing: Sequence[float] | None,
                 target_size: Sequence[int] | None,
                 field,
                 recursive: bool = False,
                 mp: bool = False,
                 workers: int | None = None,
                 target_folder: str | None = None):
        super().__init__(source_folder, dest_folder, mp=mp, workers=workers, recursive=recursive)
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.field = field
        self.target_folder = target_folder
        self.dest_folder: str
        
        # Determine resampling mode based on parameters
        if target_folder is not None:
            self.resampling_mode = ResamplingMode.TARGET_IMAGE
        else:
            self.resampling_mode = ResamplingMode.SPACING_SIZE

    def _get_target_path(self, input_path: str, field: Literal['image', 'label']) -> str | None:
        assert self.resampling_mode == ResamplingMode.TARGET_IMAGE, "Target path requested in non-target-image mode."
        assert self.target_folder is not None, "Target folder must be specified for target image mode."
        
        if self.recursive:
            # Preserve relative path structure
            target_rel = os.path.relpath(input_path, self.source_folder)
            target_path = os.path.join(self.target_folder, target_rel)
        else:
            # Just use the filename
            target_path = os.path.join(self.target_folder, os.path.basename(input_path))
        
        return target_path
    
    def process_one(self, file_path: str):
        """Process a single file.
        
        Skips if output file already exists.
        """
        if self.recursive:
            rel_path = os.path.relpath(file_path, self.source_folder)
            output_path = os.path.join(self.dest_folder, rel_path)
        else:
            output_path = os.path.join(self.dest_folder, os.path.basename(file_path))
        
        if Path(output_path).exists():
            print(f"Output file already exists, skipping: {output_path}")
            return None
        
        return self.resample_one_sample(file_path, self.field, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Resample a dataset with dimension-wise spacing/size rules or target image.")
    parser.add_argument("mode", type=str, choices=["image", "label", "dataset"], help="Resample mode: single-folder 'image'/'label' or paired 'dataset'.")
    parser.add_argument("source_folder", type=str, help="The source folder. For 'dataset' mode, it must contain 'image' and 'label' subfolders.")
    parser.add_argument("dest_folder", type=str, help="The destination folder. For 'dataset' mode, outputs to 'image' and 'label' subfolders.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively process subdirectories.")
    parser.add_argument("--mp", action="store_true", help="Whether to use multiprocessing.")
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")

    # Allow specifying both lists; -1 means ignore that dimension
    # Accept as str first to conveniently handle -1
    parser.add_argument("--spacing", type=str, nargs='+', default=["-1", "-1", "-1"],
                        help="Target spacing (ZYX order). Use -1 to ignore a dimension (e.g., 1.5 -1 1.5)")
    parser.add_argument("--size", type=str, nargs='+', default=["-1", "-1", "-1"],
                        help="Target size (ZYX order). Use -1 to ignore a dimension (e.g., -1 256 256)")
    
    # target_folder mode
    parser.add_argument("--target-folder", dest="target_folder", type=str, default=None,
                        help="Folder containing target reference images. For 'dataset' mode it should contain matching 'image' and 'label' subfolders. Mutually exclusive with --spacing and --size.")
    
    return parser.parse_args()


def validate_and_prepare_args(args):
    """Validate arguments and prepare resampling parameters.
    
    Enforces mutual exclusivity between --target-folder and --spacing/--size.
    
    Returns:
        (target_spacing, target_size): Lists of target values for each dimension.
                                       Use -1 to indicate "no change" for that dimension.
                                       Returns (None, None) only when no resampling is specified.
    """
    # Check mutual exclusivity between target_folder and spacing/size
    target_specified = args.target_folder is not None
    spacing_specified = any(s != "-1" for s in args.spacing)
    size_specified = any(s != "-1" for s in args.size)
    
    if target_specified and (spacing_specified or size_specified):
        raise ValueError(
            "--target-folder is mutually exclusive with --spacing and --size. "
            "Use either --target-folder or --spacing/--size, not both."
        )
    
    if target_specified:
        # Target image mode: spacing/size parameters are not used
        # Validate target folder exists
        if not os.path.isdir(args.target_folder):
            raise ValueError(f"Target folder does not exist: {args.target_folder}")
        # Return placeholder values that won't be accessed
        target_spacing = [-1, -1, -1]
        target_size = [-1, -1, -1]
    else:
        # Spacing/size mode: parse and validate parameters
        target_spacing = [float(s) for s in args.spacing]
        target_size = [int(s) for s in args.size]

        # Validate list lengths
        if len(target_spacing) != 3:
            raise ValueError(f"--spacing must have 3 values (received {len(target_spacing)})")
        if len(target_size) != 3:
            raise ValueError(f"--size must have 3 values (received {len(target_size)})")

        # Validate per-dimension exclusivity (can't specify both spacing and size for same dimension)
        for i in range(3):
            if target_spacing[i] != -1 and target_size[i] != -1:
                raise ValueError(f"Cannot specify both spacing and size for dimension {i}.")
                
        # Check if any resampling is actually specified
        if all(s == -1 for s in target_spacing) and all(sz == -1 for sz in target_size):
            print("Warning: No spacing or size specified, skipping resampling.")
            return None, None

    # Print configuration
    print(f"Resampling {args.source_folder} -> {args.dest_folder}")
    if target_specified:
        print(f"  Mode: TARGET_IMAGE from {args.target_folder}")
    else:
        print(f"  Mode: SPACING_SIZE")
        print(f"  Spacing: {target_spacing} | Size: {target_size}")
    print(f"  Recursive: {args.recursive} | Multiprocessing: {args.mp} | Workers: {args.workers}")
    
    return target_spacing, target_size


def main():
    args = parse_args()
    target_spacing, target_size = validate_and_prepare_args(args)
    
    # Handle case when no resampling rules are specified
    if target_spacing is None or target_size is None:
        # In this case, --target-folder must be provided
        if args.target_folder is None:
            raise ValueError(
                "No resampling parameters specified. "
                "Either use --spacing/--size or --target-folder."
            )
        print(f"Using target image folder for resampling: {args.target_folder}")

    # Save configuration
    config_data = vars(args)
    config_data['target_spacing_validated'] = target_spacing
    config_data['target_size_validated'] = target_size
    try:
        os.makedirs(args.dest_folder, exist_ok=True)
        with open(os.path.join(args.dest_folder, "resample_configs.json"), "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")

    # Execute using appropriate processor
    if args.mode == "dataset":
        processor = ResampleProcessor(
            args.source_folder,
            args.dest_folder,
            target_spacing,
            target_size,
            args.mp,
            args.workers,
            args.target_folder
        )
    else:
        processor = SingleResampleProcessor(
            args.source_folder,
            args.dest_folder,
            target_spacing,
            target_size,
            args.mode,
            args.recursive,
            args.mp,
            args.workers,
            args.target_folder
        )
    
    processor.process()
    print(f"Resampling completed. The resampled dataset is saved in {args.dest_folder}.")



if __name__ == '__main__':
    main()
