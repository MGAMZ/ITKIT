import os, pdb, argparse, json
from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk

from itkit.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size, sitk_resample_to_image
from itkit.process.base_processor import DatasetProcessor, SingleFolderProcessor


class _ResampleMixin:
    """Mixin class for shared resampling logic."""
    
    def resample_one_sample(self, input_path:str, field:str, output_path:str):
        # Normalize extension to .mha
        output_path = output_path.replace(".nii.gz", ".mha").replace(".nii", ".mha").replace(".mhd", ".mha")
        if os.path.exists(output_path):
            return None
            
        try:
            image_itk = sitk.ReadImage(input_path)
        except Exception as e:
            print(f"Error reading {input_path}: {e}")
            return None
        
        # Apply resampling logic
        if self.target_folder:
            # Use target image for resampling
            # Note: The logic for finding the relative path differs slightly between processors.
            # This part is kept in the specific processor's `process_one` method.
            # Here, we assume a direct mapping can be found.
            source_base_folder = self.source_folder
            if isinstance(self, DatasetProcessor):
                # For dataset mode, relpath should be from 'image' or 'label' subfolder
                source_base_folder = os.path.join(self.source_folder, field)

            target_rel = os.path.relpath(input_path, source_base_folder)
            target_path = os.path.join(self.target_folder, field if isinstance(self, DatasetProcessor) else "", target_rel)
            
            if os.path.exists(target_path):
                target_image = sitk.ReadImage(target_path)
                image_resampled = sitk_resample_to_image(image_itk, target_image, field)
            else:
                print(f"Warning: Target file not found for {input_path} at {target_path}. Skipping.")
                return None
        else:
            # Use spacing/size rules
            image_resampled = self._apply_spacing_size_rules(image_itk, field)
        
        # Save output
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(image_resampled, output_path, useCompression=True)
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
            return None
        
        # Return metadata
        final_spacing = image_resampled.GetSpacing()[::-1]
        final_size = image_resampled.GetSize()[::-1]
        final_origin = image_resampled.GetOrigin()[::-1]
        name = os.path.basename(input_path)
        
        return {name: {"spacing": final_spacing, "size": final_size, "origin": final_origin}}

    def _apply_spacing_size_rules(self, image_itk:sitk.Image, field:str):
        """Apply spacing and size resampling rules"""
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
    """Processor for resampling datasets with image/label structure"""
    
    def __init__(self,
                 source_folder: str,
                 dest_folder: str,
                 target_spacing: Sequence[float],
                 target_size: Sequence[float],
                 recursive: bool = False,
                 mp: bool = False,
                 workers: int | None = None,
                 target_folder: str | None = None):
        super().__init__(source_folder, dest_folder, mp, workers, recursive)
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.target_folder = target_folder
    
    def process_one(self, args):
        """Process one image-label pair"""
        assert self.dest_folder is not None, "Destination folder must be specified."
        img_path, lbl_path = args
        
        # Process image
        img_meta = self.resample_one_sample(
            img_path, "image", 
            os.path.join(self.dest_folder, "image", os.path.basename(img_path))
        )
        
        # Process label  
        lbl_meta = self.resample_one_sample(
            lbl_path, "label",
            os.path.join(self.dest_folder, "label", os.path.basename(lbl_path))
        )
        
        return {"image": img_meta, "label": lbl_meta} if img_meta or lbl_meta else None


class SingleResampleProcessor(SingleFolderProcessor, _ResampleMixin):
    """Processor for resampling single folders (image or label mode)"""
    
    def __init__(self,
                 source_folder: str,
                 dest_folder: str,
                 target_spacing: Sequence[float],
                 target_size: Sequence[float],
                 field,
                 recursive: bool = False,
                 mp: bool = False,
                 workers: int | None = None,
                 target_folder: str | None = None):
        super().__init__(source_folder, dest_folder, mp, workers, recursive)
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.field = field
        self.target_folder = target_folder
        self.dest_folder: str
    
    def process_one(self, file_path:str):
        """Process one file"""
        # Determine output path
        if self.recursive:
            rel_path = os.path.relpath(file_path, self.source_folder)
            output_path = os.path.join(self.dest_folder, rel_path)
        else:
            output_path = os.path.join(self.dest_folder, os.path.basename(file_path))
        
        # Normalize extension to .mha
        output_path = output_path.replace(".nii.gz", ".mha").replace(".nii", ".mha").replace(".mhd", ".mha")
        
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
    """Validate arguments and prepare resampling parameters."""
    # Check mutual exclusivity between target_folder and spacing/size
    target_specified = args.target_folder is not None
    spacing_specified = any(s != "-1" for s in args.spacing)
    size_specified = any(s != "-1" for s in args.size)
    
    if target_specified and (spacing_specified or size_specified):
        raise ValueError("--target-folder is mutually exclusive with --spacing and --size. Use either --target-folder or --spacing/--size, not both.")
    
    if target_specified:
        # Use target_folder mode
        if not os.path.isdir(args.target_folder):
            raise ValueError(f"Target folder does not exist: {args.target_folder}")
        # Set invalid placeholders for spacing/size
        target_spacing = [-1, -1, -1]
        target_size = [-1, -1, -1]
    else:
        # Use spacing/size mode
        target_spacing = [float(s) for s in args.spacing]
        target_size = [int(s) for s in args.size]

        # Check list lengths match dimension count
        if len(target_spacing) != 3:
            raise ValueError(f"--spacing must have {3} values (received {len(target_spacing)})")
        if len(target_size) != 3:
                raise ValueError(f"--size must have {3} values (received {len(target_size)})")

        # Validate per-dimension exclusivity
        for i in range(3):
            if target_spacing[i] != -1 and target_size[i] != -1:
                raise ValueError(f"Cannot specify both spacing and size for dimension {i}.")
                
        # Ensure at least one resampling rule is specified
        if all(s == -1 for s in target_spacing) and all(sz == -1 for sz in target_size):
            print("Warning: No spacing or size specified, skipping resampling.")
            return None, None

    # Print configuration
    print(f"Resampling {args.source_folder} -> {args.dest_folder}")
    if target_specified:
        print(f"  Target Folder: {args.target_folder}")
    else:
        print(f"  Spacing: {target_spacing} | Size: {target_size}")
    print(f"  Mode: {args.mode} | Recursive: {args.recursive} | Multiprocessing: {args.mp} | Workers: {args.workers}")
    
    return target_spacing, target_size


def main():
    args = parse_args()
    target_spacing, target_size = validate_and_prepare_args(args)
    
    if target_spacing is None:
        return

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

    # Execute using new processors
    if args.mode == "dataset":
        processor = ResampleProcessor(
            args.source_folder, args.dest_folder, target_spacing, target_size,
            args.recursive, args.mp, args.workers, args.target_folder
        )
        processor.process()
        processor.save_meta()
    else:
        processor = SingleResampleProcessor(
            args.source_folder, args.dest_folder, target_spacing, target_size, args.mode,
            args.recursive, args.mp, args.workers, args.target_folder
        )
        processor.process()
        processor.save_meta()
    
    print(f"Resampling completed. The resampled dataset is saved in {args.dest_folder}.")



if __name__ == '__main__':
    main()
