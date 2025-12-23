import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from itkit.process.base_processor import DatasetProcessor, SingleFolderProcessor
from itkit.process.metadata_models import SeriesMetadata


class DatasetOrientProcessor(DatasetProcessor):
    """Processor for orienting datasets with image/label structure."""
    
    def __init__(self,
                 source_folder: str,
                 dest_folder: str,
                 orient: str,
                 mp: bool = False,
                 workers: int | None = None):
        super().__init__(source_folder, dest_folder, mp=mp, workers=workers)
        self.orient = orient
    
    def process_one(self, args: tuple[str, str]) -> SeriesMetadata | None:
        """Process one image-label pair."""
        img_path, lbl_path = args
        
        # Compute output paths preserving structure
        img_rel_path = os.path.relpath(img_path, os.path.join(self.source_folder, "image"))
        lbl_rel_path = os.path.relpath(lbl_path, os.path.join(self.source_folder, "label"))
        
        img_dst_path = os.path.join(self.dest_folder, "image", img_rel_path)
        lbl_dst_path = os.path.join(self.dest_folder, "label", lbl_rel_path)
        
        # Skip if both already exist
        if os.path.exists(img_dst_path) and os.path.exists(lbl_dst_path):
            print(f"Target files already exist, skipping: {img_dst_path}, {lbl_dst_path}")
            return None
        
        # Process image
        try:
            img = sitk.ReadImage(img_path)
            oriented_img = sitk.DICOMOrient(img, self.orient.upper())
            os.makedirs(os.path.dirname(img_dst_path), exist_ok=True)
            sitk.WriteImage(oriented_img, img_dst_path, True)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None
        
        # Process label
        try:
            lbl = sitk.ReadImage(lbl_path)
            oriented_lbl = sitk.DICOMOrient(lbl, self.orient.upper())
            os.makedirs(os.path.dirname(lbl_dst_path), exist_ok=True)
            sitk.WriteImage(oriented_lbl, lbl_dst_path, True)
        except Exception as e:
            print(f"Error processing label {lbl_path}: {e}")
            return None
        
        # Return metadata from label (include_classes will be computed)
        return SeriesMetadata(
            name=Path(lbl_dst_path).name,
            spacing=oriented_lbl.GetSpacing()[::-1],
            size=oriented_lbl.GetSize()[::-1],
            origin=oriented_lbl.GetOrigin()[::-1],
            include_classes=np.unique(sitk.GetArrayFromImage(oriented_lbl)).tolist()
        )


class OrientProcessor(SingleFolderProcessor):
    def __init__(self,
                 source_folder: str,
                 dest_folder: str,
                 orient: str,
                 field: str = "image",
                 mp: bool = False,
                 workers: int | None = None):
        super().__init__(source_folder, dest_folder, recursive=True, mp=mp, workers=workers)
        self.dest_folder: str
        self.orient = orient
        self.field = field

    def process_one(self, args: str) -> SeriesMetadata | None:
        file_path = args
        rel_path = os.path.relpath(file_path, self.source_folder)
        dst_path = os.path.join(self.dest_folder, rel_path)

        # Skip if target file already exists
        if os.path.exists(dst_path):
            print(f"Target file already exists, skipping: {dst_path}")
            return None

        try:
            img = sitk.ReadImage(file_path)
            oriented_img = sitk.DICOMOrient(img, self.orient.upper())
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            sitk.WriteImage(oriented_img, dst_path, True)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

        return SeriesMetadata(
            name=Path(dst_path).name,
            spacing=oriented_img.GetSpacing()[::-1],
            size=oriented_img.GetSize()[::-1],
            origin=oriented_img.GetOrigin()[::-1],
            include_classes=np.unique(sitk.GetArrayFromImage(oriented_img)).tolist() if self.field == "label" else None
        )


def main():
    parser = argparse.ArgumentParser(description="Convert all .mha files under the source directory to the specified orientation (e.g. LPI) while preserving the original directory structure.")
    parser.add_argument('src_dir', help='Source directory')
    parser.add_argument('dst_dir', help='Destination directory')
    parser.add_argument('orient', help='Target orientation (e.g. LPI)')
    parser.add_argument('--field', choices=['image', 'label', 'dataset'], default='image', help='Field type for metadata (use "dataset" for standard ITKIT structure with image/label folders)')
    parser.add_argument('--mp', action='store_true', help='Use multiprocessing')
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")
    args = parser.parse_args()

    if not os.path.isdir(args.src_dir):
        print(f"Source directory does not exist: {args.src_dir}")
        return

    if os.path.abspath(args.src_dir) == os.path.abspath(args.dst_dir):
        print("Source and destination directories cannot be the same!")
        return

    if args.field == 'dataset':
        # Check for standard ITKIT structure
        img_dir = os.path.join(args.src_dir, 'image')
        lbl_dir = os.path.join(args.src_dir, 'label')
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            print(f"Error: Dataset mode requires 'image' and 'label' subfolders in {args.src_dir}")
            return
        processor = DatasetOrientProcessor(args.src_dir, args.dst_dir, args.orient, args.mp, args.workers)
        processor.process("Orienting dataset")
    else:
        processor = OrientProcessor(args.src_dir, args.dst_dir, args.orient, args.field, args.mp, args.workers)
        processor.process("Orienting files")



if __name__ == '__main__':
    main()
