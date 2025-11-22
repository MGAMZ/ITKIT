import argparse
import os
import pdb
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from itkit.io.sitk_toolkit import INTERPOLATOR
from itkit.process.base_processor import SeparateFoldersProcessor
from itkit.process.metadata_models import SeriesMetadata


class AugProcessor(SeparateFoldersProcessor):
    def __init__(self,
                 img_folder: str,
                 lbl_folder: str,
                 out_img_folder: str | None,
                 out_lbl_folder: str | None,
                 aug_num: int,
                 random_rots: list[int],
                 mp: bool = False,
                 workers: int | None = None):
        super().__init__(
            folder_A = img_folder,
            folder_B = lbl_folder,
            output_folder_A = out_img_folder,
            output_folder_B = out_lbl_folder,
            mp = mp,
            workers = workers
        )
        self.aug_num = aug_num
        self.random_rots = random_rots

    def process(self, desc: str | None = None):
        pairs = self.get_items_to_process()
        print(f"Found {len(pairs)} matching image-label pairs")
        if not pairs:
            return
        super().process(desc or "Augmenting")
        if self.dest_folder is not None:
            os.makedirs(self.dest_folder, exist_ok=True)
            self.save_meta(Path(self.dest_folder) / "meta.json")

    def random_3d_rotate(self, image: sitk.Image, label: sitk.Image, angle_ranges: Sequence[float]) -> tuple[sitk.Image, sitk.Image]:
        """
        Rotate one image-label pair.
        
        Args:
            image (sitk.Image): The input image to rotate.
            label (sitk.Image): The corresponding label image to rotate.
            angle_ranges (Sequence[float]):
                The range of angles (in degrees) for random rotation.
                Should contain three values corresponding to `Z, Y, X` axis.
        """
        
        radian_angles = [np.deg2rad(random.uniform(-angle_range, angle_range)) 
                         for angle_range in angle_ranges][::-1]
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        center_point = [origin[i] + spacing[i] * size[i] / 2.0 
                        for i in range(3)]
        
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center_point)
        transform.SetRotation(radian_angles[0], radian_angles[1], radian_angles[2])

        rotated_image = sitk.Resample(
            image,
            transform,
            INTERPOLATOR('image'),
            -3072
        )
        rotated_label = sitk.Resample(
            label,
            transform,
            INTERPOLATOR('label'),
            0
        )
        
        return rotated_image, rotated_label

    def process_one(self, args: tuple[str, str]) -> list[SeriesMetadata] | None:
        img_path, lbl_path = args
        
        # Paths
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        # Read
        try:
            image = sitk.ReadImage(img_path)
            label = sitk.ReadImage(lbl_path)
        except Exception as e:
            print(f"Error reading {img_path} or {lbl_path}: {e}")
            return None
        
        metadata_list = []
        # Multiple augmented samples from source sample.
        for i in range(self.aug_num):
            rotated_image, rotated_label = self.random_3d_rotate(image, label, self.random_rots)
            # save to mha
            if self.output_folder_A:
                aug_img_path = os.path.join(self.output_folder_A, f"{basename}_{i}.mha")
                os.makedirs(os.path.dirname(aug_img_path), exist_ok=True)
                sitk.WriteImage(rotated_image, aug_img_path, True)
                img_meta = SeriesMetadata(
                    name=Path(aug_img_path).name,
                    spacing=rotated_image.GetSpacing()[::-1],
                    size=rotated_image.GetSize()[::-1],
                    origin=rotated_image.GetOrigin()[::-1],
                    include_classes=None
                )
                metadata_list.append(img_meta)
            if self.output_folder_B:
                aug_lbl_path = os.path.join(self.output_folder_B, f"{basename}_{i}.mha")
                os.makedirs(os.path.dirname(aug_lbl_path), exist_ok=True)
                sitk.WriteImage(rotated_label, aug_lbl_path, True)
                lbl_meta = SeriesMetadata(
                    name=Path(aug_lbl_path).name,
                    spacing=rotated_label.GetSpacing()[::-1],
                    size=rotated_label.GetSize()[::-1],
                    origin=rotated_label.GetOrigin()[::-1],
                    include_classes=np.unique(sitk.GetArrayFromImage(rotated_label)).tolist()
                )
                metadata_list.append(lbl_meta)
        
        return metadata_list if metadata_list else None


def parse_args():
    parser = argparse.ArgumentParser(description='ITK data augmentation')
    parser.add_argument('img_folder', type=str, help='Folder containing image mhas')
    parser.add_argument('lbl_folder', type=str, help='Folder containing label mhas')
    parser.add_argument('-oimg', '--out-img-folder', type=str, default=None, help='Optional folder for augmented output image mhas.')
    parser.add_argument('-olbl', '--out-lbl-folder', type=str, default=None, help='Optional folder for augmented output label mhas.')
    parser.add_argument('-n', '--num', type=int, default=1, help='Number of augmented samples of each source sample.')
    parser.add_argument('--mp', action='store_true', help='Enable multiprocessing, the number of workers are `None`.')
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")
    parser.add_argument('--random-rot', type=int, nargs=3, default=None, help='Maximum ramdom rotation degree on `Z Y X` axis.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_img_folder:
        os.makedirs(args.out_img_folder, exist_ok=True)
    if args.out_lbl_folder:
        os.makedirs(args.out_lbl_folder, exist_ok=True)
    
    processor = AugProcessor(args.img_folder, args.lbl_folder, args.out_img_folder, args.out_lbl_folder, args.num, args.random_rot, args.mp, args.workers)
    processor.process()

    print("Data augmentation complete")


if __name__ == "__main__":
    main()
