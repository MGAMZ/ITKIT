import os, argparse, random, pdb
from collections.abc import Sequence
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
from itkit.io.sitk_toolkit import INTERPOLATOR
from itkit.process.base_processor import SeparateFoldersProcessor


class AugProcessor(SeparateFoldersProcessor):
    def __init__(self, img_folder, lbl_folder, out_img_folder, out_lbl_folder, num, random_rots, mp=False):
        super().__init__(img_folder, lbl_folder, out_img_folder, out_lbl_folder, mp)
        self.num = num
        self.random_rots = random_rots

    def process(self):
        pairs = self.find_pairs()
        print(f"Found {len(pairs)} matching image-label pairs")
        if not pairs:
            return
        self.process_items(pairs, "Augmenting")

    def random_3d_rotate(self, image, label, angle_ranges):
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

    def process_one(self, args):
        img_path, lbl_path = args
        
        # Paths
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        # Read
        image = sitk.ReadImage(img_path)
        label = sitk.ReadImage(lbl_path)
        
        # Multiple augmented samples from source sample.
        for i in range(self.num):
            rotated_image, rotated_label = self.random_3d_rotate(image, label, self.random_rots)
            # save to mha
            if self.out_img_folder:
                aug_img_path = os.path.join(self.out_img_folder, f"{basename}_{i}.mha")
                sitk.WriteImage(rotated_image, aug_img_path, True)
            if self.out_lbl_folder:
                aug_lbl_path = os.path.join(self.out_lbl_folder, f"{basename}_{i}.mha")
                sitk.WriteImage(rotated_label, aug_lbl_path, True)
        
        return None  # No metadata for augmentation


def parse_args():
    parser = argparse.ArgumentParser(description='ITK data augmentation')
    parser.add_argument('img_folder', type=str, help='Folder containing image mhas')
    parser.add_argument('lbl_folder', type=str, help='Folder containing label mhas')
    parser.add_argument('-oimg', '--out-img-folder', type=str, default=None, help='Optional folder for augmented output image mhas.')
    parser.add_argument('-olbl', '--out-lbl-folder', type=str, default=None, help='Optional folder for augmented output label mhas.')
    parser.add_argument('-n', '--num', type=int, default=1, help='Number of augmented samples of each source sample.')
    parser.add_argument('--mp', action='store_true', help='Enable multiprocessing, the number of workers are `None`.')
    parser.add_argument('--random-rot', type=int, nargs=3, default=None, help='Maximum ramdom rotation degree on `Z Y X` axis.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_img_folder:
        os.makedirs(args.out_img_folder, exist_ok=True)
    if args.out_lbl_folder:
        os.makedirs(args.out_lbl_folder, exist_ok=True)
    
    # Fetch files
    img_files = set(f for f in os.listdir(args.img_folder) if f.endswith('.mha'))
    lbl_files = set(f for f in os.listdir(args.lbl_folder) if f.endswith('.mha'))
    common_files = list(img_files.intersection(lbl_files))
    print(f"Found {len(common_files)} matching image-label pairs")
    
    processor = AugProcessor(args.img_folder, args.lbl_folder, args.out_img_folder, args.out_lbl_folder, args.num, args.random_rot, args.mp)
    processor.process()

    print("Data augmentation complete")


if __name__ == "__main__":
    main()
