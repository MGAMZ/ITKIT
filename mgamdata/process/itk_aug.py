import os
import argparse
import random
from collections.abc import Sequence
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
from mgamdata.io.sitk_toolkit import PIXEL_TYPE, INTERPOLATOR


def random_3d_rotate(image:sitk.Image, label:sitk.Image, angle_ranges:Sequence[float]):
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


def process_sample(args):
    """处理单个样本的增强"""
    filename, img_folder, lbl_folder, out_img_folder, out_lbl_folder, num, random_rots = args
    
    # 读取图像和标签
    img_path = os.path.join(img_folder, filename)
    lbl_path = os.path.join(lbl_folder, filename)
    
    # 使用SimpleITK读取图像
    image = sitk.ReadImage(img_path)
    label = sitk.ReadImage(lbl_path)
    
    basename = os.path.splitext(filename)[0]
    
    # 创建增强
    for i in range(num):
        # 应用随机旋转
        rotated_image, rotated_label = random_3d_rotate(image, label, random_rots)
        
        # 保存增强文件
        if out_img_folder:
            aug_img_path = os.path.join(out_img_folder, f"{basename}_{i}.mha")
            sitk.WriteImage(rotated_image, aug_img_path)
        
        if out_lbl_folder:
            aug_lbl_path = os.path.join(out_lbl_folder, f"{basename}_{i}.mha")
            sitk.WriteImage(rotated_label, aug_lbl_path)
    
    return filename


def parse_args():
    parser = argparse.ArgumentParser(description='ITK基于3D数据增强')
    parser.add_argument('img_folder', type=str, help='包含图像MHA文件的文件夹')
    parser.add_argument('lbl_folder', type=str, help='包含标签MHA文件的文件夹')
    parser.add_argument('-oimg', '--out-img-folder', type=str, default=None, help='增强后图像的输出文件夹 (可选)')
    parser.add_argument('-olbl', '--out-lbl-folder', type=str, default=None, help='增强后标签的输出文件夹 (可选)')
    parser.add_argument('-n', '--num', type=int, default=1, help='每个样本的增强数量')
    parser.add_argument('--mp', action='store_true', help='启用多进程处理')
    parser.add_argument('--random-rot', type=int, nargs=3, default=None, help='最大随机旋转角度（度）')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_img_folder:
        os.makedirs(args.out_img_folder, exist_ok=True)
    if args.out_lbl_folder:
        os.makedirs(args.out_lbl_folder, exist_ok=True)
    
    # 获取公共文件名
    img_files = set(f for f in os.listdir(args.img_folder) if f.endswith('.mha'))
    lbl_files = set(f for f in os.listdir(args.lbl_folder) if f.endswith('.mha'))
    common_files = list(img_files.intersection(lbl_files))
    print(f"找到 {len(common_files)} 对匹配的图像-标签对")
    
    # 准备任务
    process_args = [
        (
            filename,
            args.img_folder,
            args.lbl_folder,
            args.out_img_folder,
            args.out_lbl_folder,
            args.num,
            args.random_rot
        )
        for filename in common_files
    ]
    
    if args.mp:
        with Pool() as pool:
            list(tqdm(pool.imap_unordered(process_sample, process_args),
                      total=len(common_files),
                      desc="Augmenting",
                      dynamic_ncols=True))
    else:
        for arg in tqdm(process_args,
                        total=len(common_files),
                        desc="Augmenting",
                        dynamic_ncols=True):
            process_sample(arg)
    
    print("数据增强完成")


if __name__ == "__main__":
    main()
