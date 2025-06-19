import os
import pdb
import argparse
import json
import traceback
from tqdm import tqdm
from collections.abc import Sequence
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size



def resample_one_sample(args):
    """
    Resample a single sample image based on dimension-wise spacing and size rules.

    Args:
        args (tuple): A tuple containing:
            image_itk_path (str): The file path of the input image.
            target_spacing (Sequence[float]): Target spacing per dimension (-1 to ignore).
            target_size (Sequence[int]): Target size per dimension (-1 to ignore).
            output_path (str): The output file path for the resampled image.

    Returns:
        A dict containing metadata about the resampled image, or None if the output file already exists.
    """
    image_itk_path, target_spacing, target_size, output_path = args
    img_dim = 3

    # 检查目标文件是否已存在
    if os.path.exists(output_path):
        itk_name = os.path.basename(image_itk_path)
        tqdm.write(f"Skipping {itk_name}, output exists.")
        return None

    # 读取
    try:
        image_itk = sitk.ReadImage(image_itk_path)
    except Exception as e:
        traceback.print_exc()
        tqdm.write(f"Error reading {image_itk_path}: {e}")
        return None

    # --- 阶段一：Spacing 重采样 ---
    orig_spacing = image_itk.GetSpacing()[::-1]
    effective_spacing = list(orig_spacing)
    needs_spacing_resample = False
    for i in range(img_dim):
        if target_spacing[i] != -1:
            effective_spacing[i] = target_spacing[i]
            needs_spacing_resample = True

    image_after_spacing = image_itk

    if needs_spacing_resample and not np.allclose(effective_spacing, orig_spacing):
        itk_name = os.path.basename(image_itk_path)
        tqdm.write(f"Resampling {itk_name} to spacing {effective_spacing}...")
        image_after_spacing = sitk_resample_to_spacing(image_itk, effective_spacing, "image")

    # --- 阶段二：Size 重采样 ---
    current_size = image_after_spacing.GetSize()[::-1]
    effective_size = list(current_size)
    needs_size_resample = False
    for i in range(img_dim):
        if target_size[i] != -1:
            effective_size[i] = target_size[i]
            needs_size_resample = True

    image_resampled = image_after_spacing

    if needs_size_resample and effective_size != list(current_size):
        itk_name = os.path.basename(image_itk_path)
        tqdm.write(f"Resampling {itk_name} to size {effective_size}...")
        image_resampled = sitk_resample_to_size(image_after_spacing, effective_size, "image")

    # --- 阶段三：方向重采样 ---
    image_resampled = sitk.DICOMOrient(image_resampled, 'LPI')

    # 写入
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(image_resampled, output_path, useCompression=True)
    except Exception as e:
        traceback.print_exc()
        tqdm.write(f"Error writing {output_path}: {e}")
        return None

    # 获取实际spacing和size并返回元数据
    final_spacing = image_resampled.GetSpacing()[::-1]
    final_size = image_resampled.GetSize()[::-1]
    # 获取实际origin
    final_origin = image_resampled.GetOrigin()[::-1]
    itk_name = os.path.basename(image_itk_path)
    return {itk_name: {"spacing": final_spacing, "size": final_size, "origin": final_origin}}


def resample_dataset(
    source_folder: str,
    dest_folder: str,
    target_spacing: Sequence[float],
    target_size: Sequence[int],
    recursive: bool = False,
    mp: bool = False,
    workers: int|None = None,
):
    """
    Resample a dataset with dimension-wise spacing/size rules.

    Args:
        source_folder (str): The source folder containing .mha files.
        dest_folder (str): The destination folder for resampled files.
        target_spacing (Sequence[float]): Target spacing per dimension (-1 to ignore).
        target_size (Sequence[int]): Target size per dimension (-1 to ignore).
        recursive (bool): Whether to recursively process subdirectories.
        mp (bool): Whether to use multiprocessing.
        workers (int | None): Number of workers for multiprocessing.
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    # 收集所有mha文件
    image_paths = []
    output_paths = []
    
    if recursive:
        # 递归模式：遍历所有子目录
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith((".mha", ".nii", ".nii.gz", ".mhd")):
                    source_file = os.path.join(root, file)
                    # 保持相同的目录结构
                    rel_path = os.path.relpath(source_file, source_folder)
                    output_file = os.path.join(dest_folder, rel_path)
                    # 统一输出为.mha格式
                    output_file = output_file.replace(".nii.gz", ".mha").replace(".nii", ".mha").replace(".mhd", ".mha")
                    
                    image_paths.append(source_file)
                    output_paths.append(output_file)
    else:
        # 非递归模式：只处理顶层目录
        for file in os.listdir(source_folder):
            if file.endswith((".mha", ".nii", ".nii.gz", ".mhd")):
                source_file = os.path.join(source_folder, file)
                output_file = os.path.join(dest_folder, file)
                # 统一输出为.mha格式
                output_file = output_file.replace(".nii.gz", ".mha").replace(".nii", ".mha").replace(".mhd", ".mha")
                
                image_paths.append(source_file)
                output_paths.append(output_file)
    
    if not image_paths:
        tqdm.write("No image files found to process.")
        return
    
    # 生成任务列表
    task_list = [
        (image_paths[i], target_spacing, target_size, output_paths[i])
        for i in range(len(image_paths))
    ]

    # 收集每个样本的meta信息
    series_meta = dict()
    if mp:
        with (
            Pool(processes=workers) as pool,
            tqdm(
                total=len(image_paths),
                desc="Resampling",
                leave=True,
                dynamic_ncols=True,
            ) as pbar,
        ):
            result_fetcher = pool.imap_unordered(func=resample_one_sample, iterable=task_list)
            for res in result_fetcher:
                if res:
                    series_meta.update(res)
                pbar.update()
    else:
        with tqdm(
            total=len(image_paths),
            desc="Resampling",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for task_args in task_list:
                res = resample_one_sample(task_args)
                if res:
                    series_meta.update(res)
                pbar.update()
    
    # 保存每个样本的实际spacing和size到JSON
    meta_path = os.path.join(dest_folder, "series_meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(series_meta, f, indent=4)
    except Exception as e:
        tqdm.write(f"Warning: Could not save series meta file: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample a dataset with dimension-wise spacing/size rules.")
    parser.add_argument("source_folder", type=str, help="The source folder containing .mha files.")
    parser.add_argument("dest_folder", type=str, help="The destination folder for resampled files.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively process subdirectories.")
    parser.add_argument("--mp", action="store_true", help="Whether to use multiprocessing.")
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")

    # 允许同时指定，用 -1 表示不指定
    # 类型为 str 先接收，方便处理 -1
    parser.add_argument("--spacing", type=str, nargs='+', default=["-1", "-1", "-1"],
                        help="Target spacing (ZYX order). Use -1 for dimensions to ignore spacing rule. e.g., 1.5 -1 1.5")
    parser.add_argument("--size", type=str, nargs='+', default=["-1", "-1", "-1"],
                        help="Target size (ZYX order). Use -1 for dimensions to ignore size rule. e.g., -1 256 256")

    return parser.parse_args()


def main():
    args = parse_args()
    img_dim = 3 # 假设处理3D图像

    # --- 参数转换和验证 ---
    try:
        # 转换 spacing 为 float, size 为 int
        target_spacing = [float(s) for s in args.spacing]
        target_size = [int(s) for s in args.size]

        # 检查列表长度是否匹配维度
        if len(target_spacing) != img_dim:
            raise ValueError(f"--spacing must have {img_dim} values (received {len(target_spacing)})")
        if len(target_size) != img_dim:
             raise ValueError(f"--size must have {img_dim} values (received {len(target_size)})")

        # 验证每个维度的互斥性
        for i in range(img_dim):
            if target_spacing[i] != -1 and target_size[i] != -1:
                raise ValueError(f"Dimension {i} (ZYX order) cannot have both spacing ({target_spacing[i]}) and size ({target_size[i]}) specified. Use -1 for one of them.")

        # 检查是否至少指定了一个重采样操作
        if all(s == -1 for s in target_spacing) and all(sz == -1 for sz in target_size):
             tqdm.write("Warning: No resampling specified (all spacing and size values are -1).")
             return

    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        return

    print(f"Resampling {args.source_folder} to {args.dest_folder}")
    print(f"  Target Spacing (ZYX): {target_spacing}")
    print(f"  Target Size (ZYX): {target_size}")
    print(f"  Recursive mode: {args.recursive}")

    # 保存配置信息
    config_data = vars(args)
    config_data['target_spacing_validated'] = target_spacing
    config_data['target_size_validated'] = target_size
    try:
        os.makedirs(args.dest_folder, exist_ok=True)
        with open(os.path.join(args.dest_folder, "resample_configs.json"), "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")

    # 执行
    resample_dataset(
        args.source_folder,
        args.dest_folder,
        target_spacing,
        target_size,
        args.recursive,
        args.mp,
        args.workers,
    )
    print(f"Resampling completed. The resampled dataset is saved in {args.dest_folder}.")



if __name__ == '__main__':
    main()