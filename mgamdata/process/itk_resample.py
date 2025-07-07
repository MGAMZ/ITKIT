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

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size, sitk_resample_to_image



def resample_one_sample(args):
    """
    Resample a single sample image using spacing/size rules or a target reference image.
    Args tuple: (image_itk_path, target_spacing, target_size, output_path, target_image_path, field)
    Returns metadata dict or None if skipped.
    """
    # 解包参数
    logs = []
    image_itk_path, target_spacing, target_size, field, output_path, target_image_path = args
    img_dim = 3

    # 检查目标文件是否已存在
    if os.path.exists(output_path):
        itk_name = os.path.basename(image_itk_path)
        logs.append(f"Skipping {itk_name}, output exists.")
        return None, logs

    # 读取
    try:
        image_itk = sitk.ReadImage(image_itk_path)
    except Exception as e:
        traceback.print_exc()
        logs.append(f"Error reading {image_itk_path}: {e}")
        return None, logs

    # 如果指定了目标图像，则使用目标图像重采样，否则使用spacing/size重采样
    if target_image_path:
        target_image = sitk.ReadImage(target_image_path)
        image_resampled = sitk_resample_to_image(image_itk, target_image, field)
    else:
        # spacing/size重采样逻辑
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
            image_after_spacing = sitk_resample_to_spacing(image_itk, effective_spacing, field)

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
            image_resampled = sitk_resample_to_size(image_after_spacing, effective_size, field)

        # --- 阶段三：方向重采样 ---
        image_resampled = sitk.DICOMOrient(image_resampled, 'LPI')
        
        logs.append(
            f"Resampling completed for {os.path.basename(image_itk_path)}. "
            f"Output size {image_resampled.GetSize()} | spacing {image_resampled.GetSpacing()}."
        )

    # 写入
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(image_resampled, output_path, useCompression=True)
    except Exception as e:
        traceback.print_exc()
        logs.append(f"Error writing {output_path}: {e}")
        return None, logs

    # 获取实际spacing和size并返回元数据
    final_spacing = image_resampled.GetSpacing()[::-1]
    final_size = image_resampled.GetSize()[::-1]
    # 获取实际origin
    final_origin = image_resampled.GetOrigin()[::-1]
    itk_name = os.path.basename(image_itk_path)
    return {itk_name: {"spacing": final_spacing, "size": final_size, "origin": final_origin}}, logs


def resample_task(
    source_folder: str,
    dest_folder: str,
    target_spacing: Sequence[float],
    target_size: Sequence[int],
    field: str,
    recursive: bool = False,
    mp: bool = False,
    workers: int|None = None,
    target_folder: str|None = None,
):
    """
    Resample a dataset with dimension-wise spacing/size rules or target-folder reference images.

    Args:
        source_folder (str): The source folder containing .mha files.
        dest_folder (str): The destination folder for resampled files.
        target_spacing (Sequence[float]): Target spacing per dimension (-1 to ignore).
        target_size (Sequence[int]): Target size per dimension (-1 to ignore).
        recursive (bool): Whether to recursively process subdirectories.
        mp (bool): Whether to use multiprocessing.
        workers (int | None): Number of workers for multiprocessing.
        target_folder (str | None): Folder containing reference images named same as source.
        field (str): Field type for resampling ('image' or 'label').
    """
    os.makedirs(dest_folder, exist_ok=True)
    # 收集所有mha文件及其相对路径
    image_paths = []
    output_paths = []
    rel_paths = []
    
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
                    rel_paths.append(rel_path)
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
                rel_paths.append(file)
    
    if not image_paths:
        tqdm.write("No image files found to process.")
        return
    
    # 构建对应的参考图像路径列表
    if target_folder:
        target_paths = [os.path.join(target_folder, rel) for rel in rel_paths]
    else:
        target_paths = [None] * len(image_paths)
    # 生成任务列表
    task_list = [
        (image_paths[i], target_spacing, target_size, field, output_paths[i], target_paths[i])
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
            for res, logs in result_fetcher:
                for log in logs:
                    tqdm.write(log)
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
                res, logs = resample_one_sample(task_args)
                for log in logs:
                    tqdm.write(log)
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
    parser = argparse.ArgumentParser(description="Resample a dataset with dimension-wise spacing/size rules or target image.")
    parser.add_argument("field", type=str, choices=["image", "label"], help="Field type for resampling. Required when --target is specified.")
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
    
    # target_folder 模式
    parser.add_argument("--target-folder", dest="target_folder", type=str, default=None,
                        help="Folder containing target reference images matching source names. Mutually exclusive with --spacing and --size.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    img_dim = 3 # 假设处理3D图像

    # --- 参数转换和验证 ---
    try:
        # 检查 target_folder 与 spacing/size 的互斥性
        target_specified = args.target_folder is not None
        spacing_specified = any(s != "-1" for s in args.spacing)
        size_specified = any(s != "-1" for s in args.size)
        
        if target_specified and (spacing_specified or size_specified):
            raise ValueError("--target-folder is mutually exclusive with --spacing and --size. Use either --target-folder or --spacing/--size, not both.")
        
        if target_specified:
            # 使用 target_folder 模式
            if not os.path.isdir(args.target_folder):
                raise ValueError(f"Target folder does not exist: {args.target_folder}")
            # 设置无效的 spacing/size
            target_spacing = [-1, -1, -1]
            target_size = [-1, -1, -1]
        else:
            # 使用spacing/size模式
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
                    raise ValueError(f"Dimension {i} cannot同时指定 spacing 和 size。")

            # 检查至少指定一种重采样
            if all(s == -1 for s in target_spacing) and all(sz == -1 for sz in target_size):
                tqdm.write("Warning: 未指定spacing或size，跳过重采样。")
                return

        # 打印配置信息
        print(f"Resampling {args.source_folder} -> {args.dest_folder}")
        if target_specified:
            print(f"  Target Folder: {args.target_folder} | Field: {args.field}")
        else:
            print(f"  Spacing: {target_spacing} | Size: {target_size} | Field: {args.field}")
        print(f"  Recursive: {args.recursive} | Multiprocessing: {args.mp} | Workers: {args.workers}")

    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        return

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
    resample_task(
        args.source_folder,
        args.dest_folder,
        target_spacing,
        target_size,
        args.field,
        args.recursive,
        args.mp,
        args.workers,
        args.target_folder,
    )
    print(f"Resampling completed. The resampled dataset is saved in {args.dest_folder}.")



if __name__ == '__main__':
    main()
