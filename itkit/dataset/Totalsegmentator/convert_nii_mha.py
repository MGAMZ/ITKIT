import argparse
import multiprocessing
import os
from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from itkit.dataset.Totalsegmentator.meta import (
    CLASS_INDEX_MAP,
    CLASS_MERGE,
    SUBSETS,
    generate_reduced_class_map_and_label_map,
)
from itkit.io.sitk_toolkit import (
    merge_masks,
    nii_to_sitk,
    sitk_resample_to_size,
    sitk_resample_to_spacing,
)


def extract_subset_segmentations(
    corresponding_itk_image: sitk.Image,
    all_nii_files: list[str],
    subset_name: str
) -> sitk.Image:
    """
    从所有分割文件中提取一个子集，并根据子集定义重新映射标签ID。
    """
    subset_classes = SUBSETS[subset_name]

    class_to_files = {
        os.path.basename(file_path)[:-7]: file_path
        for file_path in all_nii_files
    }

    reference_array = sitk.GetArrayFromImage(corresponding_itk_image)
    merged_array = np.zeros_like(reference_array, dtype=np.uint8)

    for new_id, class_name in enumerate(subset_classes):
        if class_name == 'background':
            continue

        if class_name in class_to_files:
            nii_path = class_to_files[class_name]
            try:
                mask_itk = nii_to_sitk(nii_path, 'label')
                mask_array = sitk.GetArrayFromImage(mask_itk)
                merged_array[mask_array > 0] = new_id
            except Exception as e:
                print(f"Warning: Failed to process {nii_path}: {e}")

    merged_itk = sitk.GetImageFromArray(merged_array)
    merged_itk.CopyInformation(corresponding_itk_image)
    return merged_itk


def parse_one_case_segmentations(corresponding_itk_image:sitk.Image,
                                 case_path: str,
                                 subset: str | None = None,
                                 merge_rule: str | None = None):
    segmentation_path = os.path.join(case_path, 'segmentations')

    # 获取所有nii.gz文件
    all_nii_files = [
        os.path.join(segmentation_path, file)
        for file in os.listdir(segmentation_path)
        if file.endswith('.nii.gz')
    ]

    if subset is not None and merge_rule is not None:
        raise ValueError("Cannot specify both subset and merge_rule. Please use only one.")

    if subset is not None:
        # 如果指定了子集，使用新的提取函数
        return extract_subset_segmentations(corresponding_itk_image, all_nii_files, subset)

    elif merge_rule is not None:
        # 如果指定了merge规则，按照merge规则处理
        if merge_rule not in CLASS_MERGE:
            raise ValueError(f"Merge rule '{merge_rule}' not found in CLASS_MERGE. Available merge rules: {list(CLASS_MERGE.keys())}")

        # 获取merge规则对应的reduced_class_map和label_map
        merge_config = CLASS_MERGE[merge_rule]
        reduced_class_map, label_map = generate_reduced_class_map_and_label_map(merge_config)

        # 建立类名到文件路径的映射
        class_to_files = {}
        for file_path in all_nii_files:
            class_name = os.path.basename(file_path)[:-7]  # 去除.nii.gz后缀
            if class_name in CLASS_INDEX_MAP:  # 确保类名在原始映射中
                class_to_files[class_name] = file_path

        # 收集所有相关文件，按照原始CLASS_INDEX_MAP的顺序排序
        relevant_files = []
        for class_name in sorted(CLASS_INDEX_MAP.keys(), key=lambda x: CLASS_INDEX_MAP[x]):
            if class_name != 'background' and class_name in class_to_files:
                relevant_files.append(class_to_files[class_name])

        # 特殊处理：对于merge规则，我们需要先合并所有mask然后重新映射标签
        if not relevant_files:
            # 如果没有找到任何文件，创建一个空的掩码
            reference_array = sitk.GetArrayFromImage(corresponding_itk_image)
            empty_mask = np.zeros_like(reference_array, dtype=np.uint8)
            merged_itk = sitk.GetImageFromArray(empty_mask)
            merged_itk.CopyInformation(corresponding_itk_image)
            return merged_itk
        else:
            # 先合并所有mask，得到原始标签
            merged_itk_original = merge_masks(relevant_files)
            original_array = sitk.GetArrayFromImage(merged_itk_original)

            # 应用label_map重新映射标签
            merged_array = np.zeros_like(original_array, dtype=np.uint8)
            for old_label, new_label in label_map.items():
                merged_array[original_array == old_label] = new_label

            merged_itk = sitk.GetImageFromArray(merged_array)
            merged_itk.CopyInformation(merged_itk_original)
            return merged_itk

    else:
        # 如果未指定子集，处理所有类，按照原始CLASS_INDEX_MAP排序
        idx_sorted_paths = sorted(
            all_nii_files,
            key=lambda x: CLASS_INDEX_MAP[os.path.basename(x)[:-7]]
        )

    # 使用merge_masks合并
    if not idx_sorted_paths:
        # 如果没有找到任何文件，创建一个空的掩码
        reference_array = sitk.GetArrayFromImage(corresponding_itk_image)
        empty_mask = np.zeros_like(reference_array, dtype=np.uint8)
        merged_itk = sitk.GetImageFromArray(empty_mask)

    else:
        merged_itk = merge_masks(idx_sorted_paths)

    merged_itk.CopyInformation(corresponding_itk_image)
    return merged_itk


def convert_one_case(args):
    series_input_folder, series_output_folder, spacing, size, subset, merge_rule = args
    sample_id = os.path.basename(series_input_folder)
    input_image_nii_path = os.path.join(series_input_folder, 'ct.nii.gz')
    output_image_mha_path = os.path.join(series_output_folder, 'image', f'{sample_id}.mha')
    output_anno_mha_path = os.path.join(series_output_folder, 'label', f'{sample_id}.mha')
    os.makedirs(os.path.join(series_output_folder, 'image'), exist_ok=True)
    os.makedirs(os.path.join(series_output_folder, 'label'), exist_ok=True)
    if os.path.exists(output_image_mha_path) and os.path.exists(output_anno_mha_path):
        return

    input_image_mha = nii_to_sitk(input_image_nii_path, "image")
    merged_itk = parse_one_case_segmentations(
        input_image_mha,
        series_input_folder,
        subset=subset,
        merge_rule=merge_rule
    )

    if spacing is not None:
        assert size is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_spacing(input_image_mha, spacing, 'image')
        merged_itk = sitk_resample_to_spacing(merged_itk, spacing, 'label')
    if size is not None:
        assert spacing is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_size(input_image_mha, size, 'image')
        merged_itk = sitk_resample_to_size(merged_itk, size, 'label')

    input_image_mha = sitk.DICOMOrient(input_image_mha, 'LPI')
    merged_itk = sitk.DICOMOrient(merged_itk, 'LPI')
    sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)
    sitk.WriteImage(merged_itk, output_anno_mha_path, useCompression=True)


def convert_and_save_nii_to_mha(input_dir:str,
                                output_dir:str,
                                use_mp:bool,
                                workers:int|None=None,
                                spacing:Sequence[float|int]|None=None,
                                size:Sequence[float|int]|None=None,
                                subset:str|None=None,
                                merge_rule:str|None=None):
    task_list = []
    for series_name in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, series_name)):
            series_input_folder = os.path.join(input_dir, series_name)
            task_list.append((series_input_folder, output_dir, spacing, size, subset, merge_rule))

    if use_mp:
        with multiprocessing.Pool(workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(convert_one_case, task_list),
                total=len(task_list),
                desc="nii2mha",
                leave=False,
                dynamic_ncols=True
            ):
                pass
    else:
        for args in tqdm(task_list,
                         leave=False,
                         dynamic_ncols=True,
                         desc="nii2mha"
        ):
            convert_one_case(args)


def main():
    parser = argparse.ArgumentParser(description="Convert all NIfTI files in a directory to MHA format.")
    parser.add_argument('input_dir', type=str, help="Containing NIfTI files.")
    parser.add_argument('output_dir', type=str, help="Save MHA files.")
    parser.add_argument('--mp', action='store_true', help="Use multiprocessing.")
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help="Number of workers.")
    parser.add_argument('--spacing', type=float, nargs=3, default=None, help="Resample to this spacing.")
    parser.add_argument('--size', type=int, nargs=3, default=None, help="Crop to this size.")
    parser.add_argument('--subset', type=str, default=None, help="Use only classes from this subset (e.g., 'bones').")
    parser.add_argument('--merge', type=str, default=None, choices=list(CLASS_MERGE.keys()),
                       help=f"Merge classes according to the specified rule. Available options: {list(CLASS_MERGE.keys())}")
    args = parser.parse_args()

    # 验证subset和merge不能同时指定
    if args.subset is not None and args.merge is not None:
        parser.error("Cannot specify both --subset and --merge. Please use only one.")

    convert_and_save_nii_to_mha(
        args.input_dir,
        args.output_dir,
        args.mp,
        args.workers,
        args.spacing,
        args.size,
        args.subset,
        args.merge
    )


if __name__ == "__main__":
    main()
