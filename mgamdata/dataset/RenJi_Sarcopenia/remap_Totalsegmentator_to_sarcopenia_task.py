import os
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import multiprocessing


SARCOPENIA_TASK_REMAP = {
    'background': 0,
    'clavicula_left': 1, # 左锁骨
    'clavicula_right': 1, # 右锁骨
    'costal_cartilages': 1, # 肋软骨
    'femur_left': 1, # 左股骨
    'femur_right': 1, # 右股骨
    'hip_left': 1, # 左髋
    'hip_right': 1, # 右髋
    'humerus_left': 1, # 左肱骨
    'humerus_right': 1, # 右肱骨
    'rib_left_1': 1, # 左第1肋骨
    'rib_left_10': 1, # 左第10肋骨
    'rib_left_11': 1, # 左第11肋骨
    'rib_left_12': 1, # 左第12肋骨
    'rib_left_2': 1, # 左第2肋骨
    'rib_left_3': 1, # 左第3肋骨
    'rib_left_4': 1, # 左第4肋骨
    'rib_left_5': 1, # 左第5肋骨
    'rib_left_6': 1, # 左第6肋骨
    'rib_left_7': 1, # 左第7肋骨
    'rib_left_8': 1, # 左第8肋骨
    'rib_left_9': 1, # 左第9肋骨
    'rib_right_1': 1, # 右第1肋骨
    'rib_right_10': 1, # 右第10肋骨
    'rib_right_11': 1, # 右第11肋骨
    'rib_right_12': 1, # 右第12肋骨
    'rib_right_2': 1, # 右第2肋骨
    'rib_right_3': 1, # 右第3肋骨
    'rib_right_4': 1, # 右第4肋骨
    'rib_right_5': 1, # 右第5肋骨
    'rib_right_6': 1, # 右第6肋骨
    'rib_right_7': 1, # 右第7肋骨
    'rib_right_8': 1, # 右第8肋骨
    'rib_right_9': 1, # 右第9肋骨
    'sacrum': 1, # 骶骨
    'scapula_left': 1, # 左肩胛骨
    'scapula_right': 1, # 右肩胛骨
    'skull': 1, # 颅骨
    'sternum': 1, # 胸骨
    'vertebrae_C1': 1, # 第1颈椎
    'vertebrae_C2': 1, # 第2颈椎
    'vertebrae_C3': 1, # 第3颈椎
    'vertebrae_C4': 1, # 第4颈椎
    'vertebrae_C5': 1, # 第5颈椎
    'vertebrae_C6': 1, # 第6颈椎
    'vertebrae_C7': 1, # 第7颈椎
    'vertebrae_L1': 1, # 第1腰椎
    'vertebrae_L2': 1, # 第2腰椎
    'vertebrae_L3': 1, # 第3腰椎
    'vertebrae_L4': 1, # 第4腰椎
    'vertebrae_L5': 1, # 第5腰椎
    'vertebrae_S1': 1, # 第1骶椎
    'vertebrae_T1': 1, # 第1胸椎
    'vertebrae_T10': 1, # 第10胸椎
    'vertebrae_T11': 1, # 第11胸椎
    'vertebrae_T12': 1, # 第12胸椎
    'vertebrae_T2': 1, # 第2胸椎
    'vertebrae_T3': 1, # 第3胸椎
    'vertebrae_T4': 1, # 第4胸椎
    'vertebrae_T5': 1, # 第5胸椎
    'vertebrae_T6': 1, # 第6胸椎
    'vertebrae_T7': 1, # 第7胸椎
    'vertebrae_T8': 1, # 第8胸椎
    'vertebrae_T9': 1, # 第9胸椎
    'autochthon_left': 2, # 左侧竖脊肌
    'autochthon_right': 2, # 右侧竖脊肌
}


CLASS_MAP = {
    0: 'background',
    1: 'bones', # 骨
    2: 'erector' # 竖脊肌
}


def remap_one_case(args):
    case_input_dir, output_dir = args
    case_id = os.path.basename(case_input_dir)
    output_mask_path = os.path.join(output_dir, f'{case_id}.mha')
    if os.path.exists(output_mask_path):
        return

    reference_image_path = os.path.join(case_input_dir, 'ct.nii.gz')
    if not os.path.exists(reference_image_path):
        print(f"Warning: Reference image not found for case {case_id}, skipping.")
        return
    
    reference_itk = sitk.ReadImage(reference_image_path)
    reference_array = sitk.GetArrayFromImage(reference_itk)
    
    merged_mask_array = np.zeros_like(reference_array, dtype=np.uint8)
    
    segmentations_dir = os.path.join(case_input_dir, 'segmentations')
    if not os.path.isdir(segmentations_dir):
        print(f"Warning: Segmentations directory not found for case {case_id}, skipping.")
        return

    for class_name, target_id in SARCOPENIA_TASK_REMAP.items():
        if class_name == 'background':
            continue
        
        nii_path = os.path.join(segmentations_dir, f'{class_name}.nii.gz')
        if os.path.exists(nii_path):
            try:
                mask_itk = sitk.ReadImage(nii_path)
                mask_array = sitk.GetArrayFromImage(mask_itk)
                merged_mask_array[mask_array > 0] = target_id
            except Exception as e:
                print(f"Error processing {nii_path} for case {case_id}: {e}")

    merged_itk = sitk.GetImageFromArray(merged_mask_array)
    merged_itk.CopyInformation(reference_itk)
    merged_itk = sitk.DICOMOrient(merged_itk, 'LPI')

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    sitk.WriteImage(merged_itk, output_mask_path, useCompression=True)


def process_dataset(input_dir: str, output_dir: str, use_mp: bool, workers: int | None = None):
    task_list = []
    for case_name in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case_name)
        if os.path.isdir(case_path):
            task_list.append((case_path, output_dir))

    if use_mp:
        with multiprocessing.Pool(workers) as pool:
            list(tqdm(pool.imap(remap_one_case, task_list), total=len(task_list), desc="Remapping Sarcopenia Masks", dynamic_ncols=True))
    else:
        for args in tqdm(task_list, desc="Remapping Sarcopenia Masks", dynamic_ncols=True):
            remap_one_case(args)


def main():
    parser = argparse.ArgumentParser(
        description="Remap and merge segmentations from the TotalSegmentator dataset for the Sarcopenia task."
    )
    parser.add_argument('input_dir', type=str, 
                        help="Input directory containing the raw TotalSegmentator dataset structure (e.g., .../s0001/ct.nii.gz, .../s0001/segmentations/).")
    parser.add_argument('output_dir', type=str, 
                        help="Output directory to save the merged MHA mask files.")
    parser.add_argument('--mp', action='store_true', 
                        help="Use multiprocessing to speed up processing.")
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), 
                        help="Number of worker processes to use with multiprocessing.")
    
    args = parser.parse_args()
    
    process_dataset(
        args.input_dir,
        args.output_dir,
        args.mp,
        args.workers
    )


if __name__ == "__main__":
    main()


