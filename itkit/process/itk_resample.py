import os, pdb, argparse, json, traceback
from tqdm import tqdm
from collections.abc import Sequence
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk

from itkit.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size, sitk_resample_to_image



def _list_medical_files(root: str, recursive: bool) -> dict[str, str]:
    """List medical image files under a root directory and return a mapping
    from relative path without extension (key) to absolute file path (value).

    The key preserves subdirectories but strips known extensions, so that
    files with different extensions (e.g., .nii.gz vs .mha) can be matched.
    """
    exts = (".mha", ".nii.gz", ".nii", ".mhd")

    def strip_ext(rel_path: str) -> str:
        if rel_path.endswith(".nii.gz"):
            return rel_path[:-7]
        base, ext = os.path.splitext(rel_path)
        return base

    mapping: dict[str, str] = {}
    if recursive:
        for r, _dirs, files in os.walk(root):
            for f in files:
                fp = os.path.join(r, f)
                # ensure extension match (case-sensitive per common dataset layout)
                if f.endswith(exts):
                    rel = os.path.relpath(fp, root)
                    key = strip_ext(rel)
                    mapping[key] = fp
    else:
        for f in os.listdir(root):
            if f.endswith(exts):
                fp = os.path.join(root, f)
                key = f[:-7] if f.endswith(".nii.gz") else os.path.splitext(f)[0]
                mapping[key] = fp
    return mapping


def _dest_path_with_mha(dest_root: str, rel_key: str) -> str:
    """Compose destination absolute path using rel_key (relative path without extension),
    normalizing extension to .mha and preserving subdirectories.
    """
    out_path = os.path.join(dest_root, rel_key + ".mha")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def resample_dataset_task(
    source_folder: str,
    dest_folder: str,
    target_spacing: Sequence[float],
    target_size: Sequence[int],
    recursive: bool = False,
    mp: bool = False,
    workers: int | None = None,
    target_folder: str | None = None,
):
    """Resample a dataset organized as:
    source_folder/
      image/
      label/

    Only samples present in both image and label are processed (intersection by filename without extension).
    Outputs are written to dest_folder/image and dest_folder/label respectively.
    """
    img_src = os.path.join(source_folder, "image")
    lbl_src = os.path.join(source_folder, "label")
    if not (os.path.isdir(img_src) and os.path.isdir(lbl_src)):
        raise ValueError(f"Dataset mode requires 'image' and 'label' subfolders under: {source_folder}")

    img_dst = os.path.join(dest_folder, "image")
    lbl_dst = os.path.join(dest_folder, "label")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    # Build source mappings (rel_key -> abs path)
    img_map = _list_medical_files(img_src, recursive)
    lbl_map = _list_medical_files(lbl_src, recursive)
    inter_keys = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    if not inter_keys:
        tqdm.write("No intersecting image/label files found to process.")
        return

    # Build target mappings when provided
    if target_folder:
        tgt_img_root = os.path.join(target_folder, "image")
        tgt_lbl_root = os.path.join(target_folder, "label")
        if not (os.path.isdir(tgt_img_root) and os.path.isdir(tgt_lbl_root)):
            raise ValueError("--target-folder in dataset mode must contain 'image' and 'label' subfolders.")
        tgt_img_map = _list_medical_files(tgt_img_root, recursive)
        tgt_lbl_map = _list_medical_files(tgt_lbl_root, recursive)
    else:
        tgt_img_map = {}
        tgt_lbl_map = {}

    # Build task list for both fields
    tasks: list[tuple] = []
    for key in inter_keys:
        # image task
        img_src_path = img_map[key]
        img_dst_path = _dest_path_with_mha(img_dst, key)
        img_tgt_path = tgt_img_map.get(key)
        tasks.append((img_src_path, target_spacing, target_size, "image", img_dst_path, img_tgt_path))
        # label task
        lbl_src_path = lbl_map[key]
        lbl_dst_path = _dest_path_with_mha(lbl_dst, key)
        lbl_tgt_path = tgt_lbl_map.get(key)
        tasks.append((lbl_src_path, target_spacing, target_size, "label", lbl_dst_path, lbl_tgt_path))

    # Execute
    image_meta: dict = {}
    label_meta: dict = {}

    if mp:
        with (
            Pool(processes=workers) as pool,
            tqdm(total=len(tasks), desc="Resampling (dataset)", leave=True, dynamic_ncols=True) as pbar,
        ):
            for idx, (res, logs) in enumerate(pool.imap_unordered(func=resample_one_sample, iterable=tasks)):
                for log in logs:
                    tqdm.write(log)
                if res:
                    # Determine field from task tuple
                    _src, _sp, _sz, _field, _dst, _tgt = tasks[idx]
                    if _field == "image":
                        image_meta.update(res)
                    else:
                        label_meta.update(res)
                pbar.update()
    else:
        with tqdm(total=len(tasks), desc="Resampling (dataset)", leave=True, dynamic_ncols=True) as pbar:
            for task_args in tasks:
                res, logs = resample_one_sample(task_args)
                for log in logs:
                    tqdm.write(log)
                if res:
                    (name, meta), = res.items()
                    # Assign based on whether name exists under image or label source sets
                    if name in [os.path.basename(p) for p in img_map.values()]:
                        image_meta.update(res)
                    elif name in [os.path.basename(p) for p in lbl_map.values()]:
                        label_meta.update(res)
                    else:
                        image_meta.update(res)
                pbar.update()

    # Save metadata under each subfolder
    try:
        with open(os.path.join(img_dst, "series_meta.json"), "w") as f:
            json.dump(image_meta, f, indent=4)
    except Exception as e:
        tqdm.write(f"Warning: Could not save image series meta file: {e}")
    try:
        with open(os.path.join(lbl_dst, "series_meta.json"), "w") as f:
            json.dump(label_meta, f, indent=4)
    except Exception as e:
        tqdm.write(f"Warning: Could not save label series meta file: {e}")


def resample_one_sample(args):
    """
    Resample a single sample image using spacing/size rules or a target reference image.
    
    Args `tuple`: (image_itk_path, target_spacing, target_size, field, output_path, target_image_path)
    
    Returns metadata `dict` or `None` if skipped.
    """
    # Unpack arguments
    logs = []
    image_itk_path, target_spacing, target_size, field, output_path, target_image_path = args
    img_dim = 3

    # Check if output already exists
    if os.path.exists(output_path):
        itk_name = os.path.basename(image_itk_path)
        logs.append(f"Skipping {itk_name}, output exists.")
        return None, logs

    # Read image
    try:
        image_itk = sitk.ReadImage(image_itk_path)
    except Exception as e:
        traceback.print_exc()
        logs.append(f"Error reading {image_itk_path}: {e}")
        return None, logs

    # If target image specified: resample to it; otherwise use spacing/size rules
    if target_image_path:
        target_image = sitk.ReadImage(target_image_path)
        image_resampled = sitk_resample_to_image(image_itk, target_image, field)
    else:
        # Spacing/size resampling logic
        # --- Stage 1: Spacing resample ---
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

        # --- Stage 2: Size resample ---
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

        # --- Stage 3: Orientation adjustment ---
        image_resampled = sitk.DICOMOrient(image_resampled, 'LPI')
        
        logs.append(
            f"Resampling completed for {os.path.basename(image_itk_path)}. "
            f"Output size {image_resampled.GetSize()[::-1]} | spacing {image_resampled.GetSpacing()[::-1]}."
        )

    # Write output
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(image_resampled, output_path, useCompression=True)
    except Exception as e:
        traceback.print_exc()
        logs.append(f"Error writing {output_path}: {e}")
        return None, logs

    # Get final spacing/size and return metadata
    final_spacing = image_resampled.GetSpacing()[::-1]
    final_size = image_resampled.GetSize()[::-1]
    # Get final origin
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
    # Collect all image files and relative paths
    image_paths = []
    output_paths = []
    rel_paths = []
    
    if recursive:
        # Recursive mode: traverse all subdirectories
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith((".mha", ".nii", ".nii.gz", ".mhd")):
                    # Keep directory structure
                    source_file = os.path.join(root, file)
                    rel_path = os.path.relpath(source_file, source_folder)
                    output_file = os.path.join(dest_folder, rel_path)
                    # Normalize output extension to .mha
                    output_file = output_file.replace(".nii.gz", ".mha").replace(".nii", ".mha").replace(".mhd", ".mha")
                    
                    image_paths.append(source_file)
                    output_paths.append(output_file)
                    rel_paths.append(rel_path)
    else:
        # Non-recursive mode: top-level only
        for file in os.listdir(source_folder):
            if file.endswith((".mha", ".nii", ".nii.gz", ".mhd")):
                # Normalize output extension to .mha
                source_file = os.path.join(source_folder, file)
                output_file = os.path.join(dest_folder, file)
                output_file = output_file.replace(".nii.gz", ".mha").replace(".nii", ".mha").replace(".mhd", ".mha")
                
                image_paths.append(source_file)
                output_paths.append(output_file)
                rel_paths.append(file)
    
    if not image_paths:
        tqdm.write("No image files found to process.")
        return
    
    # Build corresponding reference image path list
    if target_folder:
        target_paths = [os.path.join(target_folder, rel) for rel in rel_paths]
    else:
        target_paths = [None] * len(image_paths)
    # Build task list
    task_list = [
        (image_paths[i], target_spacing, target_size, field, output_paths[i], target_paths[i])
        for i in range(len(image_paths))
    ]

    # Collect per-sample metadata
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
    
    # Save per-sample spacing/size metadata to JSON
    meta_path = os.path.join(dest_folder, "series_meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(series_meta, f, indent=4)
    except Exception as e:
        tqdm.write(f"Warning: Could not save series meta file: {e}")


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


def main():
    args = parse_args()
    img_dim = 3 # Assume processing 3D images

    # --- Parameter conversion and validation ---
    try:
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
            if len(target_spacing) != img_dim:
                raise ValueError(f"--spacing must have {img_dim} values (received {len(target_spacing)})")
            if len(target_size) != img_dim:
                 raise ValueError(f"--size must have {img_dim} values (received {len(target_size)})")

            # Validate per-dimension exclusivity
            for i in range(img_dim):
                if target_spacing[i] != -1 and target_size[i] != -1:
                    raise ValueError(f"Dimension {i} cannot specify both spacing and size.")

            # Ensure at least one resampling rule is specified
            if all(s == -1 for s in target_spacing) and all(sz == -1 for sz in target_size):
                tqdm.write("Warning: No spacing or size specified, skipping resampling.")
                return

        # Print configuration
        print(f"Resampling {args.source_folder} -> {args.dest_folder}")
        if target_specified:
            print(f"  Target Folder: {args.target_folder}")
        else:
            print(f"  Spacing: {target_spacing} | Size: {target_size}")
        print(f"  Mode: {args.mode} | Recursive: {args.recursive} | Multiprocessing: {args.mp} | Workers: {args.workers}")

    except ValueError as e:
        print(f"Error parsing arguments: {e}")
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

    # Execute
    if args.mode == "dataset":
        resample_dataset_task(
            args.source_folder,
            args.dest_folder,
            target_spacing,
            target_size,
            args.recursive,
            args.mp,
            args.workers,
            args.target_folder,
        )
    else:
        # single folder mode uses original implementation requiring field
        resample_task(
            args.source_folder,
            args.dest_folder,
            target_spacing,
            target_size,
            args.mode,
            args.recursive,
            args.mp,
            args.workers,
            args.target_folder,
        )
    print(f"Resampling completed. The resampled dataset is saved in {args.dest_folder}.")



if __name__ == '__main__':
    main()
