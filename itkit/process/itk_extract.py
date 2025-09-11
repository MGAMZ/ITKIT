import os, pdb, argparse, json, traceback
from tqdm import tqdm
from collections.abc import Sequence
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk


def extract_one_sample(args):
    """
    Extract and remap labels from a single sample image.
    
    Args `tuple`: (image_itk_path, label_mapping, output_path)
    
    Returns metadata `dict` or `None` if skipped.
    """
    # Unpack arguments
    logs = []
    image_itk_path, label_mapping, output_path = args

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

    # Convert to numpy array
    try:
        image_array = sitk.GetArrayFromImage(image_itk)
        # Ensure uint dtype
        if not np.issubdtype(image_array.dtype, np.unsignedinteger):
            image_array = image_array.astype(np.uint32)
    except Exception as e:
        traceback.print_exc()
        logs.append(f"Error converting image to array {image_itk_path}: {e}")
        return None, logs

    # Create output array with same shape, initialized with background (0)
    output_array = np.zeros_like(image_array, dtype=np.uint32)

    # Apply label mappings
    original_labels = set()
    extracted_labels = set()
    
    for source_label, target_label in label_mapping.items():
        mask = (image_array == source_label)
        if np.any(mask):
            output_array[mask] = target_label
            original_labels.add(int(source_label))
            extracted_labels.add(int(target_label))

    # Convert back to SimpleITK image
    try:
        output_itk = sitk.GetImageFromArray(output_array)
        # Copy metadata from original image
        output_itk.CopyInformation(image_itk)
    except Exception as e:
        traceback.print_exc()
        logs.append(f"Error converting array to image {image_itk_path}: {e}")
        return None, logs

    # Write output
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(output_itk, output_path, useCompression=True)
    except Exception as e:
        traceback.print_exc()
        logs.append(f"Error writing {output_path}: {e}")
        return None, logs

    itk_name = os.path.basename(image_itk_path)
    logs.append(
        f"Label extraction completed for {itk_name}. "
        f"Original labels: {sorted(original_labels)} -> Extracted labels: {sorted(extracted_labels)}."
    )

    # Return metadata
    return {
        itk_name: {
            "original_labels": sorted(original_labels),
            "extracted_labels": sorted(extracted_labels),
            "mapping": label_mapping,
            "output_shape": output_array.shape,
            "output_dtype": str(output_array.dtype)
        }
    }, logs


def extract_task(
    source_folder: str,
    dest_folder: str,
    label_mapping: dict,
    recursive: bool = False,
    mp: bool = False,
    workers: int | None = None,
):
    """
    Extract and remap labels from a dataset.

    Args:
        source_folder (str): The source folder containing .mha files.
        dest_folder (str): The destination folder for extracted files.
        label_mapping (dict): Mapping from source labels to target labels.
        recursive (bool): Whether to recursively process subdirectories.
        mp (bool): Whether to use multiprocessing.
        workers (int | None): Number of workers for multiprocessing.
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
    
    # Build task list
    task_list = [
        (image_paths[i], label_mapping, output_paths[i])
        for i in range(len(image_paths))
    ]

    # Collect per-sample metadata
    series_meta = dict()
    if mp:
        with (
            Pool(processes=workers) as pool,
            tqdm(
                total=len(image_paths),
                desc="Extracting labels",
                leave=True,
                dynamic_ncols=True,
            ) as pbar,
        ):
            result_fetcher = pool.imap_unordered(func=extract_one_sample, iterable=task_list)
            for res, logs in result_fetcher:
                for log in logs:
                    tqdm.write(log)
                if res:
                    series_meta.update(res)
                pbar.update()
    else:
        with tqdm(
            total=len(image_paths),
            desc="Extracting labels",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for task_args in task_list:
                res, logs = extract_one_sample(task_args)
                for log in logs:
                    tqdm.write(log)
                if res:
                    series_meta.update(res)
                pbar.update()
    
    # Save per-sample metadata to JSON
    meta_path = os.path.join(dest_folder, "extract_meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(series_meta, f, indent=4)
    except Exception as e:
        tqdm.write(f"Warning: Could not save extract meta file: {e}")


def parse_label_mappings(mapping_strings: list[str]) -> dict:
    """
    Parse label mapping strings in format "source:target" to dictionary.
    
    Args:
        mapping_strings: List of strings like ["1:0", "5:1", "3:2"]
        
    Returns:
        Dictionary mapping source labels to target labels
    """
    mapping = {}
    for mapping_str in mapping_strings:
        try:
            source, target = mapping_str.split(":")
            source_label = int(source)
            target_label = int(target)
            mapping[source_label] = target_label
        except ValueError:
            raise ValueError(f"Invalid mapping format: {mapping_str}. Expected 'source:target' format.")
    
    return mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and remap labels from a dataset.")
    parser.add_argument("source_folder", type=str, help="The source folder containing .mha files.")
    parser.add_argument("dest_folder", type=str, help="The destination folder for extracted files.")
    parser.add_argument("mappings", type=str, nargs='+', 
                        help="Label mappings in format 'source:target' (e.g., '1:0' '5:1' '3:2')")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively process subdirectories.")
    parser.add_argument("--mp", action="store_true", help="Whether to use multiprocessing.")
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # --- Parameter validation ---
    try:
        # Parse label mappings
        label_mapping = parse_label_mappings(args.mappings)
        
        if not label_mapping:
            raise ValueError("At least one label mapping must be specified.")
        
        # Check for duplicate target labels
        target_labels = list(label_mapping.values())
        if len(target_labels) != len(set(target_labels)):
            raise ValueError("Duplicate target labels found. Each target label should be unique.")
        
        # Print configuration
        print(f"Extracting labels from {args.source_folder} -> {args.dest_folder}")
        print(f"  Label mappings: {label_mapping}")
        print(f"  Recursive: {args.recursive} | Multiprocessing: {args.mp} | Workers: {args.workers}")

    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        return

    # Save configuration
    config_data = vars(args)
    config_data['label_mapping'] = label_mapping
    try:
        os.makedirs(args.dest_folder, exist_ok=True)
        with open(os.path.join(args.dest_folder, "extract_configs.json"), "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")

    # Execute
    extract_task(
        args.source_folder,
        args.dest_folder,
        label_mapping,
        args.recursive,
        args.mp,
        args.workers,
    )
    print(f"Label extraction completed. The extracted dataset is saved in {args.dest_folder}.")


if __name__ == '__main__':
    main()