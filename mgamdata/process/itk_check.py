#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk
from mgamdata.process.meta_json import load_series_meta, get_series_meta_path


# Map dimension letters to indices in ZYX order
DIM_MAP = {'Z': 0, 'Y': 1, 'X': 2}
EPS = 1e-3


def check_sample(args):
    image_path, label_path, cfg = args
    name = os.path.basename(image_path)
    try:
        img = sitk.ReadImage(image_path)
        img = sitk.DICOMOrient(img, "LPI")
    except Exception as e:
        return (name, [], [], [f"read error: {e}"])

    # size and spacing in ZYX
    size = list(img.GetSize()[::-1])
    spacing = list(img.GetSpacing()[::-1])
    reasons = validate_sample_metadata(size, spacing, cfg)
    return name, size, spacing, reasons


def validate_sample_metadata(size, spacing, cfg):
    """Validate sample metadata against configuration rules"""
    reasons = []
    
    # min/max size checks
    for i, mn in enumerate(cfg['min_size']):
        if mn != -1 and size[i] < mn:
            reasons.append(f"size[{i}]={size[i]} < min_size[{i}]={mn}")
    for i, mx in enumerate(cfg['max_size']):
        if mx != -1 and size[i] > mx:
            reasons.append(f"size[{i}]={size[i]} > max_size[{i}]={mx}")
    
    # min/max spacing
    for i, mn in enumerate(cfg['min_spacing']):
        if mn != -1 and spacing[i] < mn:
            reasons.append(f"spacing[{i}]={spacing[i]:.3f} < min_spacing[{i}]={mn}")
    for i, mx in enumerate(cfg['max_spacing']):
        if mx != -1 and spacing[i] > mx:
            reasons.append(f"spacing[{i}]={spacing[i]:.3f} > max_spacing[{i}]={mx}")
    
    # same-spacing
    if cfg['same_spacing']:
        i0, i1 = cfg['same_spacing']
        if abs(spacing[i0] - spacing[i1]) > EPS:
            reasons.append(f"spacing[{i0}]={spacing[i0]:.3f} vs spacing[{i1}]={spacing[i1]:.3f} differ")
    
    # same-size
    if cfg['same_size']:
        i0, i1 = cfg['same_size']
        if size[i0] != size[i1]:
            reasons.append(f"size[{i0}]={size[i0]} vs size[{i1}]={size[i1]} differ")
    
    return reasons


def handle_mode_operations(mode, invalid, valid_names, img_dir, lbl_dir, output_dir=None):
    """Handle operations based on mode (delete/copy/symlink/check)"""
    if mode == 'delete':
        for name, reasons in invalid:
            try:
                os.remove(os.path.join(img_dir, name))
                os.remove(os.path.join(lbl_dir, name))
            except Exception as e:
                print(f"Error deleting {name}: {e}")
        print(f"Deleted {len(invalid)} invalid samples")
                
    elif mode == 'symlink':
        if not output_dir:
            print("Error: output directory required for symlink mode")
            return
        
        out_img_dir = os.path.join(output_dir, 'image')
        out_lbl_dir = os.path.join(output_dir, 'label')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        
        success_count = 0
        for name in valid_names:
            try:
                src_img = os.path.abspath(os.path.join(img_dir, name))
                src_lbl = os.path.abspath(os.path.join(lbl_dir, name))
                dst_img = os.path.join(out_img_dir, name)
                dst_lbl = os.path.join(out_lbl_dir, name)
                os.symlink(src_img, dst_img)
                os.symlink(src_lbl, dst_lbl)
                success_count += 1
            except Exception as e:
                print(f"Error symlinking {name}: {e}")
        print(f"Symlinked {success_count} valid samples to {output_dir}")
        
    elif mode == 'copy':
        if not output_dir:
            print("Error: output directory required for copy mode")
            return
        
        import shutil
        out_img_dir = os.path.join(output_dir, 'image')
        out_lbl_dir = os.path.join(output_dir, 'label')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        
        success_count = 0
        for name in valid_names:
            try:
                src_img = os.path.join(img_dir, name)
                src_lbl = os.path.join(lbl_dir, name)
                dst_img = os.path.join(out_img_dir, name)
                dst_lbl = os.path.join(out_lbl_dir, name)
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                success_count += 1
            except Exception as e:
                print(f"Error copying {name}: {e}")
        print(f"Copied {success_count} valid samples to {output_dir}")
        
    else:  # check mode
        if not invalid:
            print("All samples conform to the specified rules.")
        else:
            print(f"Found {len(invalid)} invalid samples")


def fast_check(series_meta:dict[str, dict], cfg, img_dir, lbl_dir, mode, output_dir=None):
    invalid = []
    valid_names = []
    
    for name, entry in series_meta.items():
        size = entry.get('size', [])
        spacing = entry.get('spacing', [])
        
        reasons = validate_sample_metadata(size, spacing, cfg)
        
        if reasons:
            invalid.append((name, reasons))
            tqdm.write(f"{name}: {'; '.join(reasons)}")
        else:
            valid_names.append(name)
    
    handle_mode_operations(mode, invalid, valid_names, img_dir, lbl_dir, output_dir)
    return invalid


def full_check(img_dir, lbl_dir, cfg, mp, mode, output_dir=None):
    # Perform checks and collect metadata in one pass
    files = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in ['.mha', '.mhd', '.nii', '.gz']]
    tasks = [(os.path.join(img_dir, f), os.path.join(lbl_dir, f), cfg) for f in files]
    invalid = []
    valid_names = []
    series_meta = {}
    
    # Process all samples
    if mp:
        with Pool() as pool:
            results = list(tqdm(pool.imap_unordered(check_sample, tasks), 
                              total=len(tasks), desc="Checking", dynamic_ncols=True))
    else:
        results = list(tqdm((check_sample(t) for t in tasks), 
                           total=len(tasks), desc="Checking", dynamic_ncols=True))
    
    # Collect results
    for name, size, spacing, reasons in results:
        series_meta[name] = {'size': size, 'spacing': spacing}
        if reasons:
            invalid.append((name, reasons))
            tqdm.write(f"{name}: {'; '.join(reasons)}")
        else:
            valid_names.append(name)
    
    handle_mode_operations(mode, invalid, valid_names, img_dir, lbl_dir, output_dir)
    return invalid, series_meta


def main():
    parser = argparse.ArgumentParser(description="Check itk dataset samples (mha) under image/label for size/spacing rules.")
    parser.add_argument("mode", choices=['check', 'delete', 'copy', 'symlink'], help="Operation mode: check (validate only), delete (remove invalid), copy (copy valid files), symlink (symlink valid files)")
    parser.add_argument("sample_folder", type=str, help="Root folder containing 'image' and 'label' subfolders.")
    parser.add_argument("-o", "--output", type=str, help="Output directory for copy/symlink mode")
    parser.add_argument("--min-size", nargs=3, type=int, default=[-1, -1, -1], help="Min size per Z Y X (-1 ignore)")
    parser.add_argument("--max-size", nargs=3, type=int, default=[-1, -1, -1], help="Max size per Z Y X (-1 ignore)")
    parser.add_argument("--min-spacing", nargs=3, type=float, default=[-1, -1, -1], help="Min spacing per Z Y X (-1 ignore)")
    parser.add_argument("--max-spacing", nargs=3, type=float, default=[-1, -1, -1], help="Max spacing per Z Y X (-1 ignore)")
    parser.add_argument("--same-spacing", nargs=2, choices=['X','Y','Z'], help="Two dims that must have same spacing")
    parser.add_argument("--same-size", nargs=2, choices=['X','Y','Z'], help="Two dims that must have same size")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing")
    args = parser.parse_args()

    # Validate arguments
    if args.mode in ['copy', 'symlink'] and not args.output:
        print(f"Error: --output is required for {args.mode} mode")
        exit(1)

    # prepare config
    cfg = {
        'min_size': args.min_size,
        'max_size': args.max_size,
        'min_spacing': args.min_spacing,
        'max_spacing': args.max_spacing,
        'same_spacing': None,
        'same_size': None
    }
    if args.same_spacing:
        cfg['same_spacing'] = (DIM_MAP[args.same_spacing[0]], DIM_MAP[args.same_spacing[1]])
    if args.same_size:
        cfg['same_size'] = (DIM_MAP[args.same_size[0]], DIM_MAP[args.same_size[1]])

    img_dir = os.path.join(args.sample_folder, 'image')
    lbl_dir = os.path.join(args.sample_folder, 'label')
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        print(f"Missing 'image' or 'label' subfolders in {args.sample_folder}")
        exit(1)
        
    # Try to load existing series_meta.json
    series_meta = load_series_meta(args.sample_folder)
    if series_meta is not None:
        fast_check(series_meta, cfg, img_dir, lbl_dir, args.mode, args.output)
        return
        
    # Full scan and check, generate metadata
    invalid, series_meta = full_check(img_dir, lbl_dir, cfg, args.mp, args.mode, args.output)
    
    # Save series_meta.json
    meta_path = get_series_meta_path(args.sample_folder)
    try:
        with open(meta_path, 'w') as f:
            json.dump(series_meta, f, indent=4)
        print(f"series_meta.json generated with {len(series_meta)} entries.")
    except Exception as e:
        print(f"Warning: Could not save series_meta.json: {e}")


if __name__ == '__main__':
    main()
