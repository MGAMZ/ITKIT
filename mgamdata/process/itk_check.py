#!/usr/bin/env python3
import os
import argparse
import sys
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk
import json
from mgamdata.process.meta_json import load_series_meta, compute_series_meta, save_series_meta

# Map dimension letters to indices in ZYX order
DIM_MAP = {'Z': 0, 'Y': 1, 'X': 2}
EPS = 1e-3

def check_sample(args):
    image_path, label_path, cfg = args
    name = os.path.basename(image_path)
    try:
        img = sitk.ReadImage(image_path)
    except Exception as e:
        return (name, [f"read error: {e}"])

    # size and spacing in ZYX
    size = list(img.GetSize()[::-1])
    spacing = list(img.GetSpacing()[::-1])
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
    if reasons:
        return (name, reasons)
    return None

def fast_check(series_meta, cfg, img_dir, lbl_dir, delete):
    invalid = []
    for entry in series_meta:
        name = entry.get('name')
        size = entry.get('size', [])
        spacing = entry.get('spacing', [])
        reasons = []
        # min/max size checks
        for i, mn in enumerate(cfg['min_size']):
            if mn != -1 and size[i] < mn: reasons.append(f"size[{i}]={size[i]} < min_size[{i}]={mn}")
        for i, mx in enumerate(cfg['max_size']):
            if mx != -1 and size[i] > mx: reasons.append(f"size[{i}]={size[i]} > max_size[{i}]={mx}")
        # min/max spacing
        for i, mn in enumerate(cfg['min_spacing']):
            if mn != -1 and spacing[i] < mn: reasons.append(f"spacing[{i}]={spacing[i]:.3f} < min_spacing[{i}]={mn}")
        for i, mx in enumerate(cfg['max_spacing']):
            if mx != -1 and spacing[i] > mx: reasons.append(f"spacing[{i}]={spacing[i]:.3f} > max_spacing[{i}]={mx}")
        # same-spacing
        if cfg['same_spacing']:
            i0, i1 = cfg['same_spacing']
            if abs(spacing[i0]-spacing[i1])>EPS: reasons.append(f"spacing[{i0}] vs spacing[{i1}] differ")
        # same-size
        if cfg['same_size']:
            i0, i1 = cfg['same_size']
            if size[i0]!=size[i1]: reasons.append(f"size[{i0}] vs size[{i1}] differ")
        if reasons:
            invalid.append((name, reasons))
            tqdm.write(f"{name}: {'; '.join(reasons)}")
    # deletion or summary
    if delete:
        for name, reasons in invalid:
            try:
                os.remove(os.path.join(img_dir, name)); os.remove(os.path.join(lbl_dir, name))
            except Exception as e:
                print(f"Error deleting {name}: {e}")
    else:
        if not invalid:
            print("All samples conform to the specified rules.")
    return invalid

def full_check(img_dir, lbl_dir, cfg, mp, delete):
    # build tasks
    files = [f for f in os.listdir(img_dir) if f.lower().endswith('.mha')]
    tasks = [(os.path.join(img_dir,f), os.path.join(lbl_dir,f), cfg) for f in files]
    invalid = []
    if mp:
        with Pool() as pool:
            for res in tqdm(pool.imap_unordered(check_sample, tasks), total=len(tasks), desc="Checking", dynamic_ncols=True):
                if res: invalid.append(res); tqdm.write(f"{res[0]}: {'; '.join(res[1])}")
    else:
        for res in tqdm((check_sample(t) for t in tasks), total=len(tasks), desc="Checking", dynamic_ncols=True):
            if res: invalid.append(res); tqdm.write(f"{res[0]}: {'; '.join(res[1])}")
    if delete:
        for name, reasons in invalid:
            try: os.remove(os.path.join(img_dir,name)); os.remove(os.path.join(lbl_dir,name))
            except Exception as e: print(f"Error deleting {name}: {e}")
    else:
        if not invalid: print("All samples conform to the specified rules.")
    return invalid

def main():
    parser = argparse.ArgumentParser(description="Check itk dataset samples (mha) under image/label for size/spacing rules.")
    parser.add_argument("sample_folder", type=str, help="Root folder containing 'image' and 'label' subfolders.")
    parser.add_argument("--min-size", nargs=3, type=int, default=[-1, -1, -1], help="Min size per Z Y X (-1 ignore)")
    parser.add_argument("--max-size", nargs=3, type=int, default=[-1, -1, -1], help="Max size per Z Y X (-1 ignore)")
    parser.add_argument("--min-spacing", nargs=3, type=float, default=[-1, -1, -1], help="Min spacing per Z Y X (-1 ignore)")
    parser.add_argument("--max-spacing", nargs=3, type=float, default=[-1, -1, -1], help="Max spacing per Z Y X (-1 ignore)")
    parser.add_argument("--same-spacing", nargs=2, choices=['X','Y','Z'], help="Two dims that must have same spacing")
    parser.add_argument("--same-size", nargs=2, choices=['X','Y','Z'], help="Two dims that must have same size")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing")
    parser.add_argument("--delete", action="store_true", help="Delete non-conforming samples")
    args = parser.parse_args()

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
        sys.exit(1)
    # 尝试加载已有 series_meta.json
    series_meta = load_series_meta(args.sample_folder)
    if series_meta is not None:
        fast_check(series_meta, cfg, img_dir, lbl_dir, args.delete)
        return

    # 全量扫描并检查
    invalid = full_check(img_dir, lbl_dir, cfg, args.mp, args.delete)
    # 生成并保存 series_meta.json 供下次加速
    series_meta = compute_series_meta(args.sample_folder)
    save_series_meta(series_meta, args.sample_folder)
    print(f"series_meta.json generated with {len(series_meta)} entries.")
    return

if __name__ == '__main__':
    main()
