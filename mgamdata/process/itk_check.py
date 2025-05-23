#!/usr/bin/env python3
import os
import argparse
import sys
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk

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

    files = [f for f in os.listdir(img_dir) if f.lower().endswith('.mha')]
    tasks = []
    for f in files:
        ip = os.path.join(img_dir, f)
        lp = os.path.join(lbl_dir, f)
        tasks.append((ip, lp, cfg))

    results = []
    if args.mp:
        with Pool() as pool:
            for res in tqdm(pool.imap_unordered(check_sample, tasks),
                            total=len(tasks),
                            desc="Checking",
                            dynamic_ncols=True):
                if res:
                    results.append(res)
    else:
        for t in tqdm(tasks,
                      desc="Checking",
                      total=len(tasks),
                      dynamic_ncols=True):
            res = check_sample(t)
            if res:
                results.append(res)

    # report or delete
    for name, reasons in results:
        msg = f"{name}: " + "; ".join(reasons)
        if args.delete:
            try:
                os.remove(os.path.join(img_dir, name))
                os.remove(os.path.join(lbl_dir, name))
                print(f"Deleted {name} due to: {', '.join(reasons)}")
            except Exception as e:
                print(f"Error deleting {name}: {e}")
        else:
            print(msg)

    if not results:
        print("All samples conform to the specified rules.")

if __name__ == '__main__':
    main()
