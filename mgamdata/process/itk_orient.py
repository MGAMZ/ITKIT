import os
import argparse
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk


def convert_to_lpi(args):
    src_path, dst_path, orient = args
    try:
        img = sitk.ReadImage(src_path)
        lpi_img = sitk.DICOMOrient(img, orient.upper())
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sitk.WriteImage(lpi_img, dst_path, True)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return


def process_files(src_dir, dst_dir, orient, use_mp=False):
    mha_files = [os.path.relpath(f, src_dir) 
                 for f in glob(os.path.join(src_dir, '**', '*.mha'), recursive=True)]
    tasks = []
    for rel_path in mha_files:
        src_path = os.path.join(src_dir, rel_path)
        dst_path = os.path.join(dst_dir, rel_path)
        if os.path.exists(dst_path):
            print(f"Target file already exists, skipping: {dst_path}")
            continue
        tasks.append((src_path, dst_path, orient))

    if use_mp:
        from multiprocessing import Pool, cpu_count
        with Pool(cpu_count()) as pool:
            for i in tqdm(pool.imap_unordered(convert_to_lpi, tasks),
                          total=len(tasks),
                          desc="Converting to LPI",
                          dynamic_ncols=True):
                ...
    else:
        for arg in tqdm(tasks,
                        desc="Converting to LPI",
                        dynamic_ncols=True):
            convert_to_lpi(arg)


def main():
    parser = argparse.ArgumentParser(description="Convert all .mha files under the source directory to the specified orientation (e.g. LPI) while preserving the original directory structure.")
    parser.add_argument('src_dir', help='Source directory')
    parser.add_argument('dst_dir', help='Destination directory')
    parser.add_argument('orient', help='Target orientation (e.g. LPI)')
    parser.add_argument('--mp', action='store_true', help='Use multiprocessing')
    args = parser.parse_args()

    if not os.path.isdir(args.src_dir):
        print(f"Source directory does not exist: {args.src_dir}")
        return

    if os.path.abspath(args.src_dir) == os.path.abspath(args.dst_dir):
        print("Source and destination directories cannot be the same!")
        return

    process_files(args.src_dir, args.dst_dir, args.orient, args.mp)



if __name__ == '__main__':
    main()