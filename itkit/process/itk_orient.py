import os, argparse, pdb

import SimpleITK as sitk
from itkit.process.base_processor import SingleFolderProcessor


class OrientProcessor(SingleFolderProcessor):
    def __init__(self,
                 source_folder: str,
                 dest_folder: str,
                 orient: str,
                 mp: bool = False,
                 workers: int | None = None):
        super().__init__(source_folder, dest_folder, mp, workers, recursive=True)
        self.orient = orient

    def process_one(self, file_path: str) -> None:
        rel_path = os.path.relpath(file_path, self.source_folder)
        dst_path = os.path.join(self.dest_folder, rel_path)
        
        # Skip if target file already exists
        if os.path.exists(dst_path):
            print(f"Target file already exists, skipping: {dst_path}")
            return None
            
        try:
            img = sitk.ReadImage(file_path)
            lpi_img = sitk.DICOMOrient(img, self.orient.upper())
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            sitk.WriteImage(lpi_img, dst_path, True)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert all .mha files under the source directory to the specified orientation (e.g. LPI) while preserving the original directory structure.")
    parser.add_argument('src_dir', help='Source directory')
    parser.add_argument('dst_dir', help='Destination directory')
    parser.add_argument('orient', help='Target orientation (e.g. LPI)')
    parser.add_argument('--mp', action='store_true', help='Use multiprocessing')
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")
    args = parser.parse_args()

    if not os.path.isdir(args.src_dir):
        print(f"Source directory does not exist: {args.src_dir}")
        return

    if os.path.abspath(args.src_dir) == os.path.abspath(args.dst_dir):
        print("Source and destination directories cannot be the same!")
        return

    processor = OrientProcessor(args.src_dir, args.dst_dir, args.orient, args.mp, args.workers)
    processor.process("Orienting files")



if __name__ == '__main__':
    main()