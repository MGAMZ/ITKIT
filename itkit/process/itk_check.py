import os, argparse, json, pdb
from tqdm import tqdm

import SimpleITK as sitk
from itkit.process.meta_json import load_series_meta, get_series_meta_path
from itkit.process.base_processor import DatasetProcessor


# Map dimension letters to indices in ZYX order
DIM_MAP = {'Z': 0, 'Y': 1, 'X': 2}
EPS = 1e-3


class CheckProcessor(DatasetProcessor):
    def __init__(self,
                 source_folder: str,
                 cfg: dict,
                 mode: str,
                 output_dir: str | None = None,
                 mp: bool = False,
                 workers: int | None = None):
        super().__init__(source_folder, output_dir, mp, workers, recursive=True)
        self.cfg = cfg
        self.mode = mode
        self.output_dir = output_dir
        self.invalid = []
        self.valid_names = []

    def process_one(self, args: tuple[str, str]) -> dict | None:
        img_path, lbl_path = args
        name = os.path.basename(img_path)
        try:
            img = sitk.ReadImage(img_path)
            img = sitk.DICOMOrient(img, "LPI")
        except Exception as e:
            reasons = [f"read error: {e}"]
            return {name: {"size": [], "spacing": [], "reasons": reasons}}

        # size and spacing in ZYX
        size = list(img.GetSize()[::-1])
        spacing = list(img.GetSpacing()[::-1])
        reasons = self.validate_sample_metadata(size, spacing)
        return {name: {"size": size, "spacing": spacing, "reasons": reasons}}

    def validate_sample_metadata(self, size: list[int], spacing: list[float]) -> list[str]:
        """Validate sample metadata against configuration rules"""
        reasons = []
        
        # min/max size checks
        for i, mn in enumerate(self.cfg['min_size']):
            if mn != -1 and size[i] < mn:
                reasons.append(f"size[{i}]={size[i]} < min_size[{i}]={mn}")
        for i, mx in enumerate(self.cfg['max_size']):
            if mx != -1 and size[i] > mx:
                reasons.append(f"size[{i}]={size[i]} > max_size[{i}]={mx}")
        
        # min/max spacing
        for i, mn in enumerate(self.cfg['min_spacing']):
            if mn != -1 and spacing[i] < mn:
                reasons.append(f"spacing[{i}]={spacing[i]:.3f} < min_spacing[{i}]={mn}")
        for i, mx in enumerate(self.cfg['max_spacing']):
            if mx != -1 and spacing[i] > mx:
                reasons.append(f"spacing[{i}]={spacing[i]:.3f} > max_spacing[{i}]={mx}")
        
        # same-spacing
        if self.cfg['same_spacing']:
            i0, i1 = self.cfg['same_spacing']
            if abs(spacing[i0] - spacing[i1]) > EPS:
                reasons.append(f"spacing[{i0}]={spacing[i0]:.3f} vs spacing[{i1}]={spacing[i1]:.3f} differ")
        
        # same-size
        if self.cfg['same_size']:
            i0, i1 = self.cfg['same_size']
            if size[i0] != size[i1]:
                reasons.append(f"size[{i0}]={size[i0]} vs size[{i1}]={size[i1]} differ")
        
        return reasons

    def process(self):
        # Try to load existing series_meta.json
        series_meta = load_series_meta(self.source_folder)
        if series_meta is not None:
            print("Found existing series_meta.json, performing fast check.")
            self.fast_check(series_meta)
        else:
            print("No series_meta.json found, performing full check.")
            self.full_check()

        self.handle_mode_operations()
    
    def full_check(self):
        """Perform full check when no series_meta.json exists"""
        pairs = self.get_items_to_process()
        series_meta = {}
        
        results = self.process_items(pairs, "Checking files")
        
        # Collect results
        for result in results:
            if result is None:
                continue
            
            for name, data in result.items():
                series_meta[name] = {'size': data['size'], 'spacing': data['spacing']}
                if data['reasons']:
                    self.invalid.append((name, data['reasons']))
                    tqdm.write(f"{name}: {'; '.join(data['reasons'])}")
                else:
                    self.valid_names.append(name)
        
        # Save series_meta.json
        meta_path = get_series_meta_path(self.source_folder)
        try:
            with open(meta_path, 'w') as f:
                json.dump(series_meta, f, indent=4)
            print(f"series_meta.json generated with {len(series_meta)} entries.")
        except Exception as e:
            print(f"Warning: Could not save series_meta.json: {e}")

    def fast_check(self, series_meta: dict):
        for name, entry in series_meta.items():
            size = entry.get('size', [])
            spacing = entry.get('spacing', [])
            
            reasons = self.validate_sample_metadata(size, spacing)
            
            if reasons:
                self.invalid.append((name, reasons))
                tqdm.write(f"{name}: {'; '.join(reasons)}")
            else:
                self.valid_names.append(name)

    def handle_mode_operations(self):
        """Handle operations based on mode (delete/copy/symlink/check)"""
        if self.mode == 'delete':
            img_dir = os.path.join(self.source_folder, 'image')
            lbl_dir = os.path.join(self.source_folder, 'label')
            for name, reasons in self.invalid:
                try:
                    os.remove(os.path.join(img_dir, name))
                    os.remove(os.path.join(lbl_dir, name))
                except Exception as e:
                    print(f"Error deleting {name}: {e}")
            print(f"Deleted {len(self.invalid)} invalid samples")
                    
        elif self.mode == 'symlink':
            if not self.output_dir:
                print("Error: output directory required for symlink mode")
                return
            
            out_img_dir = os.path.join(self.output_dir, 'image')
            out_lbl_dir = os.path.join(self.output_dir, 'label')
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)
            
            success_count = 0
            for name in self.valid_names:
                try:
                    img_src = os.path.join(self.source_folder, 'image', name)
                    lbl_src = os.path.join(self.source_folder, 'label', name)
                    os.symlink(img_src, os.path.join(out_img_dir, name))
                    os.symlink(lbl_src, os.path.join(out_lbl_dir, name))
                    success_count += 1
                except Exception as e:
                    print(f"Error symlinking {name}: {e}")
            print(f"Symlinked {success_count} valid samples to {self.output_dir}")
            
        elif self.mode == 'copy':
            if not self.output_dir:
                print("Error: output directory required for copy mode")
                return
            out_img_dir = os.path.join(self.output_dir, 'image')
            out_lbl_dir = os.path.join(self.output_dir, 'label')
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)
            
            success_count = 0
            for name in self.valid_names:
                try:
                    img_src = os.path.join(self.source_folder, 'image', name)
                    lbl_src = os.path.join(self.source_folder, 'label', name)
                    import shutil
                    shutil.copy(img_src, out_img_dir)
                    shutil.copy(lbl_src, out_lbl_dir)
                    success_count += 1
                except Exception as e:
                    print(f"Error copying {name}: {e}")
            print(f"Copied {success_count} valid samples to {self.output_dir}")
            
        else:  # check mode
            if not self.invalid:
                print("All samples conform to the specified rules.")
            else:
                print(f"Found {len(self.invalid)} invalid samples")

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
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")
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

    processor = CheckProcessor(args.sample_folder, cfg, args.mode, args.output, args.mp, args.workers)
    processor.process()


if __name__ == '__main__':
    main()
