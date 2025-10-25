import os, argparse, json, shutil
from tqdm import tqdm
from enum import Enum

import SimpleITK as sitk
from itkit.process.base_processor import DatasetProcessor, SingleFolderProcessor
from itkit.process.meta_json import load_series_meta, get_series_meta_path


DIM_MAP = {"Z": 0, "Y": 1, "X": 2}
EPS = 1e-3


class ProcessorType(Enum):
    """Supported processor types for validation"""

    DATASET = "dataset"
    SINGLE = "single"


class ValidationMixin:
    """Mixin providing validation logic and operations for size/spacing checks"""

    def __init__(self, cfg: dict, mode: str):
        """Initialize validation configuration

        Args:
            cfg: Configuration dict with validation rules
            mode: Operation mode (check/delete/copy/symlink)
        """
        self.cfg = cfg
        self.mode = mode
        self.invalid = []  # List of (name, reasons, paths)
        self.valid_items = []  # List of (name, paths)

    def validate_sample_metadata(
        self, size: list[int], spacing: list[float]
    ) -> list[str]:
        """Validate sample metadata against configuration rules"""
        reasons = []

        # min-size check
        if self.cfg["min_size"]:
            for i, mn in enumerate(self.cfg["min_size"]):
                if mn != -1 and size[i] < mn:
                    reasons.append(f"size[{i}]={size[i]} < min_size[{i}]={mn}")

        # max-size check
        if self.cfg["max_size"]:
            for i, mx in enumerate(self.cfg["max_size"]):
                if mx != -1 and size[i] > mx:
                    reasons.append(f"size[{i}]={size[i]} > max_size[{i}]={mx}")

        # min-spacing check
        if self.cfg["min_spacing"]:
            for i, mn in enumerate(self.cfg["min_spacing"]):
                if mn != -1.0 and spacing[i] < mn:
                    reasons.append(
                        f"spacing[{i}]={spacing[i]:.3f} < min_spacing[{i}]={mn}"
                    )

        # max-spacing check
        if self.cfg["max_spacing"]:
            for i, mx in enumerate(self.cfg["max_spacing"]):
                if mx != -1.0 and spacing[i] > mx:
                    reasons.append(
                        f"spacing[{i}]={spacing[i]:.3f} > max_spacing[{i}]={mx}"
                    )

        # same-spacing check
        if self.cfg["same_spacing"]:
            i0, i1 = self.cfg["same_spacing"]
            if abs(spacing[i0] - spacing[i1]) > EPS:
                reasons.append(
                    f"spacing[{i0}]={spacing[i0]:.3f} vs spacing[{i1}]={spacing[i1]:.3f} differ"
                )

        # same-size check
        if self.cfg["same_size"]:
            i0, i1 = self.cfg["same_size"]
            if size[i0] != size[i1]:
                reasons.append(f"size[{i0}]={size[i0]} vs size[{i1}]={size[i1]} differ")

        return reasons

    def _read_image_metadata(
        self, img_path: str
    ) -> tuple[list[int], list[float], list[str]]:
        """Read image and return (size, spacing, reasons)"""
        try:
            img = sitk.ReadImage(img_path)
            size = list(img.GetSize()[::-1])  # Convert to ZYX
            spacing = list(img.GetSpacing()[::-1])
            reasons = self.validate_sample_metadata(size, spacing)
            return size, spacing, reasons
        except Exception as e:
            return [], [], [f"Failed to read: {str(e)}"]

    def fast_check_with_meta(self, series_meta: dict, item_dict: dict):
        """Fast check using existing series_meta.json

        Args:
            series_meta: Loaded metadata dictionary
            item_dict: Dictionary mapping names to file paths
        """
        for name, entry in series_meta.items():
            if name not in item_dict:
                continue  # Skip if file no longer exists

            size = entry.get("size", [])
            spacing = entry.get("spacing", [])
            reasons = self.validate_sample_metadata(size, spacing)

            paths = item_dict[name]
            if reasons:
                self.invalid.append((name, reasons, paths))
                tqdm.write(f"{name}: {'; '.join(reasons)}")
            else:
                self.valid_items.append((name, paths))

    def execute_operation(self, output_dir: str | None = None):
        """Execute mode-specific operations after validation"""
        if self.mode == "delete":
            self._operation_delete()
        elif self.mode == "symlink":
            self._operation_symbolic_link(output_dir)
        elif self.mode == "copy":
            self._operation_copy(output_dir)
        else:  # check mode
            if not self.invalid:
                print("All samples conform to the specified rules.")
            else:
                print(f"Found {len(self.invalid)} invalid samples")

    def _operation_delete(self):
        """Delete invalid samples"""
        import os

        for name, reasons, paths in self.invalid:
            try:
                if isinstance(paths, tuple):  # Dataset mode: (img, lbl)
                    img_path, lbl_path = paths
                    os.remove(img_path)
                    os.remove(lbl_path)
                else:  # Single mode
                    os.remove(paths)
            except Exception as e:
                print(f"Error deleting {name}: {e}")
        print(f"Deleted {len(self.invalid)} invalid samples")

    def _operation_symbolic_link(self, output_dir: str | None):
        """Symlink valid samples to output directory"""
        if not output_dir:
            print("Error: output directory required for symlink mode")
            return

        if isinstance(self.valid_items[0][1], tuple):  # Dataset mode
            out_img_dir = os.path.join(output_dir, "image")
            out_lbl_dir = os.path.join(output_dir, "label")
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)

            success_count = 0
            for name, paths in self.valid_items:
                try:
                    img_src, lbl_src = paths
                    os.symlink(img_src, os.path.join(out_img_dir, name))
                    os.symlink(lbl_src, os.path.join(out_lbl_dir, name))
                    success_count += 1
                except Exception as e:
                    print(f"Error symlinking {name}: {e}")
        else:  # Single mode
            os.makedirs(output_dir, exist_ok=True)
            success_count = 0
            for name, img_path in self.valid_items:
                try:
                    os.symlink(img_path, os.path.join(output_dir, name))
                    success_count += 1
                except Exception as e:
                    print(f"Error symlinking {name}: {e}")

        print(f"Symlinked {success_count} valid samples to {output_dir}")

    def _operation_copy(self, output_dir: str | None):
        """Copy valid samples to output directory"""
        if not output_dir:
            print("Error: output directory required for copy mode")
            return

        if isinstance(self.valid_items[0][1], tuple):  # Dataset mode
            out_img_dir = os.path.join(output_dir, "image")
            out_lbl_dir = os.path.join(output_dir, "label")
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)

            success_count = 0
            for name, paths in self.valid_items:
                try:
                    img_src, lbl_src = paths
                    shutil.copy(img_src, out_img_dir)
                    shutil.copy(lbl_src, out_lbl_dir)
                    success_count += 1
                except Exception as e:
                    print(f"Error copying {name}: {e}")
        else:  # Single mode
            os.makedirs(output_dir, exist_ok=True)
            success_count = 0
            for name, img_path in self.valid_items:
                try:
                    shutil.copy(img_path, output_dir)
                    success_count += 1
                except Exception as e:
                    print(f"Error copying {name}: {e}")

        print(f"Copied {success_count} valid samples to {output_dir}")


class DatasetCheckProcessor(DatasetProcessor, ValidationMixin):
    """Processor for checking image/label paired datasets"""

    def __init__(
        self,
        source_folder: str,
        cfg: dict,
        mode: str,
        output_dir: str | None = None,
        mp: bool = False,
        workers: int | None = None,
    ):
        DatasetProcessor.__init__(
            self, source_folder, output_dir, mp, workers, recursive=False
        )
        ValidationMixin.__init__(self, cfg, mode)

    def process_one(self, args: tuple[str, str]) -> dict | None:
        """Process one image/label pair"""
        img_path, lbl_path = args
        name = os.path.basename(img_path)
        size, spacing, reasons = self._read_image_metadata(img_path)

        if reasons:
            self.invalid.append((name, reasons, (img_path, lbl_path)))
        else:
            self.valid_items.append((name, (img_path, lbl_path)))

        return {name: {"size": size, "spacing": spacing}} if size else None

    def process(self, desc: str = "Checking") -> dict:
        """Main processing with fast check support"""
        # Try fast check first
        series_meta = load_series_meta(self.source_folder)
        if series_meta is not None:
            print("Found existing series_meta.json, performing fast check.")
            items = self.get_items_to_process()
            item_dict = {os.path.basename(img): (img, lbl) for img, lbl in items}
            self.fast_check_with_meta(series_meta, item_dict)
        else:
            print("No series_meta.json found, performing full check.")
            super().process(desc)

        # Execute operations based on mode
        self.execute_operation(self.dest_folder)
        
        # Save metadata to source folder in check mode
        if not self.dest_folder:
            meta_path = get_series_meta_path(self.source_folder)
            self.save_meta(meta_path)
        
        return self.meta


class SingleCheckProcessor(SingleFolderProcessor, ValidationMixin):
    """Processor for checking single folder of images"""

    def __init__(
        self,
        source_folder: str,
        cfg: dict,
        mode: str,
        output_dir: str | None = None,
        mp: bool = False,
        workers: int | None = None,
    ):
        SingleFolderProcessor.__init__(
            self, source_folder, output_dir, mp, workers, recursive=False
        )
        ValidationMixin.__init__(self, cfg, mode)

    def process_one(self, img_path: str) -> dict | None:
        """Process one image file"""
        name = os.path.basename(img_path)
        size, spacing, reasons = self._read_image_metadata(img_path)

        if reasons:
            self.invalid.append((name, reasons, img_path))
        else:
            self.valid_items.append((name, img_path))

        return {name: {"size": size, "spacing": spacing}} if size else None

    def process(self, desc: str = "Checking") -> dict:
        """Main processing with fast check support"""
        # Try fast check first
        series_meta = load_series_meta(self.source_folder)
        if series_meta is not None:
            print("Found existing series_meta.json, performing fast check.")
            items = self.get_items_to_process()
            item_dict = {os.path.basename(img): img for img in items}
            self.fast_check_with_meta(series_meta, item_dict)
        else:
            print("No series_meta.json found, performing full check.")
            super().process(desc)

        # Execute operations based on mode
        self.execute_operation(self.dest_folder)
        
        # Save metadata to source folder in check mode
        if not self.dest_folder:
            meta_path = get_series_meta_path(self.source_folder)
            self.save_meta(meta_path)
        
        return self.meta


def CheckProcessor(
    source_folder: str,
    cfg: dict,
    mode: str,
    processor_type: ProcessorType = ProcessorType.DATASET,
    output_dir: str | None = None,
    mp: bool = False,
    workers: int | None = None,
):
    """Factory function to create appropriate check processor"""
    if processor_type == ProcessorType.DATASET:
        return DatasetCheckProcessor(source_folder, cfg, mode, output_dir, mp, workers)
    else:
        return SingleCheckProcessor(source_folder, cfg, mode, output_dir, mp, workers)


def main():
    parser = argparse.ArgumentParser(
        description="Check itk dataset samples (mha) for size/spacing rules. "
        "Supports both dataset (image/label pairs) and single folder structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check dataset with image/label pairs
  %(prog)s check /path/to/data --processor-type dataset --min-size -1 128 128
  
  # Check single folder of images
  %(prog)s check /path/to/images --processor-type single --max-spacing 2.0 2.0 2.0
  
  # Symlink valid samples from dataset
  %(prog)s symlink /path/to/data -o /path/to/output --processor-type dataset --same-spacing X Y
        """,
    )
    parser.add_argument(
        "mode",
        choices=["check", "delete", "copy", "symlink"],
        help="Operation mode: check (validate only), delete (remove invalid), "
        "copy (copy valid files), symlink (symlink valid files)",
    )
    parser.add_argument("sample_folder", help="Path to sample folder")
    parser.add_argument(
        "-o", "--output", help="Output directory (required for copy/symlink modes)"
    )
    parser.add_argument(
        "--processor-type",
        choices=["dataset", "single"],
        default="dataset",
        help="Processor type: dataset (image/label pairs) or single (single folder)",
    )

    # Size constraints
    parser.add_argument(
        "--min-size",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        help="Minimum size constraint (use -1 to skip dimension)",
    )
    parser.add_argument(
        "--max-size",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        help="Maximum size constraint (use -1 to skip dimension)",
    )

    # Spacing constraints
    parser.add_argument(
        "--min-spacing",
        nargs=3,
        type=float,
        metavar=("Z", "Y", "X"),
        help="Minimum spacing constraint (use -1 to skip dimension)",
    )
    parser.add_argument(
        "--max-spacing",
        nargs=3,
        type=float,
        metavar=("Z", "Y", "X"),
        help="Maximum spacing constraint (use -1 to skip dimension)",
    )

    # Same dimension constraints
    parser.add_argument(
        "--same-spacing",
        nargs=2,
        choices=["Z", "Y", "X"],
        help="Require two dimensions to have same spacing (e.g., X Y)",
    )
    parser.add_argument(
        "--same-size",
        nargs=2,
        choices=["Z", "Y", "X"],
        help="Require two dimensions to have same size (e.g., X Y)",
    )

    # Performance
    parser.add_argument("--mp", action="store_true", help="Enable multiprocessing")
    parser.add_argument(
        "--workers", type=int, help="Number of workers (default: CPU count)"
    )

    args = parser.parse_args()

    # Validate output requirement
    if args.mode in ["copy", "symlink"] and not args.output:
        parser.error(f"{args.mode} mode requires --output directory")

    # Parse same-spacing/same-size to indices
    same_spacing = None
    if args.same_spacing:
        same_spacing = (DIM_MAP[args.same_spacing[0]], DIM_MAP[args.same_spacing[1]])

    same_size = None
    if args.same_size:
        same_size = (DIM_MAP[args.same_size[0]], DIM_MAP[args.same_size[1]])

    # Build configuration
    cfg = {
        "min_size": args.min_size,
        "max_size": args.max_size,
        "min_spacing": args.min_spacing,
        "max_spacing": args.max_spacing,
        "same_spacing": same_spacing,
        "same_size": same_size,
    }

    # Create processor and run
    processor_type = (
        ProcessorType.DATASET
        if args.processor_type == "dataset"
        else ProcessorType.SINGLE
    )
    processor = CheckProcessor(
        source_folder = args.sample_folder,
        cfg = cfg,
        mode = args.mode,
        processor_type = processor_type,
        output_dir = args.output,
        mp = args.mp,
        workers = args.workers,
    )

    # Process and save metadata
    processor.process("Checking")
    processor.save_meta()

    print("Check completed.")


if __name__ == "__main__":
    main()
