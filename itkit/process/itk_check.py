import argparse
import json
import os
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import SimpleITK as sitk
from tqdm import tqdm

from itkit.process.base_processor import DatasetProcessor, SingleFolderProcessor
from itkit.process.meta_json import get_series_meta_path, load_series_meta
from itkit.process.metadata_models import SeriesMetadata

DIM_MAP = {"Z": 0, "Y": 1, "X": 2}
EPS = 1e-3


class ProcessorType(Enum):
    """Supported processor types for validation"""

    DATASET = "dataset"
    SINGLE = "single"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    is_valid: bool
    reasons: list[str]
    paths: Any  # Can be str or tuple[str, str] depending on processor


class CheckMixin:
    """Mixin providing validation logic and abstract operations"""

    def __init__(self, cfg: dict, mode: str):
        """Initialize validation configuration

        Args:
            cfg: Configuration dict with validation rules
            mode: Operation mode (check/delete/copy/symlink)
        """
        self.cfg = cfg
        self.mode = mode
        self.results: list[ValidationResult] = []

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

    def fast_check_with_meta(self, series_meta: dict, item_dict: dict):
        """Fast check using existing meta.json

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
                tqdm.write(f"{name}: {'; '.join(reasons)}")
                self.results.append(ValidationResult(name, False, reasons, paths))
            else:
                self.results.append(ValidationResult(name, True, [], paths))

    def execute_operation(self, output_dir: str | None = None):
        """Execute mode-specific operations after validation"""
        invalid_items = [r for r in self.results if not r.is_valid]
        valid_items = [r for r in self.results if r.is_valid]

        if self.mode == "delete":
            self._run_delete(invalid_items)
        elif self.mode == "symlink":
            self._run_symlink(valid_items, output_dir)
        elif self.mode == "copy":
            self._run_copy(valid_items, output_dir)
        else:  # check mode
            if not invalid_items:
                print("All samples conform to the specified rules.")
            else:
                print(f"Found {len(invalid_items)} invalid samples")

    def _run_delete(self, items: list[ValidationResult]):
        for item in items:
            try:
                self.op_delete(item.paths)
            except Exception as e:
                print(f"Error deleting {item.name}: {e}")
        print(f"Deleted {len(items)} invalid samples")

    def _run_symlink(self, items: list[ValidationResult], output_dir: str | None):
        if not output_dir:
            print("Error: output directory required for symlink mode")
            return

        self.prepare_output_dir(output_dir)
        success_count = 0
        for item in items:
            try:
                self.op_symlink(item.name, item.paths, output_dir)
                success_count += 1
            except Exception as e:
                print(f"Error symlinking {item.name}: {e}")
        print(f"Symlinked {success_count} valid samples to {output_dir}")

    def _run_copy(self, items: list[ValidationResult], output_dir: str | None):
        if not output_dir:
            print("Error: output directory required for copy mode")
            return

        self.prepare_output_dir(output_dir)
        success_count = 0
        for item in items:
            try:
                self.op_copy(item.name, item.paths, output_dir)
                success_count += 1
            except Exception as e:
                print(f"Error copying {item.name}: {e}")
        print(f"Copied {success_count} valid samples to {output_dir}")

    @abstractmethod
    def op_delete(self, paths: Any):
        pass

    @abstractmethod
    def op_symlink(self, name: str, paths: Any, output_dir: str):
        pass

    @abstractmethod
    def op_copy(self, name: str, paths: Any, output_dir: str):
        pass

    @abstractmethod
    def prepare_output_dir(self, output_dir: str):
        pass


class DatasetCheckProcessor(DatasetProcessor, CheckMixin):
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
        DatasetProcessor.__init__(self, source_folder, dest_folder=None, mp=mp, workers=workers)
        CheckMixin.__init__(self, cfg, mode)
        self.output_dir = output_dir

    def process_one(self, args: tuple[str, str]) -> tuple[SeriesMetadata | None, ValidationResult]:
        """Process one image/label pair"""
        img_path, lbl_path = args
        name = os.path.basename(img_path)

        try:
            lbl = sitk.ReadImage(lbl_path)
            size = list(lbl.GetSize()[::-1])
            spacing = list(lbl.GetSpacing()[::-1])
            reasons = self.validate_sample_metadata(size, spacing)

            is_valid = len(reasons) == 0
            res = ValidationResult(name, is_valid, reasons, (img_path, lbl_path))

            return SeriesMetadata.from_sitk_image(lbl, name), res
        except Exception as e:
            res = ValidationResult(name, False, [f"Failed to read: {str(e)}"], (img_path, lbl_path))
            return None, res

    def _collect_results(self, results: list):
        """Collect both metadata and validation results from workers"""
        for meta, res in results:
            if res:
                self.results.append(res)
            if meta:
                self.meta_manager.update(meta, allow_and_overwrite_existed=self.ALLOW_AND_OVERWRITE_EXISTED_METADATA)

    def process(self, desc="Checking"):
        """Main processing with fast check support"""
        # Try fast check first
        try:
            series_meta = load_series_meta(self.source_folder)
        except json.JSONDecodeError:
            print("meta.json is corrupted, removing and performing full check.")
            meta_path = get_series_meta_path(self.source_folder)
            os.remove(meta_path)
            series_meta = None

        run_full_check = False
        if series_meta is not None:
            print("Found existing meta.json, performing fast check.")
            items = self.get_items_to_process()
            item_dict = {os.path.basename(img): (img, lbl) for img, lbl in items}
            self.fast_check_with_meta(series_meta, item_dict)
        else:
            print("No meta.json found, performing full check.")
            super().process(desc)
            run_full_check = True

        # Execute operations based on mode
        self.execute_operation(self.output_dir)

        # Save metadata to source folder if full check was run
        if run_full_check:
            meta_path = get_series_meta_path(self.source_folder)
            self.save_meta(meta_path)

    def prepare_output_dir(self, output_dir: str):
        os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "label"), exist_ok=True)

    def op_delete(self, paths: Any):
        img_path, lbl_path = paths
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(lbl_path): os.remove(lbl_path)

    def op_symlink(self, name: str, paths: Any, output_dir: str):
        img_src, lbl_src = paths

        img_dst = os.path.join(output_dir, "image", name)
        img_target = os.path.relpath(os.path.abspath(img_src), os.path.dirname(os.path.abspath(img_dst)))
        os.symlink(img_target, img_dst)

        lbl_dst = os.path.join(output_dir, "label", name)
        lbl_target = os.path.relpath(os.path.abspath(lbl_src), os.path.dirname(os.path.abspath(lbl_dst)))
        os.symlink(lbl_target, lbl_dst)

    def op_copy(self, name: str, paths: Any, output_dir: str):
        img_src, lbl_src = paths
        shutil.copy(img_src, os.path.join(output_dir, "image"))
        shutil.copy(lbl_src, os.path.join(output_dir, "label"))


class SingleCheckProcessor(SingleFolderProcessor, CheckMixin):
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
            self, source_folder, dest_folder=None, mp=mp, workers=workers, recursive=False
        )
        CheckMixin.__init__(self, cfg, mode)
        self.output_dir = output_dir

    def process_one(self, args) -> tuple[SeriesMetadata | None, ValidationResult]:
        """Process one image file"""
        img_path = args
        name = os.path.basename(img_path)

        try:
            img = sitk.ReadImage(img_path)
            size = list(img.GetSize()[::-1])
            spacing = list(img.GetSpacing()[::-1])
            reasons = self.validate_sample_metadata(size, spacing)

            is_valid = len(reasons) == 0
            res = ValidationResult(name, is_valid, reasons, img_path)

            # Always return metadata if image can be read
            return SeriesMetadata.from_sitk_image(img, name), res

        except Exception as e:
            res = ValidationResult(name, False, [f"Failed to read: {str(e)}"], img_path)
            return None, res

    def _collect_results(self, results: list):
        """Collect both metadata and validation results from workers"""
        for meta, res in results:
            if res:
                self.results.append(res)
            if meta:
                self.meta_manager.update(meta, allow_and_overwrite_existed=self.ALLOW_AND_OVERWRITE_EXISTED_METADATA)

    def process(self, desc="Checking"):
        """Main processing with fast check support"""
        # Try fast check first
        series_meta = load_series_meta(self.source_folder)
        run_full_check = False

        if series_meta is not None:
            print("Found existing meta.json, performing fast check.")
            items = self.get_items_to_process()
            item_dict = {os.path.basename(img): img for img in items}
            self.fast_check_with_meta(series_meta, item_dict)
        else:
            print("No meta.json found, performing full check.")
            super().process(desc)
            run_full_check = True

        # Execute operations based on mode
        self.execute_operation(self.output_dir)

        # Save metadata to source folder if full check was run
        if run_full_check:
            meta_path = get_series_meta_path(self.source_folder)
            self.save_meta(meta_path)

    def prepare_output_dir(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

    def op_delete(self, paths: Any):
        if os.path.exists(paths): os.remove(paths)

    def op_symlink(self, name: str, paths: Any, output_dir: str):
        dst = os.path.join(output_dir, name)
        target = os.path.relpath(os.path.abspath(paths), os.path.dirname(os.path.abspath(dst)))
        os.symlink(target, dst)

    def op_copy(self, name: str, paths: Any, output_dir: str):
        shutil.copy(paths, output_dir)


def detect_processor_type(source_folder: str) -> ProcessorType:
    """Detect processor type based on folder structure"""
    img_dir = os.path.join(source_folder, 'image')
    lbl_dir = os.path.join(source_folder, 'label')
    if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
        return ProcessorType.DATASET
    else:
        return ProcessorType.SINGLE


def CheckProcessor(
    source_folder: str,
    cfg: dict,
    mode: str,
    output_dir: str | None = None,
    mp: bool = False,
    workers: int | None = None,
):
    """Factory function to create appropriate check processor"""
    processor_type = detect_processor_type(source_folder)
    if processor_type == ProcessorType.DATASET:
        return DatasetCheckProcessor(source_folder, cfg, mode, output_dir, mp, workers)
    else:
        return SingleCheckProcessor(source_folder, cfg, mode, output_dir, mp, workers)


def main():
    parser = argparse.ArgumentParser(description="ITK Dataset/Sample Checker")

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
    processor = CheckProcessor(
        source_folder = args.sample_folder,
        cfg = cfg,
        mode = args.mode,
        output_dir = args.output,
        mp = args.mp,
        workers = args.workers,
    )

    # Process and save metadata
    processor.process("Checking")

    print("Check completed.")


if __name__ == "__main__":
    main()
