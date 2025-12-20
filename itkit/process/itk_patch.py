import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydantic import BaseModel, Field
from tqdm import tqdm

from itkit.process.base_processor import DatasetProcessor
from itkit.process.metadata_models import SeriesMetadata


class PatchMetadata(BaseModel):
    """Metadata for a single source series that was patched."""
    series_id: str = Field(..., description="Source series identifier")
    shape: tuple[int, int, int] = Field(..., description="Original image shape (Z, Y, X)")
    num_patches: int = Field(..., description="Number of patches extracted")
    anno_available: bool = Field(True, description="Whether annotations are available")
    class_within_patch: dict[str, list[int]] = Field(default_factory=dict, description="Mapping patch filename to unique classes")


class ProcessOneResult(BaseModel):
    """Result from processing one image-label pair.

    Contains both the patch-level metadata (for meta.json) and
    source-level metadata (for crop_meta.json).
    """
    patch_metadata_list: list[SeriesMetadata] = Field(default_factory=list, description="Metadata for each extracted patch")
    source_metadata: PatchMetadata = Field(..., description="Metadata about the source series")


class CropMetadata(BaseModel):
    """Overall metadata for patch extraction operation."""
    src_folder: str = Field(..., description="Source dataset folder")
    dst_folder: str = Field(..., description="Destination folder for patches")
    patch_size: list[int] = Field(..., description="Patch size (Z, Y, X)")
    patch_stride: list[int] = Field(..., description="Patch stride (Z, Y, X)")
    anno_available: list[str] = Field(default_factory=list, description="List of series IDs with available annotations")
    patch_meta: dict[str, PatchMetadata] = Field(default_factory=dict, description="Per-series patch metadata")

    def save(self, path: str | Path):
        """Save crop metadata to JSON file."""
        data = self.model_dump(mode="json")
        Path(path).write_text(json.dumps(data, indent=4))

    @classmethod
    def load(cls, path: str | Path) -> 'CropMetadata':
        """Load crop metadata from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)


def parse_patch_size(patched_dataset_folder: str | Path) -> list[int]:
    """Parse patch size from crop metadata file.

    Args:
        patched_dataset_folder: Path to the patched dataset folder

    Returns:
        Patch size as [Z, Y, X]

    Raises:
        FileNotFoundError: If crop_meta.json doesn't exist
    """
    patched_dataset_meta = Path(patched_dataset_folder) / 'crop_meta.json'
    if not patched_dataset_meta.exists():
        raise FileNotFoundError(
            f"Patched dataset meta file not found: {patched_dataset_meta}, "
            f"cannot determine patch size."
        )

    crop_meta = CropMetadata.load(patched_dataset_meta)
    return crop_meta.patch_size


class PatchProcessor(DatasetProcessor):
    def __init__(self,
                 source_folder: Path | str,
                 dst_folder: Path | str,
                 patch_size: int | list[int],
                 patch_stride: int | list[int],
                 min_fg: float,
                 keep_empty_label_prob: float,
                 still_save: bool,
                 mp: bool = False,
                 workers: int | None = None):
        super().__init__(str(source_folder), str(dst_folder), mp=mp, workers=workers)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.min_fg = min_fg
        self.keep_empty_label_prob = keep_empty_label_prob
        self.still_save = still_save
        # Prepare global image/ and label/ output directories under destination
        assert self.dest_folder is not None
        self.image_dir = Path(self.dest_folder) / "image"
        self.label_dir = Path(self.dest_folder) / "label"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        # Track source-level metadata (collected after processing)
        self.source_metadata: dict[str, PatchMetadata] = {}

    def _collect_results(self, results: list):
        # Collect metadata from the results
        for res in results:
            if res and isinstance(res, ProcessOneResult):
                # Collect patch-level metadata for meta.json
                for patch_meta in res.patch_metadata_list:
                    self.meta_manager.update(patch_meta, allow_and_overwrite_existed=self.ALLOW_AND_OVERWRITE_EXISTED_METADATA)

                # Collect source-level metadata for crop_meta.json
                self.source_metadata[res.source_metadata.series_id] = res.source_metadata

    def extract_patches(self,
                        image: sitk.Image,
                        label: sitk.Image | None,
                        patch_size: int | list[int],
                        patch_stride: int | list[int],
                        minimum_foreground_ratio: float,
                        still_save_when_no_label: bool) -> list[tuple[sitk.Image, sitk.Image | None]]:
        if label is not None:
            if image.GetSize() != label.GetSize():
                raise ValueError(f"Image ({image.GetSize()}) and label ({label.GetSize()}) must have the same size.")

        img_arr = sitk.GetArrayFromImage(image)
        if label is None:
            lbl_arr = None
        else:
            lbl_arr = sitk.GetArrayFromImage(label)

        def to_triplet(x):
            if isinstance(x, int):
                return (x, x, x)
            if isinstance(x, (list, tuple)) and len(x) == 3:
                return tuple(x)
            raise ValueError('patch_size and patch_stride must be int or 3-length list/tuple')
        pZ, pY, pX = to_triplet(patch_size)
        sZ, sY, sX = to_triplet(patch_stride)
        Z, Y, X = img_arr.shape
        if pZ > Z or pY > Y or pX > X:
            return []

        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()

        def compute_starts(L, p, s):
            starts = list(range(0, L - p + 1, s))
            if starts[-1] != L - p:
                starts.append(L - p)
            return starts
        z_starts = compute_starts(Z, pZ, sZ)
        y_starts = compute_starts(Y, pY, sY)
        x_starts = compute_starts(X, pX, sX)

        patches = []
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    img_patch_np = img_arr[z:z+pZ, y:y+pY, x:x+pX]
                    save = True
                    if label is None:
                        if not still_save_when_no_label:
                            save = False
                        lbl_patch_np = None
                    else:
                        assert lbl_arr is not None
                        lbl_patch_np = lbl_arr[z:z+pZ, y:y+pY, x:x+pX]
                        fg_ratio = np.sum(lbl_patch_np > 0) / lbl_patch_np.size
                        if (fg_ratio < minimum_foreground_ratio):
                            save = False
                        if fg_ratio == 0 and (np.random.rand() > self.keep_empty_label_prob):
                            save = False

                    if save:
                        new_origin = (
                            origin[0] + x * spacing[0],
                            origin[1] + y * spacing[1],
                            origin[2] + z * spacing[2]
                        )
                        img_patch = sitk.GetImageFromArray(img_patch_np)
                        img_patch.SetOrigin(new_origin)
                        img_patch.SetSpacing(spacing)
                        img_patch.SetDirection(direction)
                        assert (img_patch_size := img_patch.GetSize()[::-1]) == (pZ, pY, pX), (
                            f"Unexpected image patch shape: {img_patch_size}, expected {(pZ, pY, pX)}. "
                            f"Current Patch origin pixel z:{z}, y:{y}, x:{x}, Series image size: {img_arr.shape}"
                        )

                        if label is None:
                            lbl_patch = None
                        else:
                            assert lbl_patch_np is not None
                            assert lbl_arr is not None
                            lbl_patch = sitk.GetImageFromArray(lbl_patch_np)
                            lbl_patch.SetOrigin(new_origin)
                            lbl_patch.SetSpacing(spacing)
                            lbl_patch.SetDirection(direction)
                            assert lbl_patch_np is None or (lbl_patch_size := lbl_patch.GetSize()[::-1]) == (pZ, pY, pX), (
                                f"Unexpected label patch shape: {lbl_patch_size}, expected {(pZ, pY, pX)}"
                                f"Current Patch origin pixel z:{z}, y:{y}, x:{x}, Series label size: {lbl_arr.shape}"
                            )

                        patches.append((img_patch, lbl_patch))

        return patches

    def process_one(self, args: tuple[str, str]) -> ProcessOneResult | None:
        """Process one image-label pair and extract patches.

        Args:
            args: Tuple of (image_path, label_path)

        Returns:
            ProcessOneResult containing patch metadata list and source metadata,
            or None if processing failed
        """
        img_path, lbl_path = args
        case_name = os.path.basename(self._normalize_filename(img_path))

        try:
            image = sitk.ReadImage(str(img_path))
            label = sitk.ReadImage(str(lbl_path))
            image = sitk.DICOMOrient(image, 'LPI')
            label = sitk.DICOMOrient(label, 'LPI')

            img_arr = sitk.GetArrayFromImage(image)

            if not self.is_valid_sample(image, label):
                return None

            patches = self.extract_patches(image, label, self.patch_size, self.patch_stride, self.min_fg, self.still_save)

            patch_metadata_list = []
            class_within_patch = {}

            # Save patches and collect metadata for each patch
            for idx, (img_patch, lbl_patch) in enumerate(patches):
                # Use unified filenames across image/ and label/ dirs so they correspond 1:1
                fname_base = f"{case_name}_p{idx}.mha"

                # Save image patch
                sitk.WriteImage(img_patch, str(self.image_dir / fname_base), True)

                if lbl_patch is not None:
                    # Log unique classes in this patch
                    lbl_arr = sitk.GetArrayFromImage(lbl_patch)
                    class_within_patch[fname_base] = np.unique(lbl_arr).tolist()
                    # Save label patch
                    sitk.WriteImage(lbl_patch, str(self.label_dir / fname_base), True)

                    # Create metadata for this patch from the label
                    patch_meta = SeriesMetadata.from_sitk_image(lbl_patch, fname_base)
                    patch_metadata_list.append(patch_meta)

            # Create source-level metadata using PatchMetadata model
            source_meta = PatchMetadata(
                series_id=case_name,
                shape=tuple(img_arr.shape),
                num_patches=len(patches),
                anno_available=True,
                class_within_patch=class_within_patch
            )

            # Return combined result
            return ProcessOneResult(
                patch_metadata_list=patch_metadata_list,
                source_metadata=source_meta
            )

        except Exception as e:
            tqdm.write(f"Failed processing case {case_name}: {e}")
            return None

    def is_valid_sample(self, itk_img: sitk.Image, itk_lbl: sitk.Image) -> bool:
        img_size = itk_img.GetSize()
        lbl_size = itk_lbl.GetSize()
        if not np.allclose(img_size, lbl_size, atol=1.5):
            tqdm.write(f"Skipping for Size mismatch img size {img_size} | lbl size {lbl_size}.")
            return False

        img_spacing = itk_img.GetSpacing()
        lbl_spacing = itk_lbl.GetSpacing()
        if not np.allclose(img_spacing, lbl_spacing, atol=0.01):
            tqdm.write(f"Skipping for Spacing mismatch img spacing {img_spacing} | lbl spacing {lbl_spacing}.")
            return False

        if img_size[0] != img_size[1]:
            tqdm.write(f"Skipping for Non-isotropic size img size {img_size}.")
            return False

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Extract patches from a folder of MHA images")
    parser.add_argument('src_folder', type=Path,
                        help='Folder containing `image` and `label` subfolders')
    parser.add_argument('dst_folder', type=Path,
                        help='Destination root folder to save patches')
    parser.add_argument('--patch-size', type=int, nargs='+', required=True,
                        help='Patch size as int or three ints (Z Y X)')
    parser.add_argument('--patch-stride', type=int, nargs='+', required=True,
                        help='Patch stride as int or three ints (Z Y X)')
    parser.add_argument('--minimum-foreground-ratio', type=float, default=0.0,
                        help='Minimum label foreground ratio to keep patch')
    parser.add_argument('--keep-empty-label-prob', type=float, default=1.0,
                         help='Probability to keep patches that contain only background (0.0-1.0)')
    parser.add_argument('--still-save-when-no-label', action='store_true',
                        help='If label missing, still extract patches unconditionally')
    parser.add_argument('--mp', action='store_true',
                        help='Use multiprocessing to process cases')
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")
    return parser.parse_args()


def main():
    args = parse_args()

    processor = PatchProcessor(
        source_folder = args.src_folder,
        dst_folder = args.dst_folder,
        patch_size = args.patch_size,
        patch_stride = args.patch_stride,
        min_fg = args.minimum_foreground_ratio,
        keep_empty_label_prob = args.keep_empty_label_prob,
        still_save = args.still_save_when_no_label,
        mp = args.mp,
        workers = args.workers
    )

    try:
        processor.process("Patching")

        # Save standard metadata files (only contains patch-level metadata)
        # These meta.json files contain SeriesMetadata for each patch
        processor.save_meta(args.dst_folder / "meta.json")
        processor.save_meta(args.dst_folder / "image" / "meta.json")
        processor.save_meta(args.dst_folder / "label" / "meta.json")

        # Create and save CropMetadata (contains source-level information)
        crop_meta = CropMetadata(
            src_folder=str(args.src_folder),
            dst_folder=str(args.dst_folder),
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            anno_available=list(processor.source_metadata.keys()),
            patch_meta=processor.source_metadata
        )
        crop_meta.save(args.dst_folder / "crop_meta.json")

        print(f"Patching completed. Results saved to {args.dst_folder}")
        print(f"  - Processed {len(processor.source_metadata)} cases")
        print(f"  - Generated {len(processor.meta_manager.meta)} patches")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during processing: {e}")


if __name__ == '__main__':
    main()
