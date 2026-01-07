import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydantic import BaseModel, Field, field_validator


class SeriesMetadata(BaseModel):
    """
    Metadata for a single image file.

    Attributes:
        name: Filename or series UID
        spacing: Image spacing in (Z, Y, X) order
        size: Image size in (Z, Y, X) order
        origin: Image origin in (Z, Y, X) order
    """

    name: str = Field(..., description="Filename or series UID")
    spacing: tuple[float, float, float] = Field(..., description="Image spacing (Z, Y, X)")
    size: tuple[int, int, int] = Field(..., description="Image size (Z, Y, X)")
    origin: tuple[float, float, float] = Field(..., description="Image origin (Z, Y, X)")
    include_classes: tuple[int, ...] | None = Field(None, description="Classes to include in processing")

    @classmethod
    def from_sitk_image(cls, image: sitk.Image, name: str) -> 'SeriesMetadata':
        # `sitkUInt8` is treated as label image with possible classes 0-255
        if image.GetPixelID() == sitk.sitkUInt8:
            img_arr = sitk.GetArrayFromImage(image)
            include_classes = np.unique(img_arr).tolist()
        else:
            include_classes = None

        return cls(
            name=name,
            spacing=tuple(image.GetSpacing()[::-1]),
            size=tuple(image.GetSize()[::-1]),
            origin=tuple(image.GetOrigin()[::-1]),
            include_classes=include_classes
        )

    @field_validator('spacing', mode='before')
    @classmethod
    def validate_spacing(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(float(x) for x in v)
        raise ValueError("spacing must be a list or tuple")

    @field_validator('size', mode='before')
    @classmethod
    def validate_size(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        raise ValueError("size must be a list or tuple")

    @field_validator('origin', mode='before')
    @classmethod
    def validate_origin(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(float(x) for x in v)
        raise ValueError("origin must be a list or tuple")

    def validate_itk_image(self, image: sitk.Image) -> bool:
        # Get image properties (XYZ order)
        img_spacing = image.GetSpacing()
        img_size = image.GetSize()
        img_origin = image.GetOrigin()

        # Convert to ZYX order for comparison
        img_spacing_zyx = tuple(img_spacing[::-1])
        img_size_zyx = tuple(img_size[::-1])
        img_origin_zyx = tuple(img_origin[::-1])

        # Validate spacing
        if not np.allclose(img_spacing_zyx, self.spacing, rtol=1e-5):
            raise ValueError(f"Spacing mismatch: expected {self.spacing}, got {img_spacing_zyx}")

        # Validate size
        if img_size_zyx != self.size:
            raise ValueError(f"Size mismatch: expected {self.size}, got {img_size_zyx}")

        # Validate origin
        if not np.allclose(img_origin_zyx, self.origin, rtol=1e-5):
            raise ValueError(f"Origin mismatch: expected {self.origin}, got {img_origin_zyx}")

        return True


class MetadataManager:
    def __init__(self, meta_file_path:str|Path|None=None):
        if (meta_file_path is None) or (not Path(meta_file_path).exists()):
            self.meta: dict[str, SeriesMetadata] = {}
        else:
            try:
                data = json.loads(Path(meta_file_path).read_text())
                self.meta = {
                    name: SeriesMetadata.model_validate({"name": name, **meta})
                    for name, meta in data.items()
                }
            except (json.JSONDecodeError, Exception):
                # If file is corrupted or invalid, start with empty metadata
                self.meta = {}

    @property
    def series_uids(self) -> list[str]:
        return list(self.meta.keys())

    def update(self, image_meta:SeriesMetadata, allow_and_overwrite_existed:bool=True):
        if (image_meta.name not in self.meta) or allow_and_overwrite_existed:
            self.meta[image_meta.name] = image_meta
        elif self.meta[image_meta.name] != image_meta:
            raise ValueError(f"Metadata for {image_meta.name} already exists and differs.\n"
                             f"`image_meta`: {image_meta}\n"
                             f"`Existed`: {self.meta[image_meta.name]}")
        else:
            pass

    def load_and_merge(self, meta_file_path: str | Path, allow_and_overwrite_existed: bool = False):
        """Load metadata from a file and merge it with existing metadata.
        
        Args:
            meta_file_path: Path to the metadata file to load
            allow_and_overwrite_existed: If True, overwrite existing metadata with loaded data.
                                         If False (default), keep existing metadata when conflicts occur.
        """
        if not Path(meta_file_path).exists():
            return
        
        try:
            data = json.loads(Path(meta_file_path).read_text())
            for name, meta_dict in data.items():
                meta = SeriesMetadata.model_validate({"name": name, **meta_dict})
                self.update(meta, allow_and_overwrite_existed=allow_and_overwrite_existed)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load metadata from {meta_file_path}: {e}")

    def save(self, path: str|Path):
        data = {
            name: meta.model_dump(mode="json", exclude={'name'})
            for name, meta in self.meta.items()
        }
        Path(path).write_text(json.dumps(data, indent=4))
