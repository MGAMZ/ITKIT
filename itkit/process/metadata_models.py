import pdb, json
from pathlib import Path
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


class MetadataManager:
    def __init__(self, meta_file_path:str|Path|None=None):
        if (meta_file_path is None) or (not Path(meta_file_path).exists()):
            self.meta: dict[str, SeriesMetadata] = {}
        else:
            data = json.loads(Path(meta_file_path).read_text())
            self.meta = {
                name: SeriesMetadata.model_validate({"name": name, **meta})
                for name, meta in data.items()
            }
    
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
    
    def save(self, path: str|Path):
        data = {
            name: meta.model_dump(mode="json", exclude={'name'})
            for name, meta in self.meta.items()
        }
        Path(path).write_text(json.dumps(data, indent=4))
