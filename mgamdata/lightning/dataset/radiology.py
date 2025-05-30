import os
import re
from typing import Literal
from pathlib import Path
from .base import LightningBaseDataset


class LightningSeriesVolume(LightningBaseDataset):
    """
    Base class for series volume datasets.
    
    This class handles datasets organized as series of 3D volumes,
    supporting both supervised and semi-supervised learning modes.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        data_root_mha: str | Path | None = None,
        mode: Literal["semi", "sup"] = "sup",
        split: str | None = None,
        **kwargs
    ) -> None:
        """
        Initialize series volume dataset.
        
        Args:
            data_root: Root directory of the dataset
            data_root_mha: Root directory for MHA files (defaults to data_root)
            mode: Learning mode - 'semi' includes samples without labels, 
                  'sup' excludes samples without labels
            split: Dataset split ('train', 'val', 'test', 'all', or None)
        """
        self.mode = mode
        self.data_root_mha = Path(data_root_mha) if data_root_mha else Path(data_root)
        
        super().__init__(data_root=data_root, split=split, **kwargs)
        
        if data_root_mha is None:
            print(f"data_root_mha is not specified, using data_root: {self.data_root_mha}")
    
    def _split(self) -> list[str]:
        """
        Split the dataset into train/val/test sets.
        
        Returns:
            List of series names for the current split
        """
        split_at = "label" if self.mode == "sup" else "image"
        mha_dir = self.data_root_mha / split_at
        
        if not mha_dir.exists():
            raise FileNotFoundError(f"MHA directory not found: {mha_dir}")
        
        all_series = [
            file.stem  # Remove .mha extension
            for file in mha_dir.glob("*.mha")
        ]
        
        # Sort by numeric value in filename
        all_series = sorted(all_series, key=lambda x: abs(int(re.search(r"\d+", x).group())))
        
        train_end = int(len(all_series) * self.SPLIT_RATIO[0])
        val_end = train_end + int(len(all_series) * self.SPLIT_RATIO[1]) + 1
        
        if self.split == "train":
            return all_series[:train_end]
        elif self.split == "val":
            return all_series[train_end:val_end]
        elif self.split == "test":
            return all_series[val_end:]
        elif self.split == "all" or self.split is None:
            return all_series
        else:
            raise ValueError(f"Unsupported split: {self.split}")


class LightningSemiSup3DMha(LightningSeriesVolume):
    """
    PyTorch Lightning compatible dataset for 3D MHA files in semi-supervised learning.
    
    This class handles 3D MHA format medical images, supporting both supervised
    and semi-supervised learning modes. It's designed to work seamlessly with
    PyTorch Lightning data modules.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        data_root_mha: str | Path | None = None,
        mode: Literal["semi", "sup"] = "sup",
        split: str | None = None,
        img_suffix: str = ".mha",
        **kwargs
    ) -> None:
        """
        Initialize the 3D MHA semi-supervised dataset.
        
        Args:
            data_root: Root directory of the dataset
            data_root_mha: Root directory for MHA files (defaults to data_root)
            mode: Learning mode - 'semi' for semi-supervised, 'sup' for supervised
            split: Dataset split ('train', 'val', 'test', 'all', or None)
            img_suffix: File suffix for MHA files
        """
        super().__init__(
            data_root=data_root,
            data_root_mha=data_root_mha,
            mode=mode,
            split=split,
            img_suffix=img_suffix,
            **kwargs
        )
    
    def sample_iterator(self):
        """
        Iterate over all samples in the dataset.
        
        Yields:
            Tuple of (image_mha_path, label_mha_path)
        """
        for series in self._split():
            image_mha_path = str(self.data_root / "image" / f"{series}.mha")
            label_mha_path = str(self.data_root / "label" / f"{series}.mha")
            
            if not os.path.exists(image_mha_path):
                print(f"Warning: {series} image mha file not found. Full path: {image_mha_path}")
                continue
                
            yield (image_mha_path, label_mha_path)
