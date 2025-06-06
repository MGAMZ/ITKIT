import os
import re
from typing import Literal
from pathlib import Path

import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset


class MhaDataset(Dataset):
    SPLIT_RATIO = (0.7, 0.05, 0.25)  # train, val, test
    DEFAULT_ORIENTATION = "LPI"

    def __init__(
        self,
        data_root: str | Path,
        debug: bool = False,
        split_accordance: str | Path | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.data_root = Path(data_root)
        self.debug = debug
        self.split_accordance = Path(split_accordance) if split_accordance else Path(data_root)
        self.all_series_uids = self._search_series()

    def _search_series(self) -> list[str]:
        mha_dir = self.split_accordance / "label"
        if not mha_dir.exists():
            raise FileNotFoundError(f"MHA directory not found: {mha_dir}")
        all_series = [file.stem for file in mha_dir.glob("*.mha")]
        return sorted(all_series, key=lambda x: abs(int(re.search(r"\d+", x).group())))

    def sample_iterator(self, subset: Literal['train', 'val', 'test', 'all']):
        for series in self.all_series_uids:
            image_mha_path = str(self.data_root / "image" / f"{series}.mha")
            label_mha_path = str(self.data_root / "label" / f"{series}.mha")
            if not os.path.exists(image_mha_path):
                print(f"Warning: {series} image mha file not found. Full path: {image_mha_path}")
                continue
            yield (image_mha_path, label_mha_path)

    def _preprocess(self, image:np.ndarray, label:np.ndarray):
        image = np.clip(image, 0-300, 0+300)
        image = (image + 300) / 600
        return image[None], label

    def __getitem__(self, index):
        series_uid = self.all_series_uids[index]
        image_mha_path = str(self.data_root / "image" / f"{series_uid}.mha")
        label_mha_path = str(self.data_root / "label" / f"{series_uid}.mha")
        
        image_mha = sitk.ReadImage(image_mha_path)
        label_mha = sitk.ReadImage(label_mha_path)
        image_arr = sitk.GetArrayFromImage(image_mha)
        label_arr = sitk.GetArrayFromImage(label_mha)
        image_arr, label_arr = self._preprocess(image_arr, label_arr)
        
        return {
            "image": torch.from_numpy(image_arr).to(torch.float32),
            "label": torch.from_numpy(label_arr).to(torch.uint8),
            "series_uid": series_uid,
            "image_mha_path": image_mha_path,
            "label_mha_path": label_mha_path
        }

    def __len__(self):
        return len(self.all_series_uids) if self.debug is False else min(10, len(self.all_series_uids))
