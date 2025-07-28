import os
import re
from typing import Literal
from pathlib import Path
from tqdm import tqdm

from .base import BaseDataset


class MhaDataset(BaseDataset):
    SPLIT_RATIO = (0.7, 0.05, 0.25)  # train, val, test
    DEFAULT_ORIENTATION = "LPI"

    def __init__(
        self,
        image_root: str | Path,
        label_root: str | Path,
        split_accordance: str | Path,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.split_accordance = Path(split_accordance)
        self.index_dataset()

    def _split(self) -> list[str]:
        if not self.split_accordance.exists():
            raise FileNotFoundError(f"Split accordance directory not found: {self.split_accordance}")
        all_series = [file.stem for file in self.split_accordance.glob("*.mha")]
        all_series = sorted(all_series, key=lambda x: abs(int(re.search(r"\d+", x).group())))
        
        split_id_train_val = int(len(all_series) * self.SPLIT_RATIO[0])
        split_id_val_test = int(len(all_series) * (self.SPLIT_RATIO[0] + self.SPLIT_RATIO[1]))
        
        if self.split == 'train':
            return all_series[:split_id_train_val]
        elif self.split == 'val':
            return all_series[split_id_train_val:split_id_val_test+1]
        elif self.split == 'test':
            return all_series[split_id_val_test:]
        elif self.split == 'all' or self.split is None:
            return all_series
        else:
            raise ValueError(f"Invalid split: {self.split}. Choose from 'train', 'val', 'test', or 'all'.")

    def index_dataset(self):
        splited_series = set(self._split())
        existed_series = set([f.stem for f in self.image_root.glob("*.mha")])
        self.available_series = []

        for series in tqdm(splited_series.intersection(existed_series),
                           desc=f"Indexing Dataset | Split {self.split}"):
            image_mha_path = str(self.image_root / f"{series}.mha")
            label_mha_path = str(self.label_root / f"{series}.mha")
            if not os.path.exists(image_mha_path):
                print(f"Warning: {series} image mha file not found. Full path: {image_mha_path}")
                continue
            self.available_series.append({
                "series_uid": series,
                "image_mha_path": image_mha_path,
                "label_mha_path": label_mha_path
            })

    def __getitem__(self, index):
        return self._preprocess(self.available_series[index])

    def __len__(self):
        return 10 if self.debug else len(self.available_series)
