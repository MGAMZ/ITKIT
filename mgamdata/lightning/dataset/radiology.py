import os
import re
from typing import Literal
from pathlib import Path

from .base import BaseDataset


class MhaDataset(BaseDataset):
    SPLIT_RATIO = (0.7, 0.05, 0.25)  # train, val, test
    DEFAULT_ORIENTATION = "LPI"

    def __init__(
        self,
        data_root: str | Path,
        split_accordance: str | Path | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.data_root = Path(data_root)
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

    def __getitem__(self, index):
        series_uid = self.all_series_uids[index]
        sample = {
            "series_uid": series_uid,
            "image_mha_path": str(self.data_root / "image" / f"{series_uid}.mha"),
            "label_mha_path": str(self.data_root / "label" / f"{series_uid}.mha")
        }
        return self._preprocess(sample)

    def __len__(self):
        return len(self.all_series_uids) if self.debug is False else min(10, len(self.all_series_uids))
