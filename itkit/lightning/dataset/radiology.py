import os
import pdb
import re
from copy import deepcopy
from pathlib import Path
from typing import Literal

from deprecated import deprecated
from tqdm import tqdm

from .base import BaseDataModule, BaseDataset


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
        self.all_seriesUID: list[str] | None = None
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.split_accordance = Path(split_accordance)

    def _split(self) -> list[str]:
        if not self.split_accordance.exists():
            raise FileNotFoundError(f"Split accordance directory not found: {self.split_accordance}")
        if self.all_seriesUID is None:
            all_seriesUID = [file.stem for file in self.split_accordance.glob("*.mha")]
            all_seriesUID = sorted(all_seriesUID, key=lambda x: abs(int(re.search(r"\d+", x).group())))
            self.all_seriesUID = all_seriesUID
        
        assert self.all_seriesUID is not None
        split_id_train_val = int(len(self.all_seriesUID) * self.SPLIT_RATIO[0])
        split_id_val_test = int(len(self.all_seriesUID) * (self.SPLIT_RATIO[0] + self.SPLIT_RATIO[1]))
        
        if self.split in ('train', 'fit'):
            return self.all_seriesUID[:split_id_train_val]
        elif self.split in ('val', 'validate'):
            return self.all_seriesUID[split_id_train_val:split_id_val_test+1]
        elif self.split in ('test', ):
            return self.all_seriesUID[split_id_val_test:]
        elif self.split in ('all', 'predict', 'none', None):
            return self.all_seriesUID
        else:
            raise ValueError(f"Invalid split: {self.split}.")

    def index_dataset(self):
        available_file_names = [
            f.name
            for seriesUID in self._split()
            for f in self.image_root.glob(f"*{seriesUID}*.mha")
        ]
        
        self.available_series = []
        for avail_file_name in tqdm(available_file_names, f"Indexing Dataset | Split {self.split}"):
            image_mha_path = str(self.image_root / avail_file_name)
            label_mha_path = str(self.label_root / avail_file_name)
            self.available_series.append({
                "series_uid": avail_file_name.split(".")[0], # without `.mha`
                "image_mha_path": image_mha_path,
                "label_mha_path": label_mha_path
            })

    def get_self_copy(self, split: str | None):
        dataset_copy = deepcopy(self)
        dataset_copy.split = split
        dataset_copy.index_dataset()
        return dataset_copy

    def __getitem__(self, index):
        return self._preprocess(self.available_series[index].copy())

    def __len__(self):
        return 20 if self.debug else len(self.available_series)

@deprecated("The patched dataset structure now align to common MhaDataset, so there're no need to use this class.")
class MhaPatchedDataset(MhaDataset):
    def index_dataset(self):
        splited_series = set(self._split())
        existed_series = [f for f in os.listdir(self.image_root) if os.path.isdir(self.image_root / f)]
        self.available_series = []
        for series in tqdm(splited_series.intersection(existed_series),
                           desc=f"Indexing Dataset | Split {self.split}"):
            for mha in (self.image_root / series).glob("*_image.mha"):
                image_mha_path = str(self.image_root / series / mha.name)
                label_mha_path = image_mha_path.replace("_image.mha", "_label.mha")
                self.available_series.append({
                    "series_uid": series,
                    "image_mha_path": image_mha_path,
                    "label_mha_path": label_mha_path
                })


class LargeVolumeDataModule(BaseDataModule):
    def __init__(self,
                 patched_train_dataset: MhaDataset | None = None,
                 whole_volume_val_dataset: MhaDataset | None = None,
                 whole_volume_test_dataset: MhaDataset | None = None,
                 **kwargs):
        self.patched_train_dataset = patched_train_dataset
        self.whole_volume_val_dataset = whole_volume_val_dataset
        self.whole_volume_test_dataset = whole_volume_test_dataset
        super().__init__(**kwargs)
    
    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        if stage == 'fit':
            assert self.patched_train_dataset is not None, "Patched train dataset must be provided for training."
            self.train = self.patched_train_dataset.get_self_copy(stage)
        elif stage == 'validate':
            assert self.whole_volume_val_dataset is not None, "Whole volume val dataset must be provided for validation."
            self.val = self.whole_volume_val_dataset.get_self_copy(stage)
        elif stage in ('test', 'predict'):
            assert self.whole_volume_test_dataset is not None, "Whole volume test dataset must be provided for testing."
            self.test = self.whole_volume_test_dataset.get_self_copy(stage)
        else:
            raise ValueError(f"Invalid stage: {stage}.")
