import pdb
from abc import abstractmethod
from collections.abc import Callable
from typing_extensions import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning import LightningDataModule

from itkit.lightning.utils import multi_sample_collate



class BaseDataModule(LightningDataModule):
    TRAIN_LOADER_ARGS = {
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "collate_fn": multi_sample_collate,
        "persistent_workers": True,
    }
    VAL_TEST_LOADER_ARGS = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": True,
        "collate_fn": multi_sample_collate,
        "persistent_workers": True,
    }

    def __init__(self,
                 train_loader_args = {},
                 val_test_loader_args = {},
                 override_transfer_batch_to_device: torch.device | None = None,
                 **kwargs):
        """
        Args:
            train_loader_args (dict):
                arguments for the training dataloader
            val_test_loader_args (dict):
                arguments for the validation and test dataloaders
            override_transfer_batch_to_device (torch.device):
                Override the device to transfer batch, 
                maybe useful when wants to ignore lightning automatic data transfer after DataLoader.
        """        
        self.train_loader_args = {**self.TRAIN_LOADER_ARGS, **train_loader_args}
        self.val_test_loader_args = {**self.VAL_TEST_LOADER_ARGS, **val_test_loader_args}
        self.override_transfer_batch_to_device = override_transfer_batch_to_device
        super().__init__(**kwargs)

    def prepare_data(self):
        """
        download, IO, etc. Useful with shared filesystems
        only called on 1 GPU/TPU in distributed
        """

    def teardown(self, stage: Literal['fit', 'validate', 'test', 'predict']) -> None:
        """
        clean up state after the trainer stops, delete files...
        called on every process in DDP
        """

    def on_exception(self, exception):
        """ clean up state after the trainer faced an exception """

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if self.override_transfer_batch_to_device is not None:
            return super().transfer_batch_to_device(batch, self.override_transfer_batch_to_device, dataloader_idx)
        else:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def train_dataloader(self) -> DataLoader:
        self.setup('fit')
        return DataLoader(self.train, **self.train_loader_args)

    def val_dataloader(self) -> DataLoader:
        self.setup('validate')
        return DataLoader(self.val, **self.val_test_loader_args)

    def test_dataloader(self) -> DataLoader:
        self.setup('test')
        return DataLoader(self.test, **self.val_test_loader_args)

class UniversalDataModule(BaseDataModule):
    def __init__(self, dataset: Dataset, **kwargs):
        self.dataset = dataset
        super().__init__(**kwargs)

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        train_end_idx = int(len(self.dataset) * self.dataset.SPLIT_RATIO[0])
        val_end_idx = train_end_idx + max(1, int(len(self.dataset) * self.dataset.SPLIT_RATIO[1]))
        test_end_idx = len(self.dataset)

        if stage == 'predict':
            self.train = self.val = self.test = self.dataset
        else:
            self.train = Subset(self.dataset, range(train_end_idx))
            self.val = Subset(self.dataset, range(train_end_idx, val_end_idx))
            self.test = Subset(self.dataset, range(val_end_idx, test_end_idx))


class BaseDataset(Dataset):
    SPLIT_RATIO = (0.7, 0.05, 0.25)  # train, val, test

    def __init__(self,
                 split: Literal['train', 'val', 'test'] | None = None,
                 pipeline: list[Callable] = [],
                 debug: bool = False):
        super().__init__()
        self.split = split
        self.pipeline = pipeline
        self.debug = debug
        assert self.split is None, "The dataset split for Lightning is implemented in DataModule, not in Dataset. " \
                                   f"Please set split to None, got {self.split}."

    def _preprocess(self, sample:dict):
        for transform in self.pipeline:
            sample = transform(sample)
        return sample
    
    @abstractmethod
    def __getitem__(self, index) -> dict:
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        ...
