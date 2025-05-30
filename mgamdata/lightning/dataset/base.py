from abc import abstractmethod
from collections.abc import Generator
from typing import Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class LightningBaseDataset(Dataset):
    """
    PyTorch Lightning compatible base dataset class for medical imaging.
    
    This class provides the basic structure for medical imaging datasets,
    including data splitting and sample iteration functionality.
    """
    
    SPLIT_RATIO = [0.8, 0.05, 0.15]  # train, val, test
    
    def __init__(
        self,
        data_root: str | Path,
        split: str | None = None,
        debug: bool = False,
        dataset_name: str | None = None,
        transform: Any | None = None,
        **kwargs
    ) -> None:
        """
        Initialize the base dataset.
        
        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test', 'all', or None)
            debug: If True, only use 16 samples for debugging
            dataset_name: Name of the dataset (defaults to class name)
            transform: Data transforms to apply
        """
        self.data_root = Path(data_root)
        self.split = split
        self.debug = debug
        self.dataset_name = dataset_name or self.__class__.__name__
        self.transform = transform
        
        # Load data list
        self.data_list = self.load_data_list()
        
        if self.debug:
            print(f"{self.dataset_name} dataset {self.split} split loaded {len(self.data_list)} samples, "
                  f"DEBUG MODE ENABLED, ONLY 16 SAMPLES ARE USED")
            self.data_list = self.data_list[:16]
        else:
            print(f"{self.dataset_name} dataset {self.split} split loaded {len(self.data_list)} samples.")
    
    @abstractmethod
    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        """
        Abstract method to iterate over dataset samples.
        
        Yields:
            Tuple of (image_path, label_path)
        """
    
    def load_data_list(self) -> list[dict[str, Any]]:
        """
        Load data list from sample iterator.
        
        Returns:
            List of dictionaries containing sample information
        """
        data_list = []
        for image_path, label_path in self.sample_iterator():
            data_list.append({
                'img_path': image_path,
                'label_path': label_path,
            })
        return data_list
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        sample = self.data_list[idx].copy()
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class LightningDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for medical imaging datasets.
    
    This DataModule provides a standardized way to handle data loading,
    splitting, and transformation for medical imaging tasks.
    """
    
    def __init__(
        self,
        dataset_class: type,
        data_root: str | Path,
        batch_size: int = 1,
        num_workers: int = 4,
        train_transform: Any | None = None,
        val_transform: Any | None = None,
        test_transform: Any | None = None,
        **dataset_kwargs
    ) -> None:
        """
        Initialize the DataModule.
        
        Args:
            dataset_class: The dataset class to use
            data_root: Root directory of the dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            train_transform: Transform for training data
            val_transform: Transform for validation data
            test_transform: Transform for test data
            **dataset_kwargs: Additional arguments for dataset initialization
        """
        super().__init__()
        self.save_hyperparameters(ignore=['dataset_class', 'train_transform', 'val_transform', 'test_transform'])
        
        self.dataset_class = dataset_class
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.dataset_kwargs = dataset_kwargs
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets for different stages.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(
                data_root=self.data_root,
                split="train",
                transform=self.train_transform,
                **self.dataset_kwargs
            )
            self.val_dataset = self.dataset_class(
                data_root=self.data_root,
                split="val",
                transform=self.val_transform,
                **self.dataset_kwargs
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(
                data_root=self.data_root,
                split="test",
                transform=self.test_transform,
                **self.dataset_kwargs
            )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Return the training data loader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Return the validation data loader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Return the test data loader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

