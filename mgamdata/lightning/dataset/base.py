from abc import abstractmethod
from collections.abc import Generator
from typing import Any
from pathlib import Path
from typing_extensions import Literal

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class BaseDataset(LightningDataModule):
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
    ) -> None:
        """
        Initialize the base dataset.
        
        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test', 'all', or None)
            debug: If True, only use 16 samples for debugging
            dataset_name: Name of the dataset (defaults to class name)
            pipeline: Data processing pipeline (default is identity function)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.debug = debug
        self.data_list = self.load_data_list()

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

    def prepare_data(self):
        """
        download, IO, etc. Useful with shared filesystems
        only called on 1 GPU/TPU in distributed
        """

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        """
        make assignments here (val/train/test split)
        called on every process in DDP
        
        EXAMPLE:
        
        dataset = RandomDataset(1, 100)
        self.train, self.val, self.test = data.random_split(
            dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
        )
        """

    def teardown(self, stage: Literal['fit', 'validate', 'test', 'predict']) -> None:
        """
        clean up state after the trainer stops, delete files...
        called on every process in DDP
        """

    def on_exception(self, exception):
        """ clean up state after the trainer faced an exception """

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test)


