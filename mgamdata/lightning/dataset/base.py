from typing_extensions import Literal
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    TRAIN_LOADER_ARGS = {
        "shuffle": True,
        "num_workers": 1,
        "pin_memory": True,
        "persistent_workers": True,
    }
    VAL_TEST_LOADER_ARGS = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 1,
        "pin_memory": True,
        "persistent_workers": True,
    }
    
    def __init__(
        self,
        dataset: Dataset,
        train_loader_args = {},
        val_test_loader_args = {},
    ):
        super().__init__()
        self.dataset = dataset
        self.train_loader_args = {**self.TRAIN_LOADER_ARGS, **train_loader_args}
        self.val_test_loader_args = {**self.VAL_TEST_LOADER_ARGS, **val_test_loader_args}

    def prepare_data(self):
        """
        download, IO, etc. Useful with shared filesystems
        only called on 1 GPU/TPU in distributed
        """

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        train_end_idx = int(len(self.dataset) * self.dataset.SPLIT_RATIO[0])
        val_end_idx = train_end_idx + max(1, int(len(self.dataset) * self.dataset.SPLIT_RATIO[1]))
        test_end_idx = len(self.dataset)
        self.train = Subset(self.dataset, range(train_end_idx))
        self.val = Subset(self.dataset, range(train_end_idx, val_end_idx))
        self.test = Subset(self.dataset, range(val_end_idx, test_end_idx))

    def teardown(self, stage: Literal['fit', 'validate', 'test', 'predict']) -> None:
        """
        clean up state after the trainer stops, delete files...
        called on every process in DDP
        """

    def on_exception(self, exception):
        """ clean up state after the trainer faced an exception """

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, **self.train_loader_args)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, **self.val_test_loader_args)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, **self.val_test_loader_args)


