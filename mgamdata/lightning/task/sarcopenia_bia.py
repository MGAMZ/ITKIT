import re
import torch
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import Tensor

from ..dataset.base import BaseDataset
from ..pipeline.base import BaseTransform


class SarcopeniaBIARegressionTask(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer_config: dict = {'lr': 1e-4, 'weight_decay': 1e-5},
        scheduler_config: dict | None = None,
        gt_key: str = 'bia',
    ):
        """
        Args:
            model (torch.nn.Module): The neural network model for regression. It should accept
                a 3D tensor and output a single value.
            criterion (torch.nn.Module): The loss function (e.g., MSELoss, L1Loss).
            optimizer_config (dict): Configuration for the AdamW optimizer.
            scheduler_config (dict | None): Configuration for the learning rate scheduler.
                If None, no scheduler is used.
            gt_key (str): The key to access the ground truth BIA value in the batch dictionary.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.model = model
        self.criterion = criterion
        self.gt_key = gt_key

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).flatten(1).mean(dim=1)

    def _parse_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Extracts image and ground truth BIA from a batch."""
        image = batch['image'].to(device=self.device, non_blocking=True)
        gt_bia = batch[self.gt_key].to(device=self.device, non_blocking=True)
        return image, gt_bia

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Performs a single training step."""
        image, gt_bia = self._parse_batch(batch)
        pred_bia = self(image).squeeze()  # Ensure output is scalar-like
        loss = self.criterion(pred_bia, gt_bia.float())
        self.log('train/loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=len(image))
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        """Performs a single validation step."""
        image, gt_bia = self._parse_batch(batch)
        pred_bia = self(image).squeeze()

        loss = self.criterion(pred_bia, gt_bia.float())
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(image))
        mae = torch.nn.functional.l1_loss(pred_bia, gt_bia.float())
        self.log('val/mae', mae, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=len(image))
        mape = torch.mean(torch.abs((gt_bia.float() - pred_bia) / (gt_bia.float() + 1e-8))) * 100
        self.log('val/mape', mape, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=len(image))

        return {'val_loss': loss, 'predictions': pred_bia, 'targets': gt_bia}

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        """Performs a single test step."""
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        """Performs a prediction step."""
        image = batch['image'].to(device=self.device, non_blocking=True)
        batch['prediction'] = self(image).squeeze()
        return batch

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_config)

        if self.hparams.scheduler_config is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.hparams.scheduler_config
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


class SarcopeniaBIADataset(BaseDataset):
    """
    Dataset for Sarcopenia BIA regression.

    This dataset combines 3D CT images (in .mha format) with Body Impedance Analysis (BIA)
    values from a corresponding metadata file (CSV or Excel). It can optionally include
    a segmentation mask, which is converted to one-hot and concatenated to the image.

    Args:
        image_root (str | Path): Path to the directory containing .mha image files.
        meta_path (str | Path): Path to the CSV or Excel file containing BIA values.
        split_accordance (str | Path): Path to a directory whose contents (.mha files)
            define the dataset's samples.
        series_uid_col (str): The name of the column in the metadata file that contains the SeriesUID.
        bia_col (str): The name of the column in the metadata file that contains the BIA target value.
        segmentation_root (str | Path | None): Optional path to the directory containing segmentation .mha files.
        **kwargs: Additional arguments passed to the `BaseDataset` constructor.
    """
    def __init__(
        self,
        image_root: str | Path,
        meta_path: str | Path,
        split_accordance: str | Path,
        series_uid_col: str = 'SeriesUID',
        bia_col: str = '45. BMI (Body Mass Index)',
        segmentation_root: str | Path | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_root = Path(image_root)
        self.meta_path = Path(meta_path)
        self.split_accordance = Path(split_accordance)
        self.series_uid_col = series_uid_col
        self.bia_col = bia_col
        self.segmentation_root = Path(segmentation_root) if segmentation_root else None

        # Load BIA values from metadata file into a lookup dictionary
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        file_ext = self.meta_path.suffix.lower()
        if file_ext == '.csv':
            df = pd.read_csv(self.meta_path, skiprows=1)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(self.meta_path, skiprows=1)
        else:
            raise ValueError(f"Unsupported metadata file format: {file_ext}. Please use .csv or .xlsx.")

        # Ensure the UID column is treated as string to match file names
        df[self.series_uid_col] = df[self.series_uid_col].astype(str)
        self.bia_map = df.set_index(self.series_uid_col)[self.bia_col].to_dict()

        # Get all series UIDs from the split accordance directory
        all_series_uids = self._search_series()

        # Filter UIDs to ensure they exist across all specified data sources
        self.valid_series_uids = []
        for uid in all_series_uids:
            image_exists = (self.image_root / f"{uid}.mha").exists()
            meta_exists = uid in self.bia_map
            
            seg_exists = True
            if self.segmentation_root:
                seg_exists = (self.segmentation_root / f"{uid}.mha").exists()

            if image_exists and meta_exists and seg_exists:
                self.valid_series_uids.append(uid)
        
        if len(self.valid_series_uids) < len(all_series_uids):
            print(f"Warning: Found {len(all_series_uids)} series in split_accordance, "
                  f"but only {len(self.valid_series_uids)} are available across all specified paths (image, meta, seg).")
        
        if len(self.valid_series_uids) == 0:
            print("Warning: No valid samples found. Check your paths and UID matches.")

    def _search_series(self) -> list[str]:
        """Scans the split_accordance directory for series UIDs based on .mha filenames."""
        if not self.split_accordance.exists():
            raise FileNotFoundError(f"Split accordance directory not found: {self.split_accordance}")
        
        all_series = [file.stem for file in self.split_accordance.glob("*.mha")]
        
        # Attempt to sort numerically based on numbers in the filename
        try:
            # Use a regex that finds all numbers and uses the first one for sorting
            return sorted(all_series, key=lambda x: int(re.search(r"\d+", x).group()))
        except (AttributeError, ValueError):
            # Fallback to alphanumeric sort if no numbers are found or parsing fails
            print("Warning: Could not sort series UIDs numerically. Falling back to alphanumeric sort.")
            return sorted(all_series)

    def __len__(self) -> int:
        """Returns the number of valid samples."""
        return len(self.valid_series_uids) if self.debug is False else min(10, len(self.valid_series_uids))

    def __getitem__(self, index) -> dict:
        """
        Retrieves a sample, including image, BIA value, and optional segmentation path.
        """
        series_uid = self.valid_series_uids[index]
        bia_value = self.bia_map[series_uid]
        
        sample = {
            "series_uid": series_uid,
            "image_mha_path": str(self.image_root / f"{series_uid}.mha"),
            "bia": np.array(bia_value)
        }

        if self.segmentation_root:
            sample['label_mha_path'] = str(self.segmentation_root / f"{series_uid}.mha")
        
        # Apply the processing pipeline (e.g., loading, transformations)
        return self._preprocess(sample)


class ConcatImageAndSemanticChannel(BaseTransform):
    def __call__(self, sample:dict):
        sample['image'] = np.concatenate(
            [sample['image'], sample['label']],
            axis=0
        )
        del sample['label']
        return sample
