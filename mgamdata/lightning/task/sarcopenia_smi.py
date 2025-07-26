import re
import pdb
import torch
from tqdm import tqdm
from typing import Any
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import Tensor
from sklearn.metrics import r2_score
from torch.utils.checkpoint import checkpoint

from ..dataset.base import BaseDataset


class SarcopeniaSMIRegressionTask(pl.LightningModule):
    def __init__(
        self,
        enable_image_input: bool,
        enable_semantic_input: bool,
        backbone: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer_config: dict = {'lr': 1e-4, 'weight_decay': 1e-5},
        scheduler_config: dict | None = None,
        gt_key: str = 'smi',
        num_semantic_classes: int = 7,
    ) -> None:
        """
        Args:
            backbone (torch.nn.Module): The neural network backbone for regression. It should accept
                a 3D tensor and output a single value.
            criterion (torch.nn.Module): The loss function (e.g., MSELoss, L1Loss).
            optimizer_config (dict): Configuration for the AdamW optimizer.
            scheduler_config (dict | None): Configuration for the learning rate scheduler.
                If None, no scheduler is used.
            gt_key (str): The key to access the ground truth SMI value in the batch dictionary.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'criterion'])
        self.enable_image_input = enable_image_input
        self.enable_semantic_input = enable_semantic_input
        self.backbone = backbone
        self.criterion = criterion
        self.gt_key = gt_key
        self.num_semantic_classes = num_semantic_classes
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        self.out_proj = torch.nn.Linear(self.backbone.embed_dims[-1], 1)  # pyright: ignore

    def forward(self, x: Tensor) -> Tensor:
        backbone_out:Tensor = self.backbone(x)[-1] # [B, C, Z, Y, X]
        spatial_average = torch.nn.functional.adaptive_avg_pool3d(backbone_out, (1, 1, 1))  # [B, C, 1, 1, 1]
        sample_average = spatial_average.squeeze((2, 3, 4)) # [B, C]
        pred = self.out_proj(sample_average).squeeze(1)  # [B]
        return pred

    def _parse_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Extracts image and ground truth SMI from a batch.
        
        Including two steps:
        1. Converts the label to one-hot encoding
        2. Concatenates it with the image.
        """
        if self.enable_image_input:
            image = batch['image'].to(device=self.device, dtype=torch.float32, non_blocking=True)
        if self.enable_semantic_input:
            label = batch['label'].to(device=self.device, dtype=torch.float32, non_blocking=True)
            label_dtype = label.dtype
            label = torch.nn.functional.one_hot(label[:, 0, ...].long(), num_classes=self.num_semantic_classes).permute(0, 4, 1, 2, 3).to(dtype=label_dtype)
            label = label[:, 1:, ...]  # Exclude the background class (index 0)
        gt_smi = batch[self.gt_key].to(device=self.device, non_blocking=True)
        
        if self.enable_image_input and self.enable_semantic_input:
            out = torch.cat([image, label], dim=1)  # pyright: ignore
        elif self.enable_image_input:
            out = image  # pyright: ignore
        else:
            out = label  # pyright: ignore
        return out, gt_smi

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Performs a single training step."""
        image, gt_smi = self._parse_batch(batch)
        pred_smi = self(image)  # Ensure output is scalar-like
        loss = self.criterion(pred_smi, gt_smi.float())
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False, logger=True, sync_dist=True, batch_size=len(image))
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        """Performs a single validation step."""
        image, gt_smi = self._parse_batch(batch)
        pred_smi = self(image)

        loss = self.criterion(pred_smi, gt_smi.float())
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(image))
        mae = torch.nn.functional.l1_loss(pred_smi, gt_smi.float())
        self.log('val/mae', mae, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=len(image))
        mape = torch.mean(torch.abs((gt_smi.float() - pred_smi) / (gt_smi.float() + 1e-5))) * 100
        self.log('val/mape(%)', mape, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=len(image))
        r2 = r2_score(gt_smi.float().detach().cpu().numpy(), pred_smi.detach().cpu().numpy())
        self.log('val/r2', float(r2), on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=len(image))

        return {'val_loss': loss,
                'val_mae': mae,
                'val_mape(%)': mape,
                'val_r2': r2,
                'predictions': pred_smi,
                'targets': gt_smi}

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
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)

        if self.scheduler_config is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_config
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


class SarcopeniaSMIDataset(BaseDataset):
    SMI_MEAN = 0  # 7.19
    SMI_MULTIPLIER = 1.0  # 1.0
    
    """
    Args:
        image_root (str | Path): Path to the directory containing .mha image files.
        meta_path (str | Path): Path to the CSV or Excel file containing SMI values.
        split_accordance (str | Path): Path to a directory whose contents (.mha files)
            define the dataset's samples.
        series_uid_col (str): The name of the column in the metadata file that contains the SeriesUID.
        smi_col (str): The name of the column in the metadata file that contains the SMI target value.
        segmentation_root (str | Path | None): Optional path to the directory containing segmentation .mha files.
        **kwargs: Additional arguments passed to the `BaseDataset` constructor.
    """
    def __init__(
        self,
        image_root: str | Path,
        meta_path: str | Path,
        split_accordance: str | Path,
        series_uid_col: str = 'SeriesUID',
        smi_col: str = 'SMI (Skeletal Muscle Index)-BIA',
        segmentation_root: str | Path | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_root = Path(image_root)
        self.meta_path = Path(meta_path)
        self.split_accordance = Path(split_accordance)
        self.series_uid_col = series_uid_col
        self.smi_col = smi_col
        self.segmentation_root = Path(segmentation_root) if segmentation_root else None

        # Load SMI values from metadata file into a lookup dictionary
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
        # Filter out rows with NaN values in the SMI column
        df = df.dropna(subset=[self.smi_col])
        self.smi_map = df.set_index(self.series_uid_col)[self.smi_col].to_dict()

        # Get all series UIDs from the split accordance directory
        all_series_uids = self._search_series()

        # Filter UIDs to ensure they exist across all specified data sources
        self.valid_series_uids = []
        for uid in all_series_uids:
            image_exists = (self.image_root / f"{uid}.mha").exists()
            meta_exists = uid in self.smi_map
            
            seg_exists = True
            if self.segmentation_root:
                seg_exists = (self.segmentation_root / f"{uid}.mha").exists()

            if image_exists and meta_exists and seg_exists:
                self.valid_series_uids.append(uid)
        
        if len(self.valid_series_uids) < len(all_series_uids):
            print(f"Found {len(all_series_uids)} series in split_accordance, "
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
        Retrieves a sample, including image, SMI value, and optional segmentation path.
        """
        series_uid = self.valid_series_uids[index]
        smi_value = self.smi_map[series_uid]
        
        sample = {
            "series_uid": series_uid,
            "image_mha_path": str(self.image_root / f"{series_uid}.mha"),
            "smi": np.array((smi_value - self.SMI_MEAN) * self.SMI_MULTIPLIER)
        }

        if self.segmentation_root:
            sample['label_mha_path'] = str(self.segmentation_root / f"{series_uid}.mha")
        
        # Apply the processing pipeline (e.g., loading, transformations)
        return self._preprocess(sample)


class SMIRegVis3DCallback(pl.Callback):
    """
    Callback for visualizing 3D regression task inputs and results during validation and testing.

    This callback creates visualizations showing each input channel of a sample on different axial slices.
    It also displays the ground truth and predicted SMI values in the title.
    """

    def __init__(
        self,
        log_every_n_batches: int = 10,
        log_every_n_epochs: int = 1,
        max_samples_per_epoch: int = 5,
        slice_indices: list[int] = None,
        figsize: tuple[int, int] = (16, 8),
        cmap: str = 'gray',
    ):
        """
        Initialize the visualization callback.

        Args:
            log_every_n_batches (int): Log visualization every N batches.
            log_every_n_epochs (int): Log visualization every N epochs.
            max_samples_per_epoch (int): Maximum number of samples to visualize per epoch.
            slice_indices (list[int] | None): Specific slice indices to visualize. If None, defaults to [Z/4, Z/2, 3Z/4].
            figsize (tuple[int, int]): Figure size for matplotlib plots.
            cmap (str): Colormap for the images.
        """
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples_per_epoch = max_samples_per_epoch
        self.slice_indices = slice_indices
        self.figsize = figsize
        self.cmap = cmap
        self.samples_visualized_this_epoch = 0

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.samples_visualized_this_epoch = 0

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.samples_visualized_this_epoch = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: SarcopeniaSMIRegressionTask,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_visualize(trainer, pl_module, outputs, batch, batch_idx, stage='val')

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: SarcopeniaSMIRegressionTask,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_visualize(trainer, pl_module, outputs, batch, batch_idx, stage='test')

    def _maybe_visualize(
        self,
        trainer: pl.Trainer,
        pl_module: SarcopeniaSMIRegressionTask,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        stage: str
    ) -> None:
        should_log_batch = batch_idx % self.log_every_n_batches == 0
        should_log_epoch = trainer.current_epoch % self.log_every_n_epochs == 0
        within_sample_limit = self.samples_visualized_this_epoch < self.max_samples_per_epoch
        if not (should_log_batch and should_log_epoch and within_sample_limit and trainer.logger):
            return

        model_input, gt_smi = pl_module._parse_batch(batch)
        pred_smi = outputs['predictions']

        sample_idx = 0
        sample_input = model_input[sample_idx].cpu().numpy()
        gt_smi_val = gt_smi[sample_idx].cpu().item()
        pred_smi_val = pred_smi[sample_idx].cpu().item()

        num_channels, z_size = sample_input.shape[0], sample_input.shape[1]
        slice_indices = self.slice_indices or [z_size // 4, z_size // 2, 3 * z_size // 4]
        
        fig, axes = plt.subplots(len(slice_indices), num_channels, figsize=self.figsize)
        fig.suptitle(
            f'{stage.capitalize()} Vis - Batch {batch_idx}, Epoch {trainer.current_epoch}\n'
            f'GT SMI: {gt_smi_val:.4f}, Pred SMI: {pred_smi_val:.4f}',
            fontsize=16
        )

        col_titles = ['Image'] + [f'Semantic Ch {i}' for i in range(1, num_channels)]
        for col, title in enumerate(col_titles):
            ax = axes[0, col] if len(slice_indices) > 1 else axes[col]
            ax.set_title(title, fontsize=12, fontweight='bold')

        for row, slice_idx in enumerate(slice_indices):
            for col in range(num_channels):
                ax = axes[row, col] if len(slice_indices) > 1 else axes[col]
                ax.imshow(sample_input[col, min(slice_idx, z_size - 1)], cmap=self.cmap)
                if col == 0:
                    ax.set_ylabel(f'Slice {slice_idx}', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'add_figure'):
            trainer.logger.experiment.add_figure(
                f'{stage}_SMIRegVis3D/batch_{batch_idx}',
                fig,
                trainer.global_step
            )
        
        plt.close(fig)
        self.samples_visualized_this_epoch += 1


class SarcopeniaSMISlidingWindowRegressionTask(SarcopeniaSMIRegressionTask):
    """
    Sliding window version of SarcopeniaSMIRegressionTask to handle large input matrices.
    
    This class extends the base regression task to support sliding window inference
    and training with gradient checkpointing to reduce memory usage.
    """
    
    def __init__(
        self,
        enable_image_input: bool,
        enable_semantic_input: bool,
        backbone: torch.nn.Module,
        criterion: torch.nn.Module,
        patch_size: Sequence[int] = (64, 256, 256),
        patch_stride: Sequence[int] = (32, 128, 128),
        optimizer_config: dict = {'lr': 1e-4, 'weight_decay': 1e-5},
        scheduler_config: dict | None = None,
        gt_key: str = 'smi',
        num_semantic_classes: int = 7,
    ) -> None:
        """
        Args:
            patch_size (Sequence[int]): Size of sliding window patches (Z, Y, X).
            patch_stride (Sequence[int]): Stride of sliding window (Z, Y, X).
            Other args are the same as parent class.
        """
        super().__init__(
            enable_image_input=enable_image_input,
            enable_semantic_input=enable_semantic_input,
            backbone=backbone,
            criterion=criterion,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            gt_key=gt_key,
            num_semantic_classes=num_semantic_classes,
        )
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
    
    def _get_patch_positions(self, Z: int, Y: int, X: int) -> list[tuple[int, int, int, int, int, int]]:
        """
        Calculate patch positions for sliding window.
        
        Args:
            Z, Y, X (int): Input tensor dimensions
            
        Returns:
            list[tuple]: List of (z_start, z_end, y_start, y_end, x_start, x_end) tuples
        """
        patch_z, patch_y, patch_x = self.patch_size
        stride_z, stride_y, stride_x = self.patch_stride
        
        # 计算滑动窗口的起始位置
        z_starts = list(range(0, max(1, Z - patch_z + 1), stride_z))
        y_starts = list(range(0, max(1, Y - patch_y + 1), stride_y))
        x_starts = list(range(0, max(1, X - patch_x + 1), stride_x))
        
        # 确保最后一个patch覆盖到边界
        if z_starts[-1] + patch_z < Z:
            z_starts.append(max(0, Z - patch_z))
        if y_starts[-1] + patch_y < Y:
            y_starts.append(max(0, Y - patch_y))
        if x_starts[-1] + patch_x < X:
            x_starts.append(max(0, X - patch_x))
        
        positions = []
        for z_start in z_starts:
            for y_start in y_starts:
                for x_start in x_starts:
                    z_end = min(z_start + patch_z, Z)
                    y_end = min(y_start + patch_y, Y)
                    x_end = min(x_start + patch_x, X)
                    positions.append((z_start, z_end, y_start, y_end, x_start, x_end))
        
        return positions
    
    def _process_patch_at_position(self, x: Tensor, z_start: int, z_end: int, 
                                   y_start: int, y_end: int, x_start: int, x_end: int) -> Tensor:
        """
        Process a patch at specific position through the backbone.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, Z, Y, X]
            z_start, z_end, y_start, y_end, x_start, x_end (int): Patch boundaries
            
        Returns:
            Tensor: Processed features of shape [B, C]
        """
        patch_z, patch_y, patch_x = self.patch_size
        
        # 直接从源矩阵中提取patch
        patch = x[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
        
        # 如果patch小于目标大小，进行padding
        if patch.shape[2:] != (patch_z, patch_y, patch_x):
            pad_z = patch_z - patch.shape[2]
            pad_y = patch_y - patch.shape[3]
            pad_x = patch_x - patch.shape[4]
            patch = torch.nn.functional.pad(
                patch, 
                (0, pad_x, 0, pad_y, 0, pad_z), 
                mode='constant', 
                value=0
            )
        
        # 通过backbone处理
        backbone_out = self.backbone(patch)[-1]  # [B, C, Z, Y, X]
        spatial_average = torch.nn.functional.adaptive_avg_pool3d(backbone_out, (1, 1, 1))  # [B, C, 1, 1, 1]
        sample_average = spatial_average.squeeze((2, 3, 4))  # [B, C]
        return sample_average
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with sliding window mechanism.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, Z, Y, X]
            
        Returns:
            Tensor: Predicted SMI values of shape [B]
        """
        B, C, Z, Y, X = x.shape
        
        # 如果输入尺寸小于或等于patch_size，直接使用原来的forward
        if (Z <= self.patch_size[0] and 
            Y <= self.patch_size[1] and 
            X <= self.patch_size[2]):
            return super().forward(x)
        
        # 获取所有patch位置
        patch_positions = self._get_patch_positions(Z, Y, X)
        num_patches = len(patch_positions)
        
        feature_dim = self.backbone.embed_dims[-1]  # pyright: ignore
        aggregated_features = torch.zeros(B, feature_dim, device=x.device, dtype=x.dtype)
        
        for z_start, z_end, y_start, y_end, x_start, x_end in patch_positions:
            if self.training:
                # 训练模式下使用gradient checkpointing
                patch_feature:torch.Tensor = checkpoint( # pyright: ignore
                    self._process_patch_at_position, 
                    x, z_start, z_end, y_start, y_end, x_start, x_end,
                    use_reentrant=False
                )
            else:
                # 推断模式下直接计算
                patch_feature = self._process_patch_at_position(
                    x, z_start, z_end, y_start, y_end, x_start, x_end
                )
            
            # 累加到输出矩阵
            aggregated_features += patch_feature
        
        # 归一化
        aggregated_features /= num_patches

        # 输出映射
        pred = self.out_proj(aggregated_features).squeeze(1)  # [B]
        
        return pred
