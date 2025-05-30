"""
Lightning-based segmentation task modules.

This module provides PyTorch Lightning implementations for segmentation tasks,
including 2D and 3D segmentation. These classes follow Lightning framework 
conventions and use modern Python 3.12 type annotations.
"""

import torch
from torch import Tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass, field
from torch.optim import Optimizer


class VoxelData:
    """Data wrapper for 3D voxel data with shape validation.
    
    Ensures data has 4 channels: (C, Z, Y, X)
    Acts like a tensor - accessing VoxelData directly returns the underlying data.
    """
    
    def __init__(self, data: Tensor):
        """Initialize VoxelData with shape validation.
        
        Args:
            data: Tensor with shape (C, Z, Y, X)
            
        Raises:
            ValueError: If data doesn't have exactly 4 dimensions
        """
        if data.dim() != 4:
            raise ValueError(f"VoxelData requires 4D tensor (C, Z, Y, X), got {data.dim()}D tensor with shape {data.shape}")
        self._data = data
    
    @property
    def data(self) -> Tensor:
        """Get the underlying tensor data."""
        return self._data
    
    @property
    def shape(self) -> torch.Size:
        """Get data shape."""
        return self._data.shape
    
    @property
    def device(self) -> torch.device:
        """Get data device."""
        return self._data.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get data dtype."""
        return self._data.dtype
    
    def to(self, device: torch.device) -> 'VoxelData':
        """Move data to device."""
        return VoxelData(self._data.to(device))
    
    # Magic methods to make it behave like a tensor
    def __getattr__(self, name):
        """Delegate attribute access to underlying tensor."""
        return getattr(self._data, name)
    
    def __getitem__(self, key):
        """Support tensor indexing."""
        return self._data[key]
    
    def __len__(self):
        """Support len() operation."""
        return len(self._data)
    
    def __iter__(self):
        """Support iteration."""
        return iter(self._data)
    
    def __repr__(self) -> str:
        return f"VoxelData(shape={self.shape}, dtype={self.dtype}, device={self.device})"


class PixelData:
    """Data wrapper for 2D pixel data with shape validation.
    
    Ensures data has 3 channels: (C, Y, X)
    Acts like a tensor - accessing PixelData directly returns the underlying data.
    """
    
    def __init__(self, data: Tensor):
        """Initialize PixelData with shape validation.
        
        Args:
            data: Tensor with shape (C, Y, X)
            
        Raises:
            ValueError: If data doesn't have exactly 3 dimensions
        """
        if data.dim() != 3:
            raise ValueError(f"PixelData requires 3D tensor (C, Y, X), got {data.dim()}D tensor with shape {data.shape}")
        self._data = data
    
    @property
    def data(self) -> Tensor:
        """Get the underlying tensor data."""
        return self._data
    
    @property
    def shape(self) -> torch.Size:
        """Get data shape."""
        return self._data.shape
    
    @property
    def device(self) -> torch.device:
        """Get data device."""
        return self._data.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get data dtype."""
        return self._data.dtype
    
    def to(self, device: torch.device) -> 'PixelData':
        """Move data to device."""
        return PixelData(self._data.to(device))
    
    # Magic methods to make it behave like a tensor
    def __getattr__(self, name):
        """Delegate attribute access to underlying tensor."""
        return getattr(self._data, name)
    
    def __getitem__(self, key):
        """Support tensor indexing."""
        return self._data[key]
    
    def __len__(self):
        """Support len() operation."""
        return len(self._data)
    
    def __iter__(self):
        """Support iteration."""
        return iter(self._data)
    
    def __repr__(self) -> str:
        return f"PixelData(shape={self.shape}, dtype={self.dtype}, device={self.device})"


class SegmentationDataModule(pl.LightningDataModule):
    """Lightning DataModule for segmentation tasks.
    
    This class leverages Lightning's built-in patterns and factory methods
    for simplified and consistent data handling.
    """
    
    def __init__(
        self,
        dataloader_kwargs: dict[str, Any] | None,
        train_dataset: Any | None = None,
        val_dataset: Any | None = None,
        test_dataset: Any | None = None,
        predict_dataset: Any | None = None,
        **kwargs
    ):
        """Initialize the segmentation data module.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset  
            test_dataset: Test dataset
            predict_dataset: Prediction dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory in data loaders
            persistent_workers: Whether to keep workers persistent
            shuffle_train: Whether to shuffle training data
            drop_last_train: Whether to drop last incomplete batch in training
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset', 'test_dataset', 'predict_dataset'])
        
        # Store datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.dataloader_kwargs = dataloader_kwargs or {}
    
    def _create_dataloader(self, dataset: Any) -> DataLoader:
        return DataLoader(dataset, **self.dataloader_kwargs)
    
    def train_dataloader(self) -> DataLoader | None:
        if self.train_dataset is None:
            return None
        return DataLoader(self.train_dataset, **self.dataloader_kwargs)
    
    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, **self.dataloader_kwargs)
    
    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, **self.dataloader_kwargs)
    
    def predict_dataloader(self) -> DataLoader | None:
        if self.predict_dataset is None:
            return None
        return DataLoader(self.predict_dataset, **self.dataloader_kwargs)


@dataclass
class SegmentationDataElement:
    """Simplified data element for segmentation tasks.
    
    This class provides a lightweight data container for individual
    segmentation samples with essential attributes only.
    """
    # Essential data
    img: Tensor | None = None
    gt_sem_seg: Tensor | PixelData | VoxelData | None = None
    
    # Predictions (set during inference)
    pred_sem_seg: Tensor | PixelData | VoxelData | None = None
    seg_logits: Tensor | PixelData | VoxelData | None = None
    
    # Optional metadata
    img_path: str | None = None
    gt_sem_seg_path: str | None = None
    img_shape: tuple[int, ...] | None = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value by key with mm-style compatibility."""
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set attribute value by key."""
        setattr(self, key, value)


class LightningSegmentationBase(pl.LightningModule):
    """Base Lightning module for segmentation tasks.
    
    This class provides a Lightning-native implementation that maintains compatibility
    with mm-style APIs while following Lightning conventions and modern Python 3.12
    type annotations. The neural network model is now passed as a parameter, allowing
    complete decoupling of model implementation from task logic.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module | list[torch.nn.Module],
        num_classes: int,
        optimizer_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
        gt_sem_seg_key: str = 'gt_sem_seg',
        binary_segment_threshold: float | None = None,
        inference_patch_size: tuple[int, ...] | None = None,
        inference_patch_stride: tuple[int, ...] | None = None,
        inference_patch_accumulate_device: str = 'cuda',
        log_predictions: bool = False,
        **kwargs
    ):
        """Initialize the Lightning segmentation module.
        
        Args:
            model: Neural network model (should output logits directly)
            criterion: Loss function(s) module(s)
            num_classes: Number of segmentation classes
            optimizer_config: Optimizer configuration dict with 'type' key
            scheduler_config: Learning rate scheduler configuration dict with 'type' key
            gt_sem_seg_key: Key for ground truth segmentation in data samples
            binary_segment_threshold: Threshold for binary segmentation (required for single-class output)
            inference_patch_size: Patch size for sliding window inference
            inference_patch_stride: Patch stride for sliding window inference
            inference_patch_accumulate_device: Device for accumulating patch results ('cuda' or 'cpu')
            log_predictions: Whether to log prediction samples during validation
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        
        # Core components - changed from backbone to model
        self.model = model
        
        # Handle criterion(s) - support both single and multiple criteria like mm
        if isinstance(criterion, (list, tuple)):
            self.criteria = list(criterion)
        else:
            self.criteria = [criterion]
        
        # Configuration
        self.num_classes = num_classes
        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}
        self.gt_sem_seg_key = gt_sem_seg_key
        self.binary_segment_threshold = binary_segment_threshold
        self.inference_patch_size = inference_patch_size
        self.inference_patch_stride = inference_patch_stride
        self.inference_patch_accumulate_device = inference_patch_accumulate_device
        self.log_predictions = log_predictions
        
        # Validation for binary segmentation configuration
        self._validate_binary_threshold()

    def _validate_binary_threshold(self) -> None:
        """Validate binary segmentation threshold configuration."""
        # This will be validated during the first forward pass when we know the output channels
        pass

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits from model
        """
        return self.model(x)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Training step following Lightning conventions.
        
        Args:
            batch: Training batch containing 'inputs' and 'data_samples'
            batch_idx: Batch index
            
        Returns:
            Training loss tensor
        """
        inputs = batch['inputs']
        data_samples = batch['data_samples']
        
        # Forward pass
        logits = self(inputs)
        
        # Extract ground truth using mm-compatible method
        gt_segs = self._extract_ground_truth(data_samples)
        
        # Compute losses using mm-style loss computation
        losses = self._compute_losses(logits, gt_segs)
        total_loss = sum(losses.values())
        
        # Log losses with Lightning conventions
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Validation step.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Dictionary containing validation metrics
        """
        inputs = batch['inputs']
        data_samples = batch['data_samples']
        
        # Forward pass
        logits = self.inference(inputs)
        
        # Extract ground truth
        gt_segs = self._extract_ground_truth(data_samples)
        
        # Compute losses
        losses = self._compute_losses(logits, gt_segs)
        total_loss = sum(losses.values())
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val/{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute predictions for metrics
        predictions = self._logits_to_predictions(logits)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, gt_segs)
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': total_loss, 'predictions': predictions, 'targets': gt_segs}

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Test step.
        
        Args:
            batch: Test batch
            batch_idx: Batch index
            
        Returns:
            Dictionary containing test metrics
        """
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Prediction step.
        
        Args:
            batch: Prediction batch
            batch_idx: Batch index
            
        Returns:
            Dictionary containing predictions
        """
        inputs = batch['inputs']
        data_samples = batch.get('data_samples', None)
        
        # Forward pass
        logits = self.inference(inputs)
        predictions = self._logits_to_predictions(logits)
        
        # Create data samples with predictions
        if data_samples is None:
            data_samples = [SegmentationDataElement() for _ in range(inputs.shape[0])]
        
        # Store predictions in data samples
        for i, (pred, logit) in enumerate(zip(predictions, logits)):
            data_samples[i] = self._store_predictions(data_samples[i], pred, logit)
        
        return {'predictions': predictions, 'logits': logits, 'data_samples': data_samples}

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer or dictionary with optimizer and scheduler
        """
        # Default optimizer configuration
        optimizer_config = {
            'type': 'Adam',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            **self.optimizer_config
        }
        
        # Build optimizer
        optimizer_type = optimizer_config.pop('type')
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_config)
        elif optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_config)
        elif optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **optimizer_config)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Configure scheduler if provided
        if self.scheduler_config:
            scheduler_config = self.scheduler_config.copy()
            scheduler_type = scheduler_config.pop('type')
            
            if scheduler_type == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
            elif scheduler_type == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
            elif scheduler_type == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val/total_loss',
                        'frequency': 1
                    }
                }
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'frequency': 1
                }
            }
        
        return optimizer

    @torch.inference_mode()
    def inference(self, inputs: Tensor) -> Tensor:
        """Perform inference with optional sliding window.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output logits
        """
        if self.inference_patch_size is not None and self.inference_patch_stride is not None:
            return self.slide_inference(inputs)
        else:
            return self.forward(inputs)

    @abstractmethod
    def slide_inference(self, inputs: Tensor) -> Tensor:
        """Sliding window inference implementation.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output logits
        """
        raise NotImplementedError("Subclasses must implement slide_inference method")

    @abstractmethod
    def _extract_ground_truth(self, data_samples: Sequence[SegmentationDataElement]) -> Tensor:
        """Extract ground truth from data samples.
        
        Args:
            data_samples: Data samples containing ground truth
            
        Returns:
            Ground truth tensor
        """
        raise NotImplementedError("Subclasses must implement _extract_ground_truth method")

    @abstractmethod
    def _logits_to_predictions(self, logits: Tensor) -> Tensor:
        """Convert logits to predictions.
        
        Args:
            logits: Model output logits
            
        Returns:
            Prediction tensor
        """
        raise NotImplementedError("Subclasses must implement _logits_to_predictions method")

    @abstractmethod
    def _store_predictions(self, data_sample: SegmentationDataElement, prediction: Tensor, logits: Tensor) -> SegmentationDataElement:
        """Store predictions in data sample.
        
        Args:
            data_sample: Data sample to store predictions in
            prediction: Prediction tensor
            logits: Logits tensor
            
        Returns:
            Updated data sample
        """
        raise NotImplementedError("Subclasses must implement _store_predictions method")

    def _compute_losses(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """Compute losses using configured criteria.
        
        Args:
            logits: Model output logits
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        for i, criterion in enumerate(self.criteria):
            loss_name = f'loss_{criterion.__class__.__name__}' if len(self.criteria) > 1 else 'loss'
            losses[loss_name] = criterion(logits, targets)
        return losses

    def _compute_metrics(self, predictions: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Compute accuracy
        if predictions.dim() == targets.dim():
            accuracy = (predictions == targets).float().mean()
            metrics['accuracy'] = accuracy
        
        # Compute IoU for each class (excluding background if multi-class)
        if self.num_classes > 1:
            ious = []
            for class_idx in range(self.num_classes):
                if class_idx == 0:  # Skip background for IoU calculation
                    continue
                pred_mask = (predictions == class_idx)
                target_mask = (targets == class_idx)
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                iou = intersection / (union + 1e-8)
                ious.append(iou)
            
            if ious:
                metrics['mean_iou'] = torch.stack(ious).mean()
        
        return metrics


class LightningSegmentation2D(LightningSegmentationBase):
    """Lightning module for 2D segmentation tasks."""
    
    def slide_inference(self, inputs: Tensor) -> Tensor:
        """2D sliding window inference.
        
        Args:
            inputs: Input tensor with shape (N, C, H, W)
            
        Returns:
            Output logits with shape (N, num_classes, H, W)
        """
        if (self.inference_patch_size is None or self.inference_patch_stride is None or 
            len(self.inference_patch_size) != 2 or len(self.inference_patch_stride) != 2):
            raise ValueError("2D inference requires 2D patch size and stride")
        
        h_stride, w_stride = self.inference_patch_stride
        h_crop, w_crop = self.inference_patch_size
        batch_size, _, h_img, w_img = inputs.size()
        
        # Get output channels from a small forward pass
        with torch.no_grad():
            temp_input = inputs[:, :, :min(h_crop, h_img), :min(w_crop, w_img)]
            temp_output = self.forward(temp_input)
            out_channels = temp_output.size(1)
        
        # Calculate grid numbers
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        
        # Initialize accumulation tensors
        accumulate_device = torch.device(self.inference_patch_accumulate_device)
        preds = torch.zeros(
            size=(batch_size, out_channels, h_img, w_img),
            dtype=torch.float16,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, h_img, w_img),
            dtype=torch.uint8,
            device=accumulate_device
        )
        
        # Sliding window inference
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                h1 = h_idx * h_stride
                w1 = w_idx * w_stride
                h2 = min(h1 + h_crop, h_img)
                w2 = min(w1 + w_crop, w_img)
                h1 = max(h2 - h_crop, 0)
                w1 = max(w2 - w_crop, 0)
                
                # Extract patch
                crop_img = inputs[:, :, h1:h2, w1:w2]
                
                # Forward pass
                crop_logits = self.forward(crop_img)
                
                # Accumulate results
                crop_logits = crop_logits.to(accumulate_device)
                preds[:, :, h1:h2, w1:w2] += crop_logits
                count_mat[:, :, h1:h2, w1:w2] += 1
        
        # Average overlapping predictions
        assert torch.all(count_mat > 0), "Some areas not covered by sliding window"
        logits = preds / count_mat
        
        return logits.to(inputs.device)

    def _extract_ground_truth(self, data_samples: Sequence[SegmentationDataElement]) -> Tensor:
        """Extract 2D ground truth from data samples.
        
        Args:
            data_samples: Data samples containing ground truth
            
        Returns:
            Ground truth tensor with shape (N, H, W)
        """
        gt_segs = []
        for data_sample in data_samples:
            gt_seg = data_sample.get(self.gt_sem_seg_key)
            # VoxelData and PixelData now behave like tensors directly
            if isinstance(gt_seg, (VoxelData, PixelData)):
                gt_segs.append(gt_seg.data)  # Still need .data for the underlying tensor
            else:
                gt_segs.append(gt_seg)
        
        gt_segs = torch.stack(gt_segs, dim=0)
        if gt_segs.dim() == 4 and gt_segs.size(1) == 1:  # Remove channel dimension if present
            gt_segs = gt_segs.squeeze(1)
        
        return gt_segs

    def _logits_to_predictions(self, logits: Tensor) -> Tensor:
        """Convert 2D logits to predictions.
        
        Args:
            logits: Model output logits with shape (N, C, H, W)
            
        Returns:
            Prediction tensor with shape (N, H, W)
        """
        out_channels = logits.shape[1]
        
        # Validate binary threshold configuration
        if out_channels > 1 and self.binary_segment_threshold is not None:
            raise ValueError(
                f"Multi-class model (output channels={out_channels}) should not set "
                f"binary_segment_threshold, current value={self.binary_segment_threshold}, should be None"
            )
        if out_channels == 1 and self.binary_segment_threshold is None:
            raise ValueError(
                f"Binary model (output channels={out_channels}) must set binary_segment_threshold, "
                "current value=None"
            )
        
        if out_channels > 1:  # Multi-class
            predictions = logits.argmax(dim=1)
        else:  # Binary
            assert self.binary_segment_threshold is not None, "Binary threshold must be set for single-channel output"
            predictions = (torch.sigmoid(logits.squeeze(1)) > self.binary_segment_threshold).long()
        
        return predictions

    def _store_predictions(self, data_sample: SegmentationDataElement, prediction: Tensor, logits: Tensor) -> SegmentationDataElement:
        """Store 2D predictions in data sample.
        
        Args:
            data_sample: Data sample to store predictions in
            prediction: Prediction tensor
            logits: Logits tensor
            
        Returns:
            Updated data sample
        """
        data_sample.seg_logits = PixelData(logits)
        data_sample.pred_sem_seg = PixelData(prediction.unsqueeze(0))  # Add channel dimension
        return data_sample


class LightningSegmentation3D(LightningSegmentationBase):
    """Lightning module for 3D segmentation tasks."""
    
    def slide_inference(self, inputs: Tensor) -> Tensor:
        """3D sliding window inference.
        
        Args:
            inputs: Input tensor with shape (N, C, Z, Y, X)
            
        Returns:
            Output logits with shape (N, num_classes, Z, Y, X)
        """
        if (self.inference_patch_size is None or self.inference_patch_stride is None or 
            len(self.inference_patch_size) != 3 or len(self.inference_patch_stride) != 3):
            raise ValueError("3D inference requires 3D patch size and stride")
        
        z_stride, y_stride, x_stride = self.inference_patch_stride
        z_crop, y_crop, x_crop = self.inference_patch_size
        batch_size, _, z_img, y_img, x_img = inputs.size()
        
        # Get output channels from a small forward pass
        with torch.no_grad():
            temp_input = inputs[:, :, :min(z_crop, z_img), :min(y_crop, y_img), :min(x_crop, x_img)]
            temp_output = self.forward(temp_input)
            out_channels = temp_output.size(1)
        
        # Calculate grid numbers
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        
        # Initialize accumulation tensors
        accumulate_device = torch.device(self.inference_patch_accumulate_device)
        preds = torch.zeros(
            size=(batch_size, out_channels, z_img, y_img, x_img),
            dtype=torch.float16,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, z_img, y_img, x_img),
            dtype=torch.uint8,
            device=accumulate_device
        )
        
        # Sliding window inference
        for z_idx in range(z_grids):
            for y_idx in range(y_grids):
                for x_idx in range(x_grids):
                    z1 = z_idx * z_stride
                    y1 = y_idx * y_stride
                    x1 = x_idx * x_stride
                    z2 = min(z1 + z_crop, z_img)
                    y2 = min(y1 + y_crop, y_img)
                    x2 = min(x1 + x_crop, x_img)
                    z1 = max(z2 - z_crop, 0)
                    y1 = max(y2 - y_crop, 0)
                    x1 = max(x2 - x_crop, 0)
                    
                    # Extract patch
                    crop_img = inputs[:, :, z1:z2, y1:y2, x1:x2]
                    
                    # Forward pass
                    crop_logits = self.forward(crop_img)
                    
                    # Accumulate results
                    crop_logits = crop_logits.to(accumulate_device)
                    preds[:, :, z1:z2, y1:y2, x1:x2] += crop_logits
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1
        
        # Average overlapping predictions
        assert torch.all(count_mat > 0), "Some areas not covered by sliding window"
        logits = preds / count_mat
        
        return logits.to(inputs.device)

    def _extract_ground_truth(self, data_samples: Sequence[SegmentationDataElement]) -> Tensor:
        """Extract 3D ground truth from data samples.
        
        Args:
            data_samples: Data samples containing ground truth
            
        Returns:
            Ground truth tensor with shape (N, Z, Y, X)
        """
        gt_segs = []
        for data_sample in data_samples:
            gt_seg = data_sample.get(self.gt_sem_seg_key)
            # VoxelData and PixelData now behave like tensors directly
            if isinstance(gt_seg, (VoxelData, PixelData)):
                gt_segs.append(gt_seg.data)  # Still need .data for the underlying tensor
            else:
                gt_segs.append(gt_seg)
        
        gt_segs = torch.stack(gt_segs, dim=0)
        if gt_segs.dim() == 5 and gt_segs.size(1) == 1:  # Remove channel dimension if present
            gt_segs = gt_segs.squeeze(1)
        
        return gt_segs

    def _logits_to_predictions(self, logits: Tensor) -> Tensor:
        """Convert 3D logits to predictions.
        
        Args:
            logits: Model output logits with shape (N, C, Z, Y, X)
            
        Returns:
            Prediction tensor with shape (N, Z, Y, X)
        """
        out_channels = logits.shape[1]
        
        # Validate binary threshold configuration
        if out_channels > 1 and self.binary_segment_threshold is not None:
            raise ValueError(
                f"Multi-class model (output channels={out_channels}) should not set "
                f"binary_segment_threshold, current value={self.binary_segment_threshold}, should be None"
            )
        if out_channels == 1 and self.binary_segment_threshold is None:
            raise ValueError(
                f"Binary model (output channels={out_channels}) must set binary_segment_threshold, "
                "current value=None"
            )
        
        if out_channels > 1:  # Multi-class
            predictions = logits.argmax(dim=1)
        else:  # Binary
            assert self.binary_segment_threshold is not None, "Binary threshold must be set for single-channel output"
            predictions = (torch.sigmoid(logits.squeeze(1)) > self.binary_segment_threshold).long()
        
        return predictions

    def _store_predictions(self, data_sample: SegmentationDataElement, prediction: Tensor, logits: Tensor) -> SegmentationDataElement:
        """Store 3D predictions in data sample.
        
        Args:
            data_sample: Data sample to store predictions in
            prediction: Prediction tensor
            logits: Logits tensor
        
        Returns:
            Updated data sample
        """
        data_sample.seg_logits = VoxelData(logits)
        data_sample.pred_sem_seg = VoxelData(prediction.unsqueeze(0))  # Add channel dimension
        return data_sample
