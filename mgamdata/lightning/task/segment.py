import pdb

import torch
from torch import Tensor
import pytorch_lightning as pl
from abc import abstractmethod
from typing import Any



class SegmentationBase(pl.LightningModule):
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
        optimizer_config: dict = {},
        scheduler_config: dict = {},
        gt_sem_seg_key: str = 'label',
        binary_segment_threshold: float | None = None,
        inference_patch_size: tuple[int, ...] | None = None,
        inference_patch_stride: tuple[int, ...] | None = None,
        inference_patch_accumulate_device: str = 'cuda',
        class_names: list[str] | None = None,
        eps = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.model = model
        if isinstance(criterion, (list, tuple)):
            self.criteria = list(criterion)
        else:
            self.criteria = [criterion]
        
        self.num_classes = num_classes
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.gt_sem_seg_key = gt_sem_seg_key
        self.binary_segment_threshold = binary_segment_threshold
        self.inference_patch_size = inference_patch_size
        self.inference_patch_stride = inference_patch_stride
        self.inference_patch_accumulate_device = inference_patch_accumulate_device
        self.class_names = class_names
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits from model
        """
        return self.model(x)

    def _logits_to_predictions(self, logits: Tensor) -> Tensor:
        if self.num_classes > 1:
            return logits.argmax(dim=1).to(torch.uint8)
        else:
            assert self.binary_segment_threshold is not None, "Binary segmentation requires a threshold"
            return (logits > self.binary_segment_threshold).to(torch.uint8)

    def _parse_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        image = batch['image'].to(device=torch.device(self.device), non_blocking=True)
        label = batch[self.gt_sem_seg_key].to(device=torch.device(self.device), non_blocking=True)
        return image, label

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        image, gt_segs = self._parse_batch(batch)
        logits:Tensor = self(image)
        losses = self._compute_losses(logits, gt_segs)
        total_loss = sum(losses.values())
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True, logger=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        return total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        image, gt_segs = self._parse_batch(batch)
        logits:Tensor = self(image)
        losses = self._compute_losses(logits, gt_segs)
        for loss_name, loss_value in losses.items():
            self.log(f'val/{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        predictions = self._logits_to_predictions(logits)
        
        metrics = self._compute_metrics(predictions, gt_segs)
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value, on_step=False, on_epoch=True, logger=True)
        
        return {'val_loss': sum(losses.values()), 
                'logits': logits, 
                'predictions': predictions, 
                'targets': gt_segs}

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        batch['predictions'] = self._logits_to_predictions(self.inference(batch['image']))
        return batch

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizer_config)

    @torch.inference_mode()
    def inference(self, inputs: Tensor) -> Tensor:
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
        metrics = {}
        
        if self.num_classes > 1:
            for class_idx in range(self.num_classes):
                class_name = self.class_names[class_idx] if self.class_names is not None else f'class_{class_idx}'
                
                pred_mask = (predictions == class_idx)
                target_mask = (targets == class_idx)
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                
                metrics[f'IoU_{class_name}'] = intersection / (union + self.eps)
                metrics[f'Dice_{class_name}'] = 2 * intersection / (pred_mask.float().sum() + target_mask.float().sum() + self.eps)
                metrics[f'Recall_{class_name}'] = intersection / (target_mask.float().sum() + self.eps)
                metrics[f'precision_{class_name}'] = intersection / (pred_mask.float().sum() + self.eps)
        
        else:
            assert self.binary_segment_threshold is not None, "Binary segmentation requires a threshold"
            class_name = self.class_names[0] if self.class_names is not None else 'binary'
            
            pred_mask = (predictions > self.binary_segment_threshold)
            target_mask = (targets > self.binary_segment_threshold)
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            metrics[f'IoU'] = intersection / (union + self.eps)
            metrics[f'Dice'] = 2 * intersection / (pred_mask.float().sum() + target_mask.float().sum() + self.eps)
            metrics[f'Recall'] = intersection / (target_mask.float().sum() + self.eps)
            metrics[f'precision'] = intersection / (pred_mask.float().sum() + self.eps)
        
        return metrics


class Segmentation3D(SegmentationBase):
    def slide_inference(self, inputs: Tensor) -> Tensor:
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
