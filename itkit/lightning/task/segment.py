import pdb
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
import pytorch_lightning as pl


@dataclass
class SlideWindowConfig:
    patch_size: list[int]
    patch_stride: list[int]
    patch_accumulate_device: torch.device = torch.device('cpu')
    patch_inference_device: torch.device = torch.device('cuda')
    num_patches_per_inference: int = 1  # Only used in training with slide window
    argmax_batchsize: int | None = None  # None means no batching, direct argmax on class dim.


def ArgMaxBatchedCalculator(logits: Tensor,
                            batch_size: int | None,
                            accumulate_device: torch.device,
                            calculate_device: torch.device) -> Tensor:
    # Fallback to normal argmax on class dim.
    if batch_size is None:
        return logits.argmax(dim=1).to(torch.uint8)
    
    # Batched argmax on last dim (e.g. for large 3D volumes)
    else:
        N, spatial_channels = logits.size(0), logits.shape[2:]
        predictions = torch.empty((N, *spatial_channels), dtype=torch.uint8, device=accumulate_device)
        
        L = spatial_channels[-1]
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            predictions[..., start:end] = logits[..., start:end].to(
                calculate_device).argmax(dim=1).to(dtype=torch.uint8, device=accumulate_device)
        
        return predictions


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
        criterion: torch.nn.Module | Sequence[torch.nn.Module],
        num_classes: int,
        optimizer_config: dict = {},
        scheduler_config: dict = {},
        gt_sem_seg_key: str = 'label',
        to_one_hot:bool = False,
        binary_segment_threshold: float | None = None,
        slide_window_config: SlideWindowConfig | None = None,
        class_names: list[str] | None = None,
        eps = 1e-5,
        fwd_dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.model = model
        if isinstance(criterion, Sequence):
            self.criteria = list(criterion)
        else:
            self.criteria = [criterion]
        
        self.num_classes = num_classes
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.gt_sem_seg_key = gt_sem_seg_key
        self.to_one_hot = to_one_hot
        self.binary_segment_threshold = binary_segment_threshold
        # Unified slide window configuration (replaces scattered patch_* args)
        self.slide_window_config = slide_window_config
        self.class_names = class_names or list(range(num_classes))
        self.eps = eps
        self.fwd_dtype = fwd_dtype
        
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits from model
        """
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizer_config)

    def _logits_to_predictions(self, logits: Tensor) -> Tensor:
        if self.num_classes > 1:
            return ArgMaxBatchedCalculator(logits,
                                           self.slide_window_config.argmax_batchsize,
                                           self.slide_window_config.patch_accumulate_device,
                                           self.slide_window_config.patch_inference_device)
        else:
            assert self.binary_segment_threshold is not None, "Binary segmentation requires a threshold"
            return (logits > self.binary_segment_threshold).to(torch.uint8)

    def _parse_batch(self, batch: dict[str, Any], device:torch.device) -> tuple[Tensor, Tensor]:
        image = batch['image']
        label = batch[self.gt_sem_seg_key].to(dtype=self.fwd_dtype)
        if self.to_one_hot:
            label = torch.nn.functional.one_hot(label.long(), num_classes=self.num_classes).permute(0,4,1,2,3).to(dtype=self.fwd_dtype)
        return image, label

    def _compute_losses(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        losses = {}
        for i, criterion in enumerate(self.criteria):
            loss_name = f'loss_{criterion.__class__.__name__}' if len(self.criteria) > 1 else 'loss'
            losses[loss_name] = criterion(logits, targets)
        return losses

    @torch.no_grad()
    def _compute_metrics(self, predictions: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """
        predictions: shape [N, ...]
        targets: shape [N, C, ...]
        """
        metrics = {}
        
        if self.num_classes > 1:
            for class_idx, class_name in enumerate(self.class_names):
                pred_mask = predictions == class_idx
                target_mask = targets[:, class_idx].bool()
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                
                metrics[f'IoU_{class_name}'] = intersection / (union + self.eps)
                metrics[f'Dice_{class_name}'] = 2 * intersection / (pred_mask.float().sum() + target_mask.float().sum() + self.eps)
                metrics[f'Recall_{class_name}'] = intersection / (target_mask.float().sum() + self.eps)
                metrics[f'Precision_{class_name}'] = intersection / (pred_mask.float().sum() + self.eps)
            
            metrics[f"IoU_Avg"] = torch.mean(torch.stack([metrics[f'IoU_{class_name}'] for class_name in self.class_names]))
            metrics[f"Dice_Avg"] = torch.mean(torch.stack([metrics[f'Dice_{class_name}'] for class_name in self.class_names]))
            metrics[f"Recall_Avg"] = torch.mean(torch.stack([metrics[f'Recall_{class_name}'] for class_name in self.class_names]))
            metrics[f"Precision_Avg"] = torch.mean(torch.stack([metrics[f'Precision_{class_name}'] for class_name in self.class_names]))

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
            metrics[f'Precision'] = intersection / (pred_mask.float().sum() + self.eps)
        
        return metrics

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        image, gt_segs = self._parse_batch(batch, self.device)
        logits: Tensor = self(image)
        
        losses = self._compute_losses(logits, gt_segs)
        total_loss = torch.stack(list(losses.values())).sum()
        
        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                self.log(f'train/{loss_name}', loss_value.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('train/total_loss', total_loss.item(), prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        image, gt_segs = self._parse_batch(batch, self.slide_window_config.patch_accumulate_device)
        logits = self.inference(image)
        
        losses = self._compute_losses(logits, gt_segs)
        for loss_name, loss_value in losses.items():
            self.log(f'val/{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        predictions = self._logits_to_predictions(logits) # [N, ...]
        
        metrics = self._compute_metrics(predictions, gt_segs)
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return {'val_loss': sum(losses.values()),
                'logits': logits,
                'predictions': predictions,
                'targets': gt_segs}

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        batch['predictions'] = self._logits_to_predictions(self.inference(batch['image']))
        return batch

    @torch.inference_mode()
    def inference(self, inputs: Tensor) -> Tensor:
        if self.slide_window_config.patch_size is not None and self.slide_window_config.patch_stride is not None:
            return self.slide_inference(inputs)
        else:
            return self.forward(inputs)

    @abstractmethod
    def slide_inference(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement slide_inference method")


class Segmentation3D(SegmentationBase):
    def slide_inference(self, inputs: Tensor) -> Tensor:
        if (self.slide_window_config.patch_size is None or self.slide_window_config.patch_stride is None or 
            len(self.slide_window_config.patch_size) != 3 or len(self.slide_window_config.patch_stride) != 3):
            raise ValueError("3D inference requires 3D patch size and stride")
        
        z_stride, y_stride, x_stride = self.slide_window_config.patch_stride
        z_crop, y_crop, x_crop = self.slide_window_config.patch_size
        batch_size, _, z_img, y_img, x_img = inputs.size()
        
        # Get output channels from a small forward pass
        with torch.no_grad():
            temp_input = inputs[:, :, :min(z_crop, z_img), :min(y_crop, y_img), :min(x_crop, x_img)]
            temp_output = self.forward(temp_input.to(self.device))
            out_channels = temp_output.size(1)
        
        # Calculate grid numbers
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        
        # Initialize accumulation tensors
        preds = torch.zeros(
            size = (batch_size, out_channels, z_img, y_img, x_img),
            dtype = torch.float16,
            device = self.slide_window_config.patch_accumulate_device,
            pin_memory = False,
        )
        count_mat = torch.zeros(
            size = (batch_size, 1, z_img, y_img, x_img),
            dtype = torch.uint8,
            device = self.slide_window_config.patch_accumulate_device,
            pin_memory = False,
        )
        # Perform Device2Host copy with Host's pin memory is much faster than with normal RAM area.
        patch_cache = torch.empty(
            size = (batch_size, out_channels, z_crop, y_crop, x_crop),
            dtype = torch.float16,
            device = self.slide_window_config.patch_accumulate_device,
            pin_memory = True,
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
                    patch = inputs[:, :, z1:z2, y1:y2, x1:x2].to(self.device)
                    
                    # Forward
                    # prevent crop_logits of previous patch inference from being overlapped by next patch copy
                    # TODO NOT SURE IF THIS STILL HAPPEN, This is only observed when using `.copy(non_blocking=True)`.
                    torch.cuda.synchronize()
                    crop_logits = self.forward(patch)
                    
                    # Accumulate results
                    # Device to Host's pin memory copy
                    preds[:, :, z1:z2, y1:y2, x1:x2] += patch_cache.copy_(crop_logits, non_blocking=True)
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1
        
        # Average overlapping predictions
        assert torch.all(count_mat > 0), "Some areas not covered by sliding window"
        logits = preds / count_mat
        
        return logits.to(inputs.device)


class Seg3D_SlideWindowTrain(Segmentation3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False # manual backward and step is deployed in this mode.
        self.strict_loading = False

    @torch.no_grad()
    def _parse_batch_slide_window(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert self.slide_window_config.patch_size is not None, "Patch size must be set for sliding window training"
        assert self.slide_window_config.patch_stride is not None, "Patch stride must be set for sliding window training"

        image_cpu = batch['image']
        label_cpu = batch[self.gt_sem_seg_key].half() # (N, ...) without channel dim.
        if self.to_one_hot:
            classes = torch.arange(self.num_classes, device=label_cpu.device, dtype=label_cpu.dtype)
            shape = [1, self.num_classes] + [1] * (label_cpu.ndim - 1) # [1, C, 1, 1, ...], will be auto broadcasted
            label_cpu = (label_cpu.unsqueeze(1) == classes.view(*shape)).to(torch.uint8)
    
        # Calc slide window index
        batch_size, _, Z, Y, X = image_cpu.shape
        z_crop, y_crop, x_crop = self.slide_window_config.patch_size
        z_stride, y_stride, x_stride = self.slide_window_config.patch_stride
        z_grids = max(Z - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(Y - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(X - x_crop + x_stride - 1, 0) // x_stride + 1

        indices = []
        for z_idx in range(z_grids):
            for y_idx in range(y_grids):
                for x_idx in range(x_grids):
                    z1 = z_idx * z_stride
                    y1 = y_idx * y_stride
                    x1 = x_idx * x_stride
                    z2 = min(z1 + z_crop, Z)
                    y2 = min(y1 + y_crop, Y)
                    x2 = min(x1 + x_crop, X)
                    z1 = max(z2 - z_crop, 0)
                    y1 = max(y2 - y_crop, 0)
                    x1 = max(x2 - x_crop, 0)
                    indices.append((z1, z2, y1, y2, x1, x2))

        return {
            'image': image_cpu,
            'label': label_cpu,
            'patch_indices': indices
        }

    @torch.no_grad()
    def _train_step_conclude_loss(self, loss_detached_list: list[dict]):
        """ average all mini-batch losses collected in loss_detached_list (each entry is dict) """
        avg_losses: dict[str, float] = {}
        n_entries = len(loss_detached_list)
        if n_entries > 0:
            # sum per-key
            sums: dict[str, float] = {}
            for entry in loss_detached_list:
                for k, v in entry.items():
                    sums[k] = sums.get(k, 0.0) + float(v)

            for k, s in sums.items():
                avg_losses[k] = s / float(n_entries)

        # log averaged losses
        total_loss_val = 0.0
        for k, v in avg_losses.items():
            self.log(f'train/{k}', v, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            total_loss_val += float(v)

        # also log total
        self.log('train/total_loss', total_loss_val, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        # return a tensor for compatibility
        return torch.tensor(total_loss_val)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        parsed = self._parse_batch_slide_window(batch)
        image_cpu: Tensor = parsed['image']
        label_cpu: Tensor = parsed['label']
        slide_window_indices = parsed['patch_indices']
        batch_size = image_cpu.size(0)
        if len(slide_window_indices) == 0:
            raise RuntimeError("No patches computed for sliding window")

        opt = self.optimizers()
        if isinstance(opt, list):
            for o in opt:
                o.zero_grad()
        else:
            opt.zero_grad()
        
        loss_detached_list: list[dict] = []
        batch_patches: list[Tensor] = []
        batch_targets: list[Tensor] = []

        # allow external control of how many patches per forward if provided by slide_window
        max_patches_per_infer = self.slide_window_config.num_patches_per_inference

        def flush_batch():
            nonlocal batch_patches, batch_targets, loss_detached_list
            if len(batch_patches) == 0:
                return

            # host to device
            image_on_device = torch.cat(batch_patches, dim=0)
            target_on_device = torch.cat(batch_targets, dim=0)

            # forward and backward
            logits = self.forward(image_on_device)
            losses = self._compute_losses(logits, target_on_device)
            loss = torch.stack(list(losses.values())).sum()
            self.manual_backward(loss)

            # log
            loss_detached_list.append({k: v.item() for k, v in losses.items()})
            batch_patches.clear()
            batch_targets.clear()

        # Generate neural network input batch and call flusher
        for b in range(batch_size):
            for (z1, z2, y1, y2, x1, x2) in slide_window_indices:
                batch_patches.append(image_cpu[b:b+1, :, z1:z2, y1:y2, x1:x2].to(self.device, non_blocking=True))
                batch_targets.append(label_cpu[b:b+1, ..., z1:z2, y1:y2, x1:x2].to(self.device, non_blocking=True))

                infer_batchsize = self.slide_window_config.num_patches_per_inference
                if max_patches_per_infer is not None:
                    infer_batchsize = min(infer_batchsize, max_patches_per_infer)
                if len(batch_patches) >= infer_batchsize:
                    flush_batch()
            
            # equal to `drop_last=False`
            flush_batch()
        
        # optimizer step (support single optimizer or list)
        if isinstance(opt, list):
            for o in opt:
                o.step()
            for o in opt:
                o.zero_grad()
        else:
            opt.step()
            opt.zero_grad()

        total_loss = self._train_step_conclude_loss(loss_detached_list)

        return total_loss.to(self.device)
