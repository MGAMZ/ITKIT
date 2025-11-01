import pdb
from typing import Any

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import Callback
import pytorch_lightning as pl

from ..task.segment import Segmentation3D


class SegVis3DCallback(Callback):
    """Callback for visualizing 3D segmentation results during validation and testing.

    This callback creates visualizations showing:
    - Original image
    - Ground truth segmentation
    - Predicted segmentation
    - Prediction confidence map

    For each sample, it shows 3 axial slices (Z-axis) at 1/4, 1/2, and 3/4 positions.
    """

    def __init__(
        self,
        log_every_n_batches: int = 10,
        log_every_n_epochs: int = 1,
        gt_key: str = "label",
        max_samples_per_epoch: int = 5,
        figsize: tuple[int, int] = (16, 12),
        cmap_image: str = "gray",
        cmap_segmentation: str = "tab10",
        alpha: float = 0.6,
        ignore_class_idx: int = 0,
    ):
        """Initialize the visualization callback.

        Args:
            log_every_n_batches: Log visualization every N batches
            log_every_n_epochs: Log visualization every N epochs
            gt_key: Key in the batch dict for ground truth segmentation, with no channel dim.
            max_samples_per_epoch: Maximum number of samples to visualize per epoch
            figsize: Figure size for matplotlib plots
            cmap_image: Colormap for original images
            cmap_segmentation: Colormap for segmentation masks
            alpha: Alpha value for overlay visualizations
            ignore_class_idx: Class index to ignore in visualization (default: 0 for background)
        """

        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.gt_key = gt_key
        self.max_samples_per_epoch = max_samples_per_epoch
        self.figsize = figsize
        self.cmap_image = cmap_image
        self.cmap_segmentation = cmap_segmentation
        self.alpha = alpha
        self.ignore_class_idx = ignore_class_idx

        self.samples_visualized_this_epoch = 0

    def _create_masked_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Create a masked segmentation array where ignore_class_idx is set to NaN for transparency.

        Args:
            segmentation: Segmentation array of shape [Z, Y, X]

        Returns:
            Masked segmentation array where ignore_class_idx pixels are NaN
        """
        masked_seg = segmentation.astype(float)
        masked_seg[segmentation == self.ignore_class_idx] = np.nan
        return masked_seg

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: Segmentation3D
    ) -> None:
        self.samples_visualized_this_epoch = 0

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: Segmentation3D
    ) -> None:
        self.samples_visualized_this_epoch = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Segmentation3D,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_visualize(
            trainer, pl_module, outputs, batch, batch_idx, stage="val"
        )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Segmentation3D,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_visualize(
            trainer, pl_module, outputs, batch, batch_idx, stage="test"
        )

    def _maybe_visualize(
        self,
        trainer: pl.Trainer,
        pl_module: Segmentation3D,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> None:
        should_log_batch = batch_idx % self.log_every_n_batches == 0
        should_log_epoch = trainer.current_epoch % self.log_every_n_epochs == 0
        within_sample_limit = (
            self.samples_visualized_this_epoch < self.max_samples_per_epoch
        )
        if (
            not should_log_batch
            or not should_log_epoch
            or not within_sample_limit
            or trainer.logger is None
        ):
            return

        # Extract data from batch and outputs
        images = batch["image"]  # Shape: [N, C, Z, Y, X]
        gt_labels = batch[self.gt_key]  # Shape: [N, 1, Z, Y, X] or [N, Z, Y, X]
        predictions = outputs["predictions"]  # Shape: [N, Z, Y, X]
        logits = outputs["logits"]  # Shape: [N, C, Z, Y, X]
        confidence_map = F.softmax(logits, dim=1).max(dim=1)[0]  # Shape: [N, Z, Y, X]
        if gt_labels.ndim == 5:
            assert (
                gt_labels.shape[1] == 1
            ), "Expected gt_labels to have shape [N, 1, Z, Y, X] or [N, Z, Y, X]"
            gt_labels = gt_labels.squeeze(1)  # Remove channel dimension if present

        # Take the first sample from the batch
        sample_idx = 0
        image = images[sample_idx, 0].cpu().float().numpy()  # Shape: [Z, Y, X]
        gt_label = gt_labels[sample_idx].cpu().float().numpy()  # Shape: [Z, Y, X]
        prediction = predictions[sample_idx].cpu().float().numpy()  # Shape: [Z, Y, X]
        confidence = (
            confidence_map[sample_idx].cpu().float().numpy()
        )  # Shape: [Z, Y, X]
        # masked segmentation map for transparency during matplotlib visualization
        gt_label[gt_label == self.ignore_class_idx] = np.nan
        prediction[prediction == self.ignore_class_idx] = np.nan

        # Get slice indices (1/4, 1/2, 3/4 of Z dimension)
        z_size = image.shape[0]
        slice_indices = [z_size // 4, z_size // 2, 3 * z_size // 4]

        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=self.figsize)
        fig.suptitle(
            f"{stage.capitalize()} Visualization - Batch {batch_idx}, Epoch {trainer.current_epoch}",
            fontsize=16,
        )

        # Column titles
        col_titles = ["Original Image", "Ground Truth", "Prediction", "Confidence Map"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=12, fontweight="bold")

        for row, slice_idx in enumerate(slice_indices):
            # Ensure slice index is within bounds
            slice_idx = min(slice_idx, z_size - 1)
            # background image
            axes[row, 0].imshow(image[slice_idx], cmap=self.cmap_image)
            axes[row, 1].imshow(image[slice_idx], cmap=self.cmap_image)
            axes[row, 2].imshow(image[slice_idx], cmap=self.cmap_image)
            axes[row, 3].imshow(image[slice_idx], cmap=self.cmap_image)

            # Original image
            axes[row, 0].set_ylabel(f"Slice {slice_idx}", fontsize=10)
            axes[row, 0].axis("off")
            # Ground truth overlay on image
            axes[row, 1].imshow(
                gt_label[slice_idx],
                cmap=self.cmap_segmentation,
                vmin=0,
                vmax=pl_module.num_classes - 1,
                alpha=self.alpha,
            )
            axes[row, 1].axis("off")
            # Prediction overlay on image
            axes[row, 2].imshow(
                prediction[slice_idx],
                cmap=self.cmap_segmentation,
                vmin=0,
                vmax=pl_module.num_classes - 1,
                alpha=self.alpha,
            )
            axes[row, 2].axis("off")
            # Confidence map
            im = axes[row, 3].imshow(
                confidence[slice_idx], cmap="hot", vmin=0, vmax=1, alpha=self.alpha
            )
            axes[row, 3].axis("off")

        fig.tight_layout()

        for logger in trainer.loggers:
            if hasattr(logger, "log_figure"):
                logger.log_figure(
                    f"{stage}_SegVis3D/batch_{batch_idx}", fig, trainer.global_step
                )

        plt.close(fig)
        self.samples_visualized_this_epoch += 1

    def state_dict(self) -> dict[str, Any]:
        return {"samples_visualized_this_epoch": self.samples_visualized_this_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.samples_visualized_this_epoch = state_dict.get(
            "samples_visualized_this_epoch", 0
        )
