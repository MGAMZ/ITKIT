"""
Example usage of Segmentation3DVisualizationCallback

This example shows how to use the visualization callback with TensorBoard logger.
"""

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor

# Your imports (adjust paths as needed)
from mgamdata.lightning.task import Segmentation3D
from mgamdata.lightning.callback import Segmentation3DVisualizationCallback
# ... other imports ...

def main():
    # Create your model, dataset, etc.
    # ... (your existing code) ...
    
    # Create TensorBoard logger
    logger = TensorBoardLogger(
        save_dir='./logs',
        name='segmentation_experiment',
        version=None,
        default_hp_metric=False
    )
    
    # Create visualization callback
    viz_callback = Segmentation3DVisualizationCallback(
        log_every_n_batches=20,  # Visualize every 20 batches
        log_every_n_epochs=1,    # Every epoch
        max_samples_per_epoch=3, # Max 3 samples per epoch
        figsize=(16, 12),        # Figure size
        cmap_image='gray',       # Grayscale for original images
        cmap_segmentation='tab10' # Colorful for segmentation masks
    )
    
    trainer = Trainer(
        max_epochs=100,
        accelerator='gpu',
        precision='16-mixed',
        logger=logger,
        callbacks=[
            DeviceStatsMonitor(),
            viz_callback
        ],
        # ... other trainer args ...
    )
    
    # Train the model
    trainer.fit(task, datamodule)
    
    # View results with TensorBoard
    print("To view visualizations, run:")
    print("tensorboard --logdir=./logs")
    print("Then open http://localhost:6006 in your browser")

if __name__ == "__main__":
    main()
