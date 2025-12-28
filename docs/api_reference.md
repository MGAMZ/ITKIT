# API Reference

This page provides detailed API documentation for ITKIT's core modules.

## IO Toolkit

ITKIT provides IO utilities for different medical image formats.

### SimpleITK Toolkit

Module: `itkit.io.sitk_toolkit`

The SimpleITK toolkit provides comprehensive functions for reading, writing, and manipulating medical images.

#### Reading Images

```python
from itkit.io import sitk_toolkit

# Read a single image
image = sitk_toolkit.read_image("path/to/image.mha")

# Read image with specific pixel type
image = sitk_toolkit.read_image("path/to/image.nii.gz", pixel_type=sitk.sitkFloat32)
```

#### Writing Images

```python
# Write image
sitk_toolkit.write_image(image, "path/to/output.mha")

# Write with compression
sitk_toolkit.write_image(image, "path/to/output.nii.gz", use_compression=True)
```

#### Image Metadata

```python
# Get spacing (in mm)
spacing = sitk_toolkit.get_spacing(image)  # Returns (z, y, x)

# Get size (in voxels)
size = sitk_toolkit.get_size(image)  # Returns (z, y, x)

# Get origin
origin = sitk_toolkit.get_origin(image)  # Returns (z, y, x)

# Get direction matrix
direction = sitk_toolkit.get_direction(image)

# Get pixel type
pixel_type = sitk_toolkit.get_pixel_type(image)
```

#### Image Transformations

```python
# Resample image
resampled = sitk_toolkit.resample_image(
    image,
    new_spacing=(1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear
)

# Change orientation
oriented = sitk_toolkit.orient_image(image, orientation="LPI")

# Crop image
cropped = sitk_toolkit.crop_image(image, lower_bound=(0, 0, 0), upper_bound=(100, 100, 100))
```

---

### DICOM Toolkit

Module: `itkit.io.dcm_toolkit`

The DICOM toolkit provides functions for reading and processing DICOM series.

#### Reading DICOM Series

```python
from itkit.io import dcm_toolkit

# Read DICOM series from folder
image = dcm_toolkit.read_dicom_series("path/to/dicom/folder")

# Read with specific series UID
image = dcm_toolkit.read_dicom_series(
    "path/to/dicom/folder",
    series_uid="1.2.3.4.5.6.7.8.9"
)
```

#### DICOM Metadata

```python
# Get DICOM tags
tags = dcm_toolkit.get_dicom_tags("path/to/dicom/file.dcm")

# Get specific tag
patient_name = dcm_toolkit.get_tag_value(tags, "PatientName")
study_date = dcm_toolkit.get_tag_value(tags, "StudyDate")
```

#### Converting DICOM

```python
# Convert DICOM series to other format
dcm_toolkit.convert_dicom_to_nifti(
    "path/to/dicom/folder",
    "path/to/output.nii.gz"
)
```

---

### NIfTI Toolkit

Module: `itkit.io.nii_toolkit`

The NIfTI toolkit provides specialized functions for NIfTI format.

#### Reading NIfTI

```python
from itkit.io import nii_toolkit

# Read NIfTI file
image = nii_toolkit.read_nifti("path/to/image.nii.gz")

# Read with nibabel
nib_image = nii_toolkit.read_nifti_nibabel("path/to/image.nii.gz")
```

#### Writing NIfTI

```python
# Write NIfTI file
nii_toolkit.write_nifti(image, "path/to/output.nii.gz")

# Write with specific data type
nii_toolkit.write_nifti(
    image,
    "path/to/output.nii.gz",
    dtype=np.float32
)
```

---

## Dataset Classes

ITKIT provides several dataset classes for different use cases.

### ITKITBaseSegDataset

Base class for segmentation datasets.

```python
from itkit.dataset import ITKITBaseSegDataset

# Create dataset
dataset = ITKITBaseSegDataset(
    root_dir="/path/to/dataset",
    mode="train",  # "train", "val", or "test"
    transform=None,  # Optional transforms
    load_label=True  # Whether to load labels
)

# Access sample
image, label = dataset[0]

# Get dataset length
num_samples = len(dataset)
```

#### Parameters

- `root_dir` (str): Path to dataset root (containing image/ and label/ folders)
- `mode` (str): Dataset split mode
- `transform` (callable, optional): Transform to apply
- `load_label` (bool): Whether to load label files

---

### SeriesVolumeDataset

Dataset for volumetric series data.

```python
from itkit.dataset import SeriesVolumeDataset

# Create dataset
dataset = SeriesVolumeDataset(
    root_dir="/path/to/dataset",
    mode="train",
    sequence_length=10,  # Number of slices per sample
    stride=5  # Stride between sequences
)
```

---

### PatchedDataset

Dataset for pre-extracted patches.

```python
from itkit.dataset import PatchedDataset

# Create dataset from patched data
dataset = PatchedDataset(
    root_dir="/path/to/patches",  # Output from itk_patch
    transform=None
)

# Access patch
image_patch, label_patch = dataset[0]
```

---

### MONAI_PatchedDataset

MONAI-compatible patched dataset.

```python
from itkit.dataset import MONAI_PatchedDataset

dataset = MONAI_PatchedDataset(
    root_dir="/path/to/patches",
    transform=monai_transforms
)
```

---

### TorchIO_PatchedDataset

TorchIO-compatible patched dataset.

```python
from itkit.dataset import TorchIO_PatchedDataset

dataset = TorchIO_PatchedDataset(
    root_dir="/path/to/patches",
    transform=torchio_transforms
)
```

---

### ITKITConcatDataset

Concatenate multiple datasets.

```python
from itkit.dataset import ITKITConcatDataset

# Combine multiple datasets
dataset1 = ITKITBaseSegDataset("/path/to/dataset1")
dataset2 = ITKITBaseSegDataset("/path/to/dataset2")

combined = ITKITConcatDataset([dataset1, dataset2])
```

---

## PyTorch Lightning Extensions

Module: `itkit.lightning`

### RefinedLightningPbar

Refined progress bar for PyTorch Lightning.

```python
from itkit.lightning import RefinedLightningPbar
import pytorch_lightning as pl

# Use in trainer
trainer = pl.Trainer(
    callbacks=[RefinedLightningPbar()]
)
```

Features:
- Cleaner progress display
- Better formatting for metrics
- Reduced console clutter

---

## MMEngine Extensions

Module: `itkit.mm`

### ITKITRunner

Custom runner for ITKIT experiments.

```python
from itkit.mm import ITKITRunner

# Create runner
runner = ITKITRunner(
    model=model,
    work_dir="/path/to/workdir",
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    # ... other parameters
)

# Train
runner.train()
```

### Task Models

#### SemanticSegment

Base semantic segmentation model.

```python
from itkit.mm.task_models import SemanticSegment

model = SemanticSegment(
    backbone=backbone_config,
    decode_head=decode_head_config,
    num_classes=3
)
```

#### SemSeg2D

2D semantic segmentation model.

```python
from itkit.mm.task_models import SemSeg2D

model = SemSeg2D(
    backbone=backbone_config,
    decode_head=decode_head_config,
    num_classes=3
)
```

#### SemSeg3D

3D semantic segmentation model.

```python
from itkit.mm.task_models import SemSeg3D

model = SemSeg3D(
    backbone=backbone_config,
    decode_head=decode_head_config,
    num_classes=3
)
```

---

## Criterions (Loss Functions)

Module: `itkit.criterions`

ITKIT provides custom loss functions for medical image segmentation.

### Usage

```python
from itkit.criterions import DiceLoss, FocalLoss

# Dice loss
dice_loss = DiceLoss(smooth=1.0)
loss = dice_loss(predictions, targets)

# Focal loss
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
loss = focal_loss(predictions, targets)
```

Common loss functions:
- `DiceLoss`: Dice coefficient loss
- `FocalLoss`: Focal loss for imbalanced classes
- `TverskyLoss`: Tversky loss
- `ComboLoss`: Combination of multiple losses

---

## Utilities

Module: `itkit.utils`

Various utility functions for common tasks.

### Visualization

```python
from itkit.utils import visualize_segmentation

# Visualize segmentation result
visualize_segmentation(
    image,
    prediction,
    ground_truth,
    save_path="output.png"
)
```

### Metrics

```python
from itkit.utils import compute_dice, compute_iou

# Compute Dice coefficient
dice = compute_dice(prediction, ground_truth)

# Compute IoU
iou = compute_iou(prediction, ground_truth)
```

---

## Command-Line Tools

All command-line tools are documented in the [Preprocessing Guide](preprocessing.md).

Available commands:
- `itk_check`: Check dataset integrity
- `itk_resample`: Resample images
- `itk_orient`: Orient images
- `itk_patch`: Extract patches
- `itk_aug`: Data augmentation
- `itk_extract`: Extract label classes
- `itk_convert`: Convert formats
- `itkit-app`: Launch GUI
- `mmrun`: Run experiments

---

## Next Steps

- [Preprocessing Guide](preprocessing.md) - Learn about command-line tools
- [Framework Integration](framework_integration.md) - Integrate with deep learning frameworks
- [Models](models.md) - Explore available models
