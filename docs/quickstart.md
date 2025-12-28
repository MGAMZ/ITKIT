# Quick Start Guide

This guide will help you get started with ITKIT's core functionality.

## Basic Usage

### Command-Line Interface

ITKIT provides several command-line tools for medical image preprocessing. All commands follow a consistent interface pattern.

To see available options for any command, use the `--help` flag:

```bash
itk_check --help
itk_resample --help
itk_orient --help
```

### Simple Example: Checking Dataset

Check if your dataset meets specific spacing and size requirements:

```bash
itk_check check /path/to/dataset \
    --min-spacing 0.5 0.5 0.5 \
    --max-spacing 2.0 2.0 2.0
```

### Simple Example: Resampling

Resample images to a target spacing:

```bash
itk_resample dataset /path/to/source /path/to/destination \
    --spacing 1.0 1.0 1.0
```

### Simple Example: Orientation

Orient images to a standard orientation (e.g., LPI):

```bash
itk_orient /path/to/source /path/to/destination LPI
```

## Using the GUI Application

ITKIT provides a PyQt6-based graphical user interface for all preprocessing operations.

### Installing GUI Support

First, install ITKIT with GUI support:

```bash
pip install "itkit[gui]"
```

### Launching the GUI

```bash
itkit-app
```

### Adjusting GUI DPI

If the GUI's DPI is not optimal, specify the `QT_SCALE_FACTOR` environment variable:

```bash
QT_SCALE_FACTOR=2 itkit-app
```

![ITKIT GUI](itkit-gui.png)

The GUI provides an intuitive interface for:

- Checking dataset integrity
- Resampling images
- Orienting images
- Extracting patches
- Data augmentation
- Label extraction
- Format conversion

## Python API Usage

You can also use ITKIT programmatically in your Python scripts.

### Reading and Writing Images

```python
from itkit.io import sitk_toolkit as sitk_io

# Read an image
image = sitk_io.read_image("/path/to/image.mha")

# Get image properties
spacing = sitk_io.get_spacing(image)
size = sitk_io.get_size(image)
origin = sitk_io.get_origin(image)

# Write an image
sitk_io.write_image(image, "/path/to/output.mha")
```

### Working with DICOM

```python
from itkit.io import dcm_toolkit as dcm_io

# Read DICOM series
image = dcm_io.read_dicom_series("/path/to/dicom/folder")

# Convert to other format
dcm_io.write_image(image, "/path/to/output.nii.gz")
```

### Dataset Operations

```python
from itkit.dataset import ITKITBaseSegDataset

# Create dataset instance
dataset = ITKITBaseSegDataset(
    root_dir="/path/to/dataset",
    mode="train"
)

# Access samples
image, label = dataset[0]
```

## Common Workflows

### Workflow 1: Preparing Data for Training

1. **Check dataset integrity:**

   ```bash
   itk_check check /data/raw_dataset --min-size 32 32 32
   ```

2. **Resample to uniform spacing:**

   ```bash
   itk_resample dataset /data/raw_dataset /data/resampled \
       --spacing 1.0 1.0 1.0 --mp
   ```

3. **Orient to standard direction:**

   ```bash
   itk_orient /data/resampled /data/oriented LPI --mp
   ```

4. **Extract patches for training:**

   ```bash
   itk_patch /data/oriented /data/patches \
       --patch-size 96 96 96 \
       --patch-stride 48 48 48 \
       --minimum-foreground-ratio 0.1 \
       --mp
   ```

### Workflow 2: Converting Dataset Format

1. **Convert to MONAI format:**

   ```bash
   itk_convert monai /data/itkit_dataset /data/monai_dataset \
       --name MyDataset \
       --modality CT \
       --labels background liver tumor \
       --mp
   ```

2. **Convert file format:**

   ```bash
   itk_convert format nii.gz /data/mha_dataset /data/nifti_dataset \
       --mp --workers 8
   ```

### Workflow 3: Data Augmentation

Generate augmented samples with random rotations:

```bash
itk_aug /data/images /data/labels \
    -oimg /data/aug_images \
    -olbl /data/aug_labels \
    -n 5 \
    --random-rot 15 15 15 \
    --mp
```

## Multiprocessing Support

Most ITKIT commands support multiprocessing with the `--mp` flag for faster processing:

```bash
itk_resample dataset /src /dst --spacing 1.0 1.0 1.0 --mp --workers 8
```
