# FAQ and Troubleshooting

Frequently asked questions and common issues with ITKIT.

## Installation Issues

### Q: I get import errors after installation

**A:** Try reinstalling ITKIT with force-reinstall:

```bash
pip install itkit --force-reinstall
```

If the issue persists, check that all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Q: SimpleITK installation fails

**A:** SimpleITK requires specific system libraries. Try:

```bash
# Update pip first
pip install --upgrade pip

# Install SimpleITK
pip install SimpleITK
```

On Linux, you may need system packages:

```bash
sudo apt-get install python3-dev
```

### Q: PyQt6 GUI won't start

**A:** Ensure you installed the GUI dependencies:

```bash
pip install "itkit[gui]"
```

If running on a server without display:

```bash
export QT_QPA_PLATFORM=offscreen
itkit-app
```

### Q: Version conflicts with other packages

**A:** Use a virtual environment to isolate ITKIT:

```bash
python -m venv itkit_env
source itkit_env/bin/activate  # On Windows: itkit_env\Scripts\activate
pip install itkit
```

## Usage Issues

### Q: itk_check reports mismatched spacing but images look correct

**A:** Medical images can have very small spacing differences due to floating-point precision. Use tolerance in your checks:

```bash
# Instead of exact values, use ranges
itk_check check /data --min-spacing 0.9 0.9 0.9 --max-spacing 1.1 1.1 1.1
```

### Q: itk_resample produces incorrect output

**A:** Check:
1. **Coordinate order:** ITKIT uses Z, Y, X order
2. **Field type:** Use `dataset` for both images and labels, or specify `image`/`label` appropriately
3. **Spacing values:** Ensure they're in millimeters

Correct usage:

```bash
itk_resample dataset /src /dst --spacing 1.0 1.0 1.0  # Z Y X order
```

### Q: Patches extracted with itk_patch are all background

**A:** Adjust the foreground ratio threshold:

```bash
itk_patch /src /dst \
    --patch-size 96 96 96 \
    --patch-stride 48 48 48 \
    --minimum-foreground-ratio 0.01  # Lower threshold
```

Or keep some empty patches:

```bash
itk_patch /src /dst \
    --patch-size 96 96 96 \
    --patch-stride 48 48 48 \
    --keep-empty-label-prob 0.2
```

### Q: Multiprocessing (--mp) doesn't speed things up

**A:** Multiprocessing overhead can exceed benefits for small datasets. Use it only when:
- Dataset has many samples (>50)
- Individual files are large
- I/O is not the bottleneck

Control number of workers:

```bash
itk_resample dataset /src /dst --spacing 1.0 1.0 1.0 --mp --workers 4
```

### Q: GUI DPI is too small/large

**A:** Set the Qt scale factor:

```bash
# Double size
QT_SCALE_FACTOR=2 itkit-app

# Half size
QT_SCALE_FACTOR=0.5 itkit-app
```

## Dataset Issues

### Q: My dataset structure doesn't match ITKIT format

**A:** You need to reorganize your data. ITKIT requires:

```
dataset/
├── image/
│   └── files
└── label/
    └── files
```

Use symbolic links if you don't want to copy:

```bash
mkdir -p dataset/image dataset/label
ln -s /original/images/* dataset/image/
ln -s /original/labels/* dataset/label/
```

Or use `itk_check` in symlink mode:

```bash
itk_check symlink /original/mixed --output /dataset/organized
```

### Q: Image and label have different sizes

**A:** This indicates preprocessing issues. Ensure:
1. Labels were created from the same source images
2. Both underwent the same preprocessing
3. Both have matching metadata

To fix, resample both to the same space:

```bash
itk_resample dataset /src /dst --spacing 1.0 1.0 1.0
```

### Q: Conversion to MONAI/TorchIO format fails

**A:** Verify:
1. Input follows ITKIT format (image/ and label/ folders)
2. File names match between image/ and label/
3. You have write permissions in output directory

Debug by converting a single sample manually:

```python
from itkit.io import sitk_toolkit
import SimpleITK as sitk

image = sitk.ReadImage("dataset/image/case001.mha")
sitk.WriteImage(image, "test_output.nii.gz")
```

## Framework Integration Issues

### Q: OpenMMLab imports fail

**A:** Install the OneDL redistributions:

```bash
pip install "itkit[advanced]"
```

### Q: MMEngine experiments won't start

**A:** Check that required variables are set:

```python
# In your experiment script
mm_workdir = "/path/to/workdir"
mm_testdir = "/path/to/testdir"
mm_configdir = "/path/to/configs"
```

And verify config directory structure:

```
configs/
└── 0.1.MyExperiment/
    ├── mgam.py
    └── model.py
```

### Q: MONAI transforms not working with ITKIT datasets

**A:** Ensure you're using MONAI-compatible dataset class:

```python
from itkit.dataset import MONAI_PatchedDataset  # Not ITKITBaseSegDataset

dataset = MONAI_PatchedDataset(
    root_dir="/data/patches",
    transform=monai_transforms
)
```

### Q: PyTorch Lightning trainer fails

**A:** Install MONAI (required for Lightning extensions):

```bash
pip install --no-deps monai
```

## Performance Issues

### Q: Processing is very slow

**A:** Try these optimizations:

1. **Use multiprocessing:**
   ```bash
   itk_resample dataset /src /dst --spacing 1.0 1.0 1.0 --mp --workers 8
   ```

2. **Use faster file formats:**
   - `.mha` is faster than `.nii.gz` (no compression overhead)
   - Avoid `.mhd` with large `.raw` files on network storage

3. **Reduce I/O:**
   - Work on local disk, not network storage
   - Use SSD instead of HDD

4. **Batch operations:**
   - Process entire directories instead of individual files

### Q: Out of memory errors

**A:** Solutions:

1. **Extract patches instead of loading full volumes:**
   ```bash
   itk_patch /data /patches --patch-size 96 96 96
   ```

2. **Reduce batch size in training**

3. **Use gradient checkpointing** in model training

4. **Process files sequentially** (don't use --mp)

## Model Training Issues

### Q: Model training crashes with CUDA out of memory

**A:** Reduce memory usage:

```python
# Smaller batch size
batch_size = 1

# Smaller patch size
patch_size = (64, 64, 64)  # Instead of (128, 128, 128)

# Mixed precision training
use_amp = True

# Gradient checkpointing
model.enable_gradient_checkpointing()
```

### Q: Validation metrics are NaN or inf

**A:** Check:
1. **Label values:** Should be integers 0, 1, 2, ... (not one-hot)
2. **Normalization:** Images should be normalized appropriately
3. **Loss function:** Ensure it matches your task
4. **Learning rate:** May be too high

### Q: Model converges but predictions are all background

**A:** This indicates class imbalance. Solutions:

1. **Use weighted loss:**
   ```python
   loss = FocalLoss(alpha=0.25, gamma=2.0)
   ```

2. **Filter patches during extraction:**
   ```bash
   itk_patch /data /patches \
       --minimum-foreground-ratio 0.1 \
       --keep-empty-label-prob 0.1
   ```

3. **Adjust class weights** in loss function

## File Format Issues

### Q: Cannot read .dcm files

**A:** DICOM files need special handling:

```python
from itkit.io import dcm_toolkit

# Read DICOM series (not individual files)
image = dcm_toolkit.read_dicom_series("/path/to/dicom/folder")
sitk.WriteImage(image, "output.mha")
```

### Q: .nii.gz files are huge

**A:** NIfTI compression varies. To reduce size:

```bash
# Convert to .mha (often smaller)
itk_convert format mha /data/nifti /data/mha

# Or use higher compression
import gzip
# Compress with maximum compression level
```

### Q: File extensions don't match content

**A:** Use ITKIT's conversion to standardize:

```bash
itk_convert format mha /data/mixed /data/standardized
```

## Getting More Help

If your issue isn't covered here:

1. **Check documentation:**
   - [Installation Guide](installation.md)
   - [Quick Start](quickstart.md)
   - [Preprocessing Guide](preprocessing.md)
   - [API Reference](api_reference.md)

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/MGAMZ/ITKIT/issues)

3. **Ask for help:**
   - Open a new issue with detailed description
   - Include error messages and minimal reproducible example
   - Contact: [312065559@qq.com](mailto:312065559@qq.com)

4. **Report bugs:**
   - Follow the [Contributing Guide](contributing.md)
   - Provide system information (OS, Python version, ITKIT version)

## Next Steps

- [Quick Start Guide](quickstart.md) - Learn basic usage
- [Preprocessing Guide](preprocessing.md) - Detailed command documentation
- [Contributing Guide](contributing.md) - Report issues or contribute fixes
