# Framework Integration

ITKIT integrates seamlessly with popular deep learning frameworks for medical image analysis. This guide covers integration with OpenMMLab, MONAI, TorchIO, and PyTorch Lightning.

## OpenMMLab Extensions

[OpenMMLab](https://github.com/open-mmlab) is an outstanding open-source deep learning image analysis framework. ITKIT provides a set of OpenMMLab extension classes that define commonly used pipelines and computational modules for medical imaging.

### Important Notice

**The upstream `OpenMMLab` project has gradually fallen out of maintenance.** ITKIT now recommends users to use the `OneDL` redistribution of `OpenMMLab` instead:

- **OneDL-mmengine**
- **OneDL-mmcv**
- **OneDL-mmsegmentation**

### Installation

Install the advanced dependencies to get OpenMMLab support:

```bash
pip install "itkit[advanced]"
```

### Experiment Runner

ITKIT provides an experiment runner based on `MMEngine`'s `Runner` class.

#### Required Global Variables

Set the following global variables in your experiment script:

- `mm_workdir`: Working directory for the experiment (logs, checkpoints, visualizations)
- `mm_testdir`: Directory to store test results (used with `--test` flag)
- `mm_configdir`: Directory where config files are located

#### Configuration Directory Structure

```plaintext
mm_configdir/
├── 0.1.Config1/
│   ├── mgam.py          # Non-model configs (required name)
│   ├── model1.py        # Model config
│   ├── model2.py        # Another model config
│   └── ...
├── 0.2.Config2/
├── 0.3.Config3/
├── 0.3.1.Config3/
└── 0.4.2.3.Config3/
```

**Version Prefix Rules:**
- Every element before the final dot must be numeric
- The suffix after the final dot should not start with a number

#### Optional Variables

- `supported_models`: List of model names to search for in config directories. If not set, all `.py` files except `mgam.py` are treated as model configs.

#### Running Experiments

```bash
# Single node
mmrun $experiment_prefix$

# Multi-node with torchrun
export mmrun=".../itkit/itkit/mm/run.py"
torchrun --nproc_per_node 4 $mmrun $experiment_prefix$
```

Use `mmrun --help` to see all available options.

#### Configuration Format

Configuration files follow the OpenMIM specification. Pure-python style config is recommended. See the [official documentation](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#python-beta) for details.

### Segmentation Framework

ITKIT provides remastered segmentation implementations based on `mmengine` BaseModel, inspired by `mmsegmentation` but more lightweight.

See `itkit/mm/task_models.py` for implementation details.

Available task models:
- `SemanticSegment`: Base semantic segmentation model
- `SemSeg2D`: 2D semantic segmentation
- `SemSeg3D`: 3D semantic segmentation

### MMEngine Plugins

ITKIT includes several plugins located in `itkit/mm/mmeng_PlugIn.py`:

1. **IterBasedTrainLoop_SupportProfiler**: TrainLoop class with profiler support
2. **LoggerJSON**: Test-time logger for quantified metrics
3. **RemasteredDDP/RemasteredFSDP**: Improved distributed training wrappers
4. **RemasteredFSDP_Strategy**: FSDP runtime strategy
5. **RuntimeInfoHook**: More stable runtime logger preventing lr overflow
6. **multi_sample_collate**: Collate function for multi-sample collection
7. **mgam_OptimWrapperConstructor**: Fixed OptimWrapper for efficient parameter iteration

---

## MONAI Integration

[MONAI](https://monai.io/) is a PyTorch-based framework for deep learning in healthcare imaging.

### Installation

```bash
pip install --no-deps monai
```

**Note:** The `itk_convert monai` command does NOT require MONAI to be installed. Install MONAI only if you plan to use MONAI-based training workflows.

### Converting to MONAI Format

Convert ITKIT dataset to MONAI Decathlon format:

```bash
itk_convert monai /data/itkit_dataset /data/monai_dataset \
    --name MyDataset \
    --modality CT \
    --labels background liver tumor \
    --split train \
    --mp
```

### Using with MONAI

After conversion, use MONAI's data loaders:

```python
from monai.data import Dataset, DataLoader, load_decathlon_datalist
import monai.transforms as transforms

# Load dataset
data_list = load_decathlon_datalist(
    data_list_file_path="/data/monai_dataset/dataset.json",
    data_list_key="training"
)

# Define transforms
train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0)),
    transforms.RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=4
    ),
])

# Create dataset
dataset = Dataset(data=data_list, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### ITKIT Dataset Classes for MONAI

ITKIT provides dataset classes compatible with MONAI:

```python
from itkit.dataset import MONAI_PatchedDataset

# Use ITKIT patched dataset with MONAI
dataset = MONAI_PatchedDataset(
    root_dir="/data/patches",
    transform=train_transforms
)
```

---

## TorchIO Integration

[TorchIO](https://torchio.readthedocs.io/) is a Python package for efficient loading, preprocessing, and augmentation of medical images.

### Installation

```bash
pip install torchio
```

**Note:** The `itk_convert torchio` command does NOT require TorchIO to be installed.

### Converting to TorchIO Format

```bash
itk_convert torchio /data/itkit_dataset /data/torchio_dataset --mp
```

### Using with TorchIO

```python
import torchio as tio

# Define transforms
transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.RandomAffine(),
    tio.RandomElasticDeformation(),
    tio.RandomFlip(axes=['LR']),
])

# Load subjects from CSV
subjects_dataset = tio.SubjectsDataset(
    subjects_csv="/data/torchio_dataset/subjects.csv",
    transform=transforms
)

# Create data loader
dataloader = torch.utils.data.DataLoader(
    subjects_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)
```

### ITKIT Dataset Classes for TorchIO

```python
from itkit.dataset import TorchIO_PatchedDataset

# Use ITKIT patched dataset with TorchIO
dataset = TorchIO_PatchedDataset(
    root_dir="/data/patches"
)
```

---

## PyTorch Lightning Integration

ITKIT is transitioning from OpenMIM to PyTorch Lightning for future development.

### Installation

PyTorch Lightning is included in the base ITKIT dependencies:

```bash
pip install itkit
```

### Lightning Extensions

ITKIT provides Lightning extensions in the `itkit/lightning/` module:

```python
from itkit.lightning import RefinedLightningPbar

# Use refined progress bar
trainer = pl.Trainer(
    callbacks=[RefinedLightningPbar()]
)
```

### Status

The Lightning integration is currently in **alpha** stage. More features are being added as the framework transitions from OpenMIM.

**Note:** Install MONAI package before using Lightning extensions:

```bash
pip install --no-deps monai
```

---

## Dataset Classes

ITKIT provides several dataset classes for different frameworks:

### Base Classes

- `ITKITBaseSegDataset`: Base class for segmentation datasets
- `SeriesVolumeDataset`: Dataset for volumetric series
- `PatchedDataset`: Dataset for pre-extracted patches
- `ITKITConcatDataset`: Concatenate multiple datasets

### Framework-Specific

- `MONAI_PatchedDataset`: MONAI-compatible patched dataset
- `TorchIO_PatchedDataset`: TorchIO-compatible patched dataset

### Example Usage

```python
from itkit.dataset import ITKITBaseSegDataset

# Create dataset
train_dataset = ITKITBaseSegDataset(
    root_dir="/data/dataset",
    mode="train",
    transform=my_transforms
)

# Use with PyTorch DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)
```

---

## IO Toolkit

ITKIT provides IO utilities for different medical image formats:

### SimpleITK Toolkit

```python
from itkit.io import sitk_toolkit

# Read/write images
image = sitk_toolkit.read_image("path/to/image.mha")
sitk_toolkit.write_image(image, "path/to/output.mha")

# Get metadata
spacing = sitk_toolkit.get_spacing(image)
size = sitk_toolkit.get_size(image)
origin = sitk_toolkit.get_origin(image)
```

### DICOM Toolkit

```python
from itkit.io import dcm_toolkit

# Read DICOM series
image = dcm_toolkit.read_dicom_series("path/to/dicom/folder")

# Convert and save
dcm_toolkit.write_image(image, "path/to/output.nii.gz")
```

### NIfTI Toolkit

```python
from itkit.io import nii_toolkit

# Read/write NIfTI files
image = nii_toolkit.read_nifti("path/to/image.nii.gz")
nii_toolkit.write_nifti(image, "path/to/output.nii.gz")
```

---

## Best Practices

1. **Choose the right framework**: 
   - OpenMMLab for established workflows (legacy support)
   - MONAI for comprehensive medical imaging features
   - TorchIO for data augmentation and preprocessing
   - PyTorch Lightning for modern, flexible training

2. **Use ITKIT preprocessing first**: Preprocess data with ITKIT commands before framework-specific operations

3. **Convert datasets appropriately**: Use `itk_convert` to prepare datasets for your chosen framework

4. **Leverage ITKIT datasets**: Use ITKIT's dataset classes for seamless integration

5. **Mix and match**: ITKIT tools can be used alongside any framework

## Next Steps

- [Models](models.md) - Explore available neural network implementations
- [API Reference](api_reference.md) - Detailed API documentation
