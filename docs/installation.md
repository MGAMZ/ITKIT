# Installation Guide

## Requirements

ITKIT is tested on:

- **Python** >= 3.10
- **numpy** >= 2.2.6
- **SimpleITK** >= 2.5.0

Lower versions may work but are not guaranteed.

## Installation Methods

### From PyPI (Recommended)

The simplest way to install ITKIT is via pip:

```bash
pip install itkit
```

### From Source

First, clone the repository:

```bash
git clone https://github.com/MGAMZ/ITKIT.git
cd ITKIT
```

Then, install the package:

```bash
pip install .
```

## Optional Dependencies

ITKIT provides several optional dependency groups for different use cases:

### Development Tools

For development and testing:

```bash
pip install "itkit[dev]"
```

Includes: pytest, pylint, black, isort, mypy, and other development tools.

### Advanced Features

For advanced image processing and deep learning features:

```bash
pip install "itkit[advanced]"
```

Includes: torchio, onedl-mmcv, onedl-mmengine, onedl-mmsegmentation.

### Pathology Support

For pathology image processing features:

```bash
pip install "itkit[pathology]"
```

Includes: opensdpc, openslide-python, openslide-bin.

### GUI Support

For graphical user interface:

```bash
pip install "itkit[gui]"
```

Includes: PyQt6.

### ONNX Deployment

For model deployment with ONNX:

```bash
pip install "itkit[onnx]"
```

Includes: onnx, onnxruntime, tensorrt, and related tools.

### Combined Installation

You can install multiple optional dependencies at once:

```bash
pip install "itkit[dev,advanced,gui]"
```

## Deep Learning Framework Compatibility

If you plan to run deep learning tasks, we recommend installing `monai` to avoid potential dependency issues:

```bash
pip install --no-deps monai
```

**Note:** The `itk_convert monai` and `itk_convert torchio` commands do not require the `monai` or `torchio` Python packages to perform the conversion. Install these packages only if you plan to run MONAI or TorchIO-based deep learning workflows.

## OpenMMLab Integration

**Important:** The upstream `OpenMMLab` project has gradually fallen out of maintenance. ITKIT now recommends users to use the `OneDL` redistribution of `OpenMMLab` instead:

- OneDL-mmengine
- OneDL-mmcv
- OneDL-mmsegmentation

These are included in the `advanced` optional dependencies.

## Verifying Installation

After installation, verify that ITKIT is properly installed:

```bash
# Check version
python -c "import itkit; print(itkit.__version__)"

# List available commands
itk_check --help
itk_resample --help
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install itkit --force-reinstall
```

### GUI DPI Issues

If the GUI's DPI is not optimal, specify the `QT_SCALE_FACTOR` environment variable:

```bash
QT_SCALE_FACTOR=2 itkit-app
```

### PyTorch Compatibility

For GPU support with PyTorch-based features, install PyTorch separately following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
