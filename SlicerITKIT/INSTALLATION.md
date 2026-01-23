# ITKIT 3D Slicer Extension - Installation Guide

## Quick Start

### Step 1: Install 3D Slicer

Download and install 3D Slicer from: https://download.slicer.org/

Recommended version: 5.0 or later

### Step 2: Install ITKIT Python Package

Open 3D Slicer's Python console (View → Python Interactor) and run:

```python
import pip
pip.main(['install', 'itkit[advanced]'])
```

This will install:
- ITKIT core library
- OpenMMLab dependencies (mmcv, mmengine, mmsegmentation)
- Required deep learning libraries

For ONNX support, also run:
```python
pip.main(['install', 'itkit[onnx]'])
```

### Step 3: Install the Slicer Extension

#### Option A: Manual Installation (Development)

1. Clone or download the ITKIT repository
2. Open 3D Slicer
3. Go to: Edit → Application Settings → Modules
4. In the "Additional module paths" section, click "Add"
5. Navigate to and select: `ITKIT/SlicerITKIT/ITKITInference`
6. Click "OK" to save settings
7. Restart 3D Slicer

#### Option B: Using Extension Manager (Future)

Once registered in the Slicer Extension Index, you'll be able to install via:
1. Open Extension Manager (View → Extension Manager)
2. Search for "ITKIT"
3. Click "Install"
4. Restart Slicer

### Step 4: Verify Installation

1. Open 3D Slicer
2. Go to: Modules (dropdown at top) → Segmentation → ITKIT Inference
3. The module should load without errors

If you see "ERROR: ITKIT is not installed", revisit Step 2.

## Detailed Installation Instructions

### For Linux

```bash
# Install 3D Slicer
wget https://download.slicer.org/... # Get appropriate link
tar -xvf Slicer-*.tar.gz
cd Slicer-*
./Slicer

# In Slicer's Python console:
import pip
pip.main(['install', 'itkit[advanced]'])
```

### For macOS

1. Download Slicer DMG from https://download.slicer.org/
2. Drag to Applications folder
3. Open Slicer
4. In Python console, run installation commands from Step 2

### For Windows

1. Download Slicer installer from https://download.slicer.org/
2. Run installer and follow prompts
3. Open Slicer
4. In Python console, run installation commands from Step 2

## Dependencies

The extension requires the following Python packages:

### Core Dependencies (installed with `itkit[advanced]`)
- itkit >= 4.0.0
- torch
- onedl-mmcv
- onedl-mmengine
- onedl-mmsegmentation
- numpy
- SimpleITK

### Optional Dependencies
- onnxruntime (for ONNX backend)
- onnxruntime-gpu (for GPU-accelerated ONNX inference)

## GPU Support

### CUDA Setup

For GPU acceleration, ensure you have:
1. NVIDIA GPU with CUDA support
2. CUDA toolkit installed (version compatible with PyTorch)
3. PyTorch with CUDA support installed

To verify GPU availability in Slicer:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

If CUDA is not available, the extension will still work but will use CPU (slower).

## Troubleshooting

### Issue: "ITKIT is not installed"

**Solution:**
```python
# In Slicer Python console
import pip
pip.main(['install', '--upgrade', 'itkit[advanced]'])
# Restart Slicer
```

### Issue: "No module named 'mmcv'"

**Solution:**
```python
import pip
pip.main(['install', 'onedl-mmcv'])
```

### Issue: CUDA/GPU not working

**Solution 1:** Verify PyTorch CUDA installation
```python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

**Solution 2:** Reinstall PyTorch with CUDA
```python
import pip
# For CUDA 11.8
pip.main(['install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu118'])
```

### Issue: "Module not found in module list"

**Solution:**
1. Check the module path is correctly added in Application Settings
2. Ensure the path points to the `ITKITInference` directory (not parent)
3. Restart Slicer after adding the path

### Issue: Import errors when running inference

**Solution:**
Check that all dependencies are installed:
```python
# In Slicer Python console
try:
    import itkit
    print(f"ITKIT version: {itkit.__version__}")
except ImportError as e:
    print(f"ITKIT error: {e}")

try:
    import mmengine
    print("MMEngine: OK")
except ImportError:
    print("MMEngine: NOT INSTALLED")

try:
    import mmseg
    print("MMSegmentation: OK")
except ImportError:
    print("MMSegmentation: NOT INSTALLED")
```

### Issue: Out of memory errors

**Solutions:**
1. Enable "Force CPU Accumulation" option
2. Reduce patch size (e.g., from 128,128,128 to 96,96,96)
3. Close other applications to free GPU/system memory
4. Use FP16 precision (faster and less memory)

## Uninstallation

### Remove Extension
1. Go to: Edit → Application Settings → Modules
2. Remove the ITKIT extension path
3. Restart Slicer

### Uninstall Python Package
```python
# In Slicer Python console
import pip
pip.main(['uninstall', 'itkit', '-y'])
```

## Getting Help

- Documentation: https://itkit.readthedocs.io/
- GitHub Issues: https://github.com/MGAMZ/ITKIT/issues
- Email: 312065559@qq.com

## Next Steps

After installation, see the main README.md for usage instructions and examples.
