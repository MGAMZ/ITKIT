# ITKIT 3D Slicer Extension

This extension provides integration between 3D Slicer and ITKIT for deep learning-based medical image segmentation.

## Overview

The ITKIT Inference module allows users to run inference using trained ITKIT models directly within 3D Slicer. It supports both MMEngine and ONNX backends for flexibility and performance.

## Features

- **Multiple Backend Support**: Choose between MMEngine (for PyTorch models) and ONNX Runtime
- **Sliding Window Inference**: Handle large volumes with configurable patch-based inference
- **GPU Acceleration**: Leverage CUDA for fast inference
- **Configurable Parameters**: Fine-tune inference settings including patch size, stride, and precision
- **Seamless Integration**: Works directly with Slicer's volume and segmentation nodes

## Installation

### Prerequisites

1. Install 3D Slicer (version 5.0 or later recommended)
2. Install ITKIT with the required dependencies:

```bash
pip install itkit[advanced]
```

This will install ITKIT along with OpenMMLab dependencies (mmcv, mmengine, mmsegmentation).

For ONNX support:
```bash
pip install itkit[onnx]
```

### Installing the Extension

#### Option 1: From Source (Development)

1. Clone or download this repository
2. Open 3D Slicer
3. Go to Edit → Application Settings → Modules
4. Add the path to `SlicerITKIT/ITKITInference` directory
5. Restart 3D Slicer

#### Option 2: Extension Manager (Future)

Once the extension is registered in the Slicer Extension Index, it will be available through the Extension Manager.

## Usage

### Basic Workflow

1. **Load Your Data**
   - Load a 3D medical image volume in Slicer (e.g., CT, MRI)

2. **Open the Module**
   - Navigate to: Modules → Segmentation → ITKIT Inference

3. **Configure Backend**
   - Select backend type: MMEngine or ONNX
   - For MMEngine:
     - Browse and select the configuration file (.py)
     - Browse and select the checkpoint file (.pth)
   - For ONNX:
     - Browse and select the ONNX model file (.onnx)

4. **Set Input**
   - Select your input volume from the dropdown

5. **Configure Inference Parameters (Optional)**
   - **Patch Size**: For sliding window inference (e.g., `96,96,96`)
     - Leave empty for full volume inference
     - Use smaller patches for large volumes or limited GPU memory
   - **Patch Stride**: Overlap between patches (e.g., `48,48,48`)
     - Leave empty to use half of patch size
     - Smaller stride = more overlap = smoother results but slower
   - **Use FP16**: Enable half-precision for faster inference
   - **Force CPU Accumulation**: Use CPU for result accumulation if GPU memory is limited

6. **Run Inference**
   - Click "Run Inference" button
   - Monitor progress in the progress bar
   - Results will appear as a new segmentation node

### Example: Using a Pre-trained Model

```python
# In Slicer Python console:
import slicer

# Get the module widget
module = slicer.modules.itkitinference.widgetRepresentation()

# Load a test volume
import SampleData
volume = SampleData.downloadSample('CTChest')

# Configure for inference (manual setup via GUI recommended)
```

## Preparing Models for Use

### MMEngine Models

Your ITKIT model configuration should include:

```python
model = dict(
    type='YourModel',
    backbone=dict(...),
    # ... other model config
    inference_config=dict(
        patch_size=(96, 96, 96),  # Optional: default patch size
        patch_stride=(48, 48, 48),  # Optional: default stride
    )
)
```

### ONNX Models

Export your trained model to ONNX format:

```python
from itkit.mm.inference import MMEngineInferBackend
import torch

# Load your trained model
backend = MMEngineInferBackend(
    cfg_path='path/to/config.py',
    ckpt_path='path/to/checkpoint.pth'
)

# Export to ONNX
dummy_input = torch.randn(1, 1, 96, 96, 96).cuda()

# Note: The model component to export depends on your model architecture
# For models with a separate backbone: use backend.model.backbone
# For models without backbone separation: use backend.model directly
# Adjust based on your specific model structure
torch.onnx.export(
    backend.model.backbone,  # or backend.model for some architectures
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
```

## Troubleshooting

### Common Issues

1. **Import Error: ITKIT not found**
   - Solution: Install ITKIT in Slicer's Python environment:
     ```
     # In Slicer Python console
     import pip
     pip.main(['install', 'itkit[advanced]'])
     ```

2. **CUDA Out of Memory**
   - Solution: Enable "Force CPU Accumulation" or use smaller patch sizes

3. **Model Config Not Loading**
   - Solution: Ensure the config file path is correct and the file is valid Python

4. **Slow Inference**
   - Solution: 
     - Enable "Use FP16" for faster inference
     - Increase patch size if GPU memory allows
     - Use ONNX backend for optimized inference

### Getting Help

- GitHub Issues: https://github.com/MGAMZ/ITKIT/issues
- Documentation: https://itkit.readthedocs.io/
- Email: 312065559@qq.com

## Development

### Project Structure

```
SlicerITKIT/
├── CMakeLists.txt              # Extension build configuration
├── ITKITInference/
│   ├── CMakeLists.txt          # Module build configuration
│   ├── ITKITInference.py       # Main module implementation
│   └── Resources/
│       └── Icons/
│           └── ITKITInference.png  # Module icon
└── README.md                    # This file
```

### Testing

The module includes basic unit tests. To run them:

```python
# In Slicer Python console
import ITKITInference
tester = ITKITInference.ITKITInferenceTest()
tester.runTest()
```

## Citation

If you use this extension in your research, please cite:

```bibtex
@misc{ITKIT,
    author = {Yiqin Zhang},
    title = {ITKIT: Feasible Medical Image Operation based on SimpleITK API},
    year = {2025},
    url = {https://github.com/MGAMZ/ITKIT}
}
```

## License

This extension is released under the MIT License, consistent with the ITKIT framework.

## Acknowledgments

- Built on top of the ITKIT framework
- Uses 3D Slicer's ScriptedLoadableModule infrastructure
- Integrates with OpenMMLab and ONNX Runtime

## Version History

- **1.0.0** (2025-01-23): Initial release
  - MMEngine backend support
  - ONNX backend support
  - Sliding window inference
  - Configurable inference parameters
