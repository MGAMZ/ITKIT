# 3D Slicer Integration

ITKIT provides a native 3D Slicer extension for running inference directly within the Slicer environment. This integration allows radiologists and researchers to seamlessly apply trained segmentation models to medical images without leaving the Slicer interface.

## Overview

The ITKIT Slicer extension (SlicerITKIT) provides a user-friendly interface for:
- Loading and configuring deep learning models
- Running inference on 3D medical volumes
- Generating segmentation results as Slicer segmentation nodes
- Configuring advanced inference parameters (sliding window, precision, etc.)

## Features

### Multiple Backend Support
- **MMEngine Backend**: Full PyTorch model support with configuration files
- **ONNX Backend**: Optimized inference with ONNX Runtime

### Sliding Window Inference
- Handle large volumes that don't fit in GPU memory
- Configurable patch size and stride for optimal results
- Automatic padding and cropping

### GPU Acceleration
- CUDA support for fast inference
- Mixed precision (FP16) support
- CPU fallback for systems without GPU

### Seamless Integration
- Works directly with Slicer volume nodes
- Outputs standard Slicer segmentation nodes
- Compatible with Slicer's segment editor and visualization tools

## Installation

See the [Installation Guide](../SlicerITKIT/INSTALLATION.md) for detailed instructions.

Quick summary:
1. Install 3D Slicer (5.0+)
2. Install ITKIT: `pip install itkit[advanced]`
3. Add the extension path to Slicer
4. Restart Slicer

## Usage

### Basic Workflow

1. **Load Your Data**
   ```python
   # Load a volume in Slicer (via GUI or Python)
   import slicer
   volumeNode = slicer.util.loadVolume('/path/to/your/image.nii.gz')
   ```

2. **Open ITKIT Inference Module**
   - Navigate to: Modules → Segmentation → ITKIT Inference

3. **Configure Backend**
   - Select backend type (MMEngine or ONNX)
   - Browse for configuration file (MMEngine only)
   - Browse for checkpoint/model file

4. **Set Parameters** (optional)
   - Patch size for sliding window (e.g., `96,96,96`)
   - Patch stride (e.g., `48,48,48`)
   - Enable FP16 for faster inference
   - Force CPU accumulation if needed

5. **Run Inference**
   - Click "Run Inference"
   - Monitor progress
   - View results in segmentation node

### Example: Abdominal CT Segmentation

```python
# Example using Python scripting in Slicer
import slicer

# Load CT scan
volumeNode = slicer.util.loadVolume('/path/to/abdomen_ct.nii.gz')

# Get the ITKIT Inference logic
logic = slicer.modules.itkitinference.widgetRepresentation().self().logic

# Run inference
segmentation = logic.process(
    inputVolume=volumeNode,
    outputSegmentation=None,
    backend="MMEngine",
    configPath="/path/to/config.py",
    checkpointPath="/path/to/checkpoint.pth",
    patchSize=(128, 128, 128),
    patchStride=(64, 64, 64),
    fp16=True,
    forceCpu=False
)

# Segmentation is now available for visualization and editing
```

## Model Configuration

### MMEngine Models

Your model configuration should be compatible with ITKIT's MMEngine integration:

```python
# config.py
model = dict(
    type='YourSegmentationModel',
    backbone=dict(
        type='YourBackbone',
        in_channels=1,
        # ... backbone config
    ),
    decode_head=dict(
        type='YourDecodeHead',
        num_classes=5,  # Number of segmentation classes
        # ... head config
    ),
    # Optional: default inference configuration
    inference_config=dict(
        patch_size=(96, 96, 96),
        patch_stride=(48, 48, 48),
        accumulate_device='cuda',
        forward_device='cuda',
    )
)
```

### ONNX Models

Export your trained model to ONNX format:

```python
import torch
from itkit.mm.inference import MMEngineInferBackend

# Load your trained model
backend = MMEngineInferBackend(
    cfg_path='path/to/config.py',
    ckpt_path='path/to/checkpoint.pth'
)

# Create dummy input matching your model's expected input shape
dummy_input = torch.randn(1, 1, 96, 96, 96).cuda()

# Export to ONNX
# Important: The model component to export depends on your specific architecture
# Common patterns:
# 1. Models with separate backbone: backend.model.backbone (e.g., SegFormer)
# 2. Unified models without backbone: backend.model (e.g., simple U-Net variants)
# 3. Models with encode-decode structure: backend.model.encode_decode
# Check your model's architecture to determine the correct component to export
torch.onnx.export(
    backend.model.backbone,  # Adjust based on your model architecture
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'},
        'output': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'}
    },
    opset_version=11
)
```

## Advanced Usage

### Scripting

You can automate inference for batch processing:

```python
import os
import slicer

# Get logic
logic = slicer.modules.itkitinference.widgetRepresentation().self().logic

# Process multiple volumes
input_dir = '/path/to/input/volumes'
output_dir = '/path/to/output/segmentations'

for filename in os.listdir(input_dir):
    if filename.endswith('.nii.gz'):
        # Load volume
        filepath = os.path.join(input_dir, filename)
        volumeNode = slicer.util.loadVolume(filepath)
        
        # Run inference
        segmentation = logic.process(
            inputVolume=volumeNode,
            outputSegmentation=None,
            backend="ONNX",
            configPath=None,
            checkpointPath="/path/to/model.onnx",
            patchSize=None,  # Use full volume
            patchStride=None,
            fp16=False,
            forceCpu=False
        )
        
        # Save segmentation
        output_path = os.path.join(output_dir, filename.replace('.nii.gz', '_seg.nii.gz'))
        slicer.util.saveNode(segmentation, output_path)
        
        # Clean up
        slicer.mrmlScene.RemoveNode(volumeNode)
        slicer.mrmlScene.RemoveNode(segmentation)

print("Batch processing complete!")
```

### Custom Inference Config

Override inference configuration at runtime:

```python
# Custom configuration for large volumes
custom_config = {
    'patch_size': (64, 64, 64),  # Smaller patches for memory constraints
    'patch_stride': (32, 32, 32),  # More overlap for smoother results
    'accumulate_device': 'cpu',  # Use CPU for accumulation
    'forward_device': 'cuda',  # Use GPU for inference
    'forward_batch_windows': 4  # Process 4 patches at once
}

# Pass to backend initialization (requires custom logic implementation)
```

## Performance Optimization

### GPU Memory Management

1. **Use FP16 Precision**
   - Enable "Use FP16" in the GUI
   - Reduces memory usage by ~50%
   - Minimal impact on accuracy

2. **Adjust Patch Size**
   - Smaller patches = less GPU memory
   - Typical sizes: 64³, 96³, 128³
   - Balance between memory and context

3. **Enable CPU Accumulation**
   - Use "Force CPU Accumulation" for very large volumes
   - Accumulate results on CPU to free GPU memory
   - Slight performance trade-off

### Inference Speed

1. **Batch Processing**
   - Process multiple patches simultaneously
   - Adjust `forward_batch_windows` in config

2. **ONNX Runtime**
   - Use ONNX backend for optimized inference
   - Especially beneficial for deployment

3. **Optimal Stride**
   - Larger stride = faster inference
   - Smaller stride = smoother results
   - Recommended: stride = patch_size / 2

## Troubleshooting

### Common Issues

**Problem**: Module doesn't appear in Slicer
- **Solution**: Check that the extension path is correctly added in Application Settings → Modules

**Problem**: "ITKIT is not installed" error
- **Solution**: Install ITKIT in Slicer's Python:
  ```python
  import pip
  pip.main(['install', 'itkit[advanced]'])
  ```

**Problem**: CUDA out of memory
- **Solution**: 
  - Enable "Force CPU Accumulation"
  - Reduce patch size
  - Use FP16 precision

**Problem**: Slow inference
- **Solution**:
  - Enable FP16
  - Use ONNX backend
  - Increase patch size if memory allows
  - Reduce overlap (larger stride)

**Problem**: Segmentation results look blocky
- **Solution**:
  - Reduce stride (more overlap)
  - Use Gaussian blending (configure in inference_config)

## Integration with Slicer Workflows

### Segment Editor Integration

After inference, use Slicer's Segment Editor to:
- Refine segmentation results
- Add/remove segments
- Apply smoothing and morphological operations
- Export to various formats

### Visualization

- Use 3D view for volume rendering
- Adjust segment colors and opacity
- Create animations and screenshots
- Export to presentation formats

### Quantification

Use Slicer's analysis modules:
- Segment Statistics for volume/surface measurements
- Label Statistics for intensity analysis
- Distance metrics between structures

## API Reference

### ITKITInferenceLogic.process()

Main inference method:

```python
def process(self,
            inputVolume: vtkMRMLScalarVolumeNode,
            outputSegmentation: vtkMRMLSegmentationNode | None,
            backend: str,
            configPath: str | None,
            checkpointPath: str,
            patchSize: tuple | None,
            patchStride: tuple | None,
            fp16: bool = False,
            forceCpu: bool = False,
            progressCallback: Callable | None = None
           ) -> vtkMRMLSegmentationNode
```

**Parameters:**
- `inputVolume`: Input 3D volume node
- `outputSegmentation`: Existing segmentation node or None to create new
- `backend`: "MMEngine" or "ONNX"
- `configPath`: Path to config file (MMEngine only)
- `checkpointPath`: Path to model checkpoint/ONNX file
- `patchSize`: Optional sliding window size as tuple (Z, Y, X)
- `patchStride`: Optional stride as tuple (Z, Y, X)
- `fp16`: Use half precision
- `forceCpu`: Force CPU accumulation
- `progressCallback`: Optional callback for progress updates

**Returns:**
- Output segmentation node with results

## Support

- GitHub: https://github.com/MGAMZ/ITKIT
- Issues: https://github.com/MGAMZ/ITKIT/issues
- Documentation: https://itkit.readthedocs.io/
- Email: 312065559@qq.com

## Contributing

Contributions to improve the Slicer extension are welcome! Please see [CONTRIBUTING.md](contributing.md) for guidelines.

## License

The ITKIT Slicer extension is released under the MIT License, consistent with both ITKIT and 3D Slicer.
