# ITKIT 3D Slicer Extension - Quick Reference

## File Structure

```
SlicerITKIT/
├── CMakeLists.txt                      # Extension build configuration
├── README.md                            # Extension overview and features
├── INSTALLATION.md                      # Detailed installation guide
└── ITKITInference/
    ├── CMakeLists.txt                  # Module build configuration
    ├── ITKITInference.py               # Main module implementation
    ├── example_usage.py                # Example scripts
    └── Resources/
        └── Icons/
            └── ITKITInference.png      # Module icon (128x128 PNG)
```

## Quick Commands

### Installation in Slicer Python
```python
import pip
pip.main(['install', 'itkit[advanced]'])
# Restart Slicer after installation
```

### Load Module Programmatically
```python
import slicer

# Get module widget
widget = slicer.modules.itkitinference.widgetRepresentation().self()
logic = widget.logic
```

### Run Inference (MMEngine)
```python
segmentation = logic.process(
    inputVolume=volumeNode,
    outputSegmentation=None,
    backend="MMEngine",
    configPath="/path/to/config.py",
    checkpointPath="/path/to/checkpoint.pth",
    patchSize=(96, 96, 96),
    patchStride=(48, 48, 48),
    fp16=True,
    forceCpu=False
)
```

### Run Inference (ONNX)
```python
segmentation = logic.process(
    inputVolume=volumeNode,
    outputSegmentation=None,
    backend="ONNX",
    configPath=None,
    checkpointPath="/path/to/model.onnx",
    patchSize=None,  # Use model defaults
    patchStride=None,
    fp16=False,
    forceCpu=False
)
```

## Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `inputVolume` | vtkMRMLScalarVolumeNode | Input 3D volume | Required |
| `outputSegmentation` | vtkMRMLSegmentationNode or None | Output node | None (creates new) |
| `backend` | str | "MMEngine" or "ONNX" | Required |
| `configPath` | str or None | Config file path (MMEngine only) | None |
| `checkpointPath` | str | Model/checkpoint path | Required |
| `patchSize` | tuple or None | Sliding window size (Z,Y,X) | None (full volume) |
| `patchStride` | tuple or None | Sliding window stride (Z,Y,X) | None (half of patch) |
| `fp16` | bool | Use half precision | False |
| `forceCpu` | bool | Force CPU accumulation | False |

## Typical Patch Sizes

| Volume Size | GPU Memory | Recommended Patch Size | Stride |
|-------------|-----------|------------------------|--------|
| Small (< 256³) | Any | None (full volume) | N/A |
| Medium (256-512³) | 8GB | (128, 128, 128) | (64, 64, 64) |
| Medium (256-512³) | 12GB+ | (160, 160, 160) | (80, 80, 80) |
| Large (> 512³) | 8GB | (96, 96, 96) | (48, 48, 48) |
| Large (> 512³) | 12GB+ | (128, 128, 128) | (64, 64, 64) |

## Performance Tips

1. **Enable FP16**: ~2x faster, ~50% less memory
2. **Larger patches**: Better context, fewer computations
3. **More stride**: Faster but less smooth results
4. **ONNX backend**: Optimized for deployment
5. **Force CPU**: Only if GPU OOM occurs

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Module not found | Check Application Settings → Modules → Additional paths |
| Import error | `pip.main(['install', 'itkit[advanced]'])` in Slicer Python |
| CUDA OOM | Enable "Force CPU Accumulation" or reduce patch size |
| Slow inference | Enable FP16, use ONNX, or increase patch size |
| Blocky results | Reduce stride (more overlap) |

## Example Workflows

### Basic Workflow
1. Load volume: `File → Add Data` or `SampleData.downloadSample('CTChest')`
2. Open module: `Modules → Segmentation → ITKIT Inference`
3. Select backend and model files
4. Configure parameters (optional)
5. Click "Run Inference"
6. View results in Segment Editor

### Batch Processing
```python
for file in volume_files:
    vol = slicer.util.loadVolume(file)
    seg = logic.process(vol, None, "ONNX", None, model_path, None, None)
    slicer.util.saveNode(seg, output_path)
    slicer.mrmlScene.RemoveNode(vol)
    slicer.mrmlScene.RemoveNode(seg)
```

### Custom Progress Callback
```python
def my_progress(percent, message):
    print(f"{percent}%: {message}")

seg = logic.process(
    ...,
    progressCallback=my_progress
)
```

## Backend Comparison

| Feature | MMEngine | ONNX |
|---------|----------|------|
| Flexibility | High (full PyTorch) | Medium (static graph) |
| Speed | Good | Excellent |
| Memory | Higher | Lower |
| Setup | Config + checkpoint | Single file |
| Use Case | Development, research | Deployment, production |

## Module Integration Points

### Input
- Accepts any `vtkMRMLScalarVolumeNode`
- Extracts numpy array (Z, Y, X)
- Preserves spatial information

### Processing
- Uses `itkit.mm.inference` backends
- Sliding window inference for large volumes
- GPU acceleration with CUDA

### Output
- Creates `vtkMRMLSegmentationNode`
- Preserves geometry from input volume
- Compatible with Segment Editor

## API Methods

### ITKITInferenceWidget
- `setup()`: Initialize GUI
- `onApplyButton()`: Handle inference trigger
- `updateProgress(int, str)`: Update UI progress

### ITKITInferenceLogic
- `process(...)`: Main inference method (see parameters above)

### ITKITInferenceTest
- `runTest()`: Execute unit tests

## Extension Metadata

- **Category**: Segmentation
- **Dependencies**: ITKIT, PyTorch, MMEngine (for MMEngine backend) or ONNX Runtime (for ONNX backend)
- **Platforms**: Linux, macOS, Windows (with CUDA support optional)
- **Slicer Version**: 5.0+

## Resources

- Full documentation: See `README.md` and `INSTALLATION.md`
- Example scripts: `example_usage.py`
- ITKIT documentation: https://itkit.readthedocs.io/
- 3D Slicer documentation: https://slicer.readthedocs.io/

## Support

- GitHub Issues: https://github.com/MGAMZ/ITKIT/issues
- Email: 312065559@qq.com
