# 3D Slicer Integration

ITKIT provides a native 3D Slicer extension for running inference directly within the Slicer environment. This integration allows radiologists and researchers to seamlessly apply trained segmentation models to medical images without leaving the Slicer interface.

## Installation

See the [Installation Guide](../SlicerITKIT/INSTALLATION.md) for detailed instructions.

Quick summary:

1. Install 3D Slicer (5.0+)
2. Install ITKIT: `pip install itkit[advanced]`
3. Add the extension path to Slicer
4. Restart Slicer

## Usage

1. **Open ITKIT Inference Module**
   - Navigate to: Modules → Segmentation → ITKIT Inference

2. **Configure Backend**
   - Select backend type (MMEngine or ONNX)
   - Browse for configuration file (MMEngine only)
   - Browse for checkpoint/model file

3. **Set Parameters** (optional)
   - Patch size for sliding window (e.g., `96,96,96`)
   - Patch stride (e.g., `48,48,48`)
   - Enable FP16 for faster inference
   - Force CPU accumulation if needed

4. **Run Inference**
   - Click "Run Inference"
   - Monitor progress
   - View results in segmentation node
