# itk_infer

Perform batch inference on 3D medical images using trained segmentation models with support for MMEngine and ONNX backends.

## Usage

```bash
itk_infer -i <input_folder> -o <output> --backend <backend> [options]
```

## Backends

- **mmengine**: Use MMEngine models with config and checkpoint files
- **onnx**: Use ONNX runtime for optimized inference

## Required Parameters

- `-i, --input-folder PATH`: Input folder containing image files (supports `*.mha`, `*.nii`, `*.nii.gz`)
- `-o, --output PATH`: Output folder for segmentation results

### Backend-Specific Requirements

**For MMEngine backend:**

- `-cfg, --cfg-path PATH`: Model configuration file path
- `-ckpt, --ckpt-path PATH`: Model checkpoint file path

**For ONNX backend:**

- `--onnx PATH`: ONNX model file path

## Optional Parameters

### Windowing Parameters

- `--wl INT`: Window level for CT preprocessing (optional; defaults to config value for MMEngine)
- `--ww INT`: Window width for CT preprocessing (optional; defaults to config value for MMEngine)

> **Note**: For ONNX backend, if `--wl/--ww` are not provided, the tool attempts to read them from the ONNX model's metadata (`window_level`/`window_width`).

### Inference Configuration

- `--patch-size Z Y X`: Override patch size for sliding window inference (three integers)
- `--patch-stride Z Y X`: Override patch stride for sliding window inference (three integers)

### Performance Options

- `--num-proc N`: Number of parallel processes (default: 1)
- `--gpus N`: Number of GPUs to use (default: 1)
- `--fp16`: Enable FP16 mixed precision for faster inference
- `--save-logits`: Save raw segmentation logits as `.zarr` files (compressed with LZ4)
- `--save-conf`: Calculate and save prediction confidence scores to `confidences.xlsx`

## Output Files

The tool generates the following outputs in the specified output folder:

1. **Segmentation Maps**: One file per input image with the same filename
   - Format: Same as input (`.mha`, `.nii`, or `.nii.gz`)
   - Orientation: Automatically reoriented to LPI
   - Metadata: Copied from input image (spacing, origin, direction)

2. **Logits (Optional)**: When `--save-logits` is enabled
   - Format: `.zarr` files with Blosc+LZ4 compression
   - Shape: `(C, Z, Y, X)` where C is the number of classes
   - Data type: float16

3. **Confidence Scores (Optional)**: When `--save-conf` is enabled
   - File: `confidences.xlsx`
   - Content: Per-image prediction confidence based on inverse entropy

## Examples

### MMEngine Backend

```bash
# Basic inference with MMEngine
itk_infer -i /data/images -o /data/results \
    --backend mmengine \
    -cfg /models/config.py \
    -ckpt /models/checkpoint.pth

# Multi-GPU inference with custom windowing
itk_infer -i /data/images -o /data/results \
    --backend mmengine \
    -cfg /models/config.py \
    -ckpt /models/checkpoint.pth \
    --wl 50 --ww 400 \
    --num-proc 4 --gpus 2

# FP16 inference with custom patch configuration
itk_infer -i /data/images -o /data/results \
    --backend mmengine \
    -cfg /models/config.py \
    -ckpt /models/checkpoint.pth \
    --patch-size 96 96 96 \
    --patch-stride 48 48 48 \
    --fp16
```

### ONNX Backend

```bash
# Basic ONNX inference
itk_infer -i /data/images -o /data/results \
    --backend onnx \
    --onnx /models/model.onnx \
    --wl 50 --ww 400

# Multi-process ONNX inference with logits and confidence
itk_infer -i /data/images -o /data/results \
    --backend onnx \
    --onnx /models/model.onnx \
    --num-proc 4 --gpus 2 \
    --save-logits --save-conf
```

## Features

### Automatic Skipping

The tool automatically skips files that have already been processed, checking for existing output files before inference. This enables resumable batch processing.

### Multi-Processing

Supports parallel processing across multiple GPUs:

- Files are evenly distributed across processes
- Each process is assigned to a GPU in round-robin fashion
- Progress bars show per-process status

### Prediction Confidence

When `--save-conf` is enabled, the tool calculates prediction confidence using inverse normalized entropy:

- **High confidence** (close to 1.0): Model is certain about predictions
- **Low confidence** (close to 0.0): Model is uncertain, predictions may be less reliable
- Useful for quality control and identifying cases requiring manual review

### Sliding Window Inference

Processes large 3D volumes by dividing them into overlapping patches:

- Configurable patch size and stride
- Automatic overlap blending
- Memory-efficient processing of arbitrarily large volumes

## Integration with 3D Slicer

For interactive inference within 3D Slicer, see the **[3D Slicer Integration](slicer_integration.md)** guide, which provides a GUI-based extension using the same inference backend.

## Performance Tips

1. **GPU Memory**: Use `--fp16` to reduce memory usage and increase speed
2. **Batch Processing**: Increase `--num-proc` to parallelize across multiple GPUs
3. **Patch Configuration**: Larger patches may improve accuracy but require more memory
4. **Windowing**: Proper `--wl/--ww` values are critical for CT image preprocessing

## Troubleshooting

**Error: "No input files found"**

- Ensure input folder contains files with supported extensions (`.mha`, `.nii`, `.nii.gz`)

**Error: "requires --wl/--ww"**

- For ONNX backend, specify windowing parameters or embed them in ONNX metadata

**Out of Memory**

- Reduce patch size using `--patch-size`
- Enable `--fp16` mode
- Reduce `--num-proc` if multiple processes compete for GPU memory

**Slow Performance**

- Enable `--fp16` for faster inference
- Increase `--num-proc` and `--gpus` for parallel processing
- Increase `--patch-stride` (less overlap means faster processing but potentially lower quality)
