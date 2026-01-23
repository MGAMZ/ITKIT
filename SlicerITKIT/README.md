# ITKIT 3D Slicer Plugin

Deep learning medical image segmentation plugin for 3D Slicer using a **client-server architecture**.

## Architecture

```
┌─────────────────┐      HTTP REST API       ┌──────────────────┐
│  3D Slicer      │  ◄─────────────────────► │  ITKIT Server    │
│  (Client)       │                          │  (Standalone)    │
│                 │  • Upload images         │                  │
│  Dependencies:  │  • Download results      │  Dependencies:   │
│  - SimpleITK    │  • Model management      │  - PyTorch       │
│  - requests     │                          │  - ITKIT         │
└─────────────────┘                          └──────────────────┘
```

**Benefits:**

- No dependency conflicts (Slicer doesn't need PyTorch)
- Server runs in separate Python environment
- Flexible deployment (local/remote/cloud/Docker)
- Similar to MONAI Label architecture

## Installation

### 1. Server Installation

```bash
cd SlicerITKIT/server

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install ITKIT
pip install itkit[advanced,onnx]  # Both
```

### 2. Slicer Plugin Installation

**Requirements:**

- 3D Slicer 5.0+
- SimpleITK (pre-installed in Slicer)
- requests library

**Install requests if needed:**

```python
# In Slicer Python console (View → Python Interactor):
import pip
pip.main(['install', 'requests'])
```

**Add module to Slicer:**

1. Edit → Application Settings → Modules
2. Click "Add" in "Additional module paths"
3. Select: `ITKIT/SlicerITKIT/ITKITInference`
4. Click OK and restart Slicer

Module will appear: Modules → Segmentation → ITKIT

## Quick Start

### 1. Start Server

```bash
# Basic (localhost)
python SlicerITKIT/server/itkit_server.py

# Allow external connections
python SlicerITKIT/server/itkit_server.py --host 0.0.0.0 --port 8000

# Enable debug logging
python SlicerITKIT/server/itkit_server.py --debug
```

Server starts at `http://localhost:8000`

### 2. Connect from Slicer

1. Open ITKIT module
2. Enter server URL: `http://localhost:8000`
3. Click "Connect to Server"
4. Status shows "Connected" with server info

### 3. Load Model

**For MMEngine (PyTorch):**

1. Backend: MMEngine
2. Config File: Select `.py` config
3. Model File: Select `.pth` checkpoint
4. Optional: Set FP16

**For ONNX:**

1. Backend: ONNX
2. Model File: Select `.onnx` file

Click "Load Model" (replaces any previously loaded model)

### 4. Run Inference

1. Input Volume: Select loaded volume
2. Optional: Set Patch Size/Stride (e.g., `96,96,96` / `48,48,48`)
3. Optional: Override Window Level/Width
4. Optional: Force CPU
5. Click "Run Inference"

Results appear as segmentation node.

## API Reference

### Server Endpoints

**GET /api/info**

- Returns server status and current model info

**POST /api/model**

- Load model (auto-unloads previous)
- Body: `{backend_type, config_path?, model_path, inference_config}`

**DELETE /api/model**

- Unload current model

**POST /api/infer**

- Run inference
- Form data: `image` (file), `force_cpu`, `window_level?`, `window_width?`
- Returns: segmentation as NIfTI

### Programmatic Usage

```python
import requests

server = "http://localhost:8000"

# Load model
requests.post(f"{server}/api/model", json={
    "backend_type": "onnx",
    "model_path": "/path/to/model.onnx",
    "inference_config": {"fp16": False}
})

# Run inference
with open("image.nii.gz", "rb") as f:
    response = requests.post(
        f"{server}/api/infer",
        files={"image": f},
        data={"force_cpu": "false"}
    )

with open("segmentation.nii.gz", "wb") as f:
    f.write(response.content)
```

## Configuration

### Inference Parameters

**In Load Model:**

- `fp16`: Use half precision (faster, less memory)

**In Run Inference:**

- `patch_size`: Sliding window size (e.g., `[96,96,96]`)
- `patch_stride`: Overlap between windows (e.g., `[48,48,48]`)
- `window_level`: HU window center for preprocessing
- `window_width`: HU window width for preprocessing
- `force_cpu`: Force CPU accumulation (for GPU OOM)

### Server Configuration

```bash
python itkit_server.py --help

Options:
  --host HOST    Host address (default: 127.0.0.1)
  --port PORT    Port number (default: 8000)
  --debug        Enable DEBUG logging
```

## Deployment

### Local Development

```bash
python itkit_server.py
```

### Docker

```bash
cd SlicerITKIT/server
docker build -t itkit-server -f ../../CI-CD/Dockerfile.itkit .
docker run -p 8000:8000 --gpus all itkit-server
```

### Remote Server

```bash
# On server with GPU
python itkit_server.py --host 0.0.0.0 --port 8000

# In Slicer, connect to:
# http://<server-ip>:8000
```

### Cloud (AWS/Azure/GCP)

- Deploy containerized server
- Configure load balancing
- Add authentication/SSL

## Troubleshooting

### Connection Failed

**Check server is running:**

```bash
curl http://localhost:8000/api/health
```

**For remote server:**

- Use `--host 0.0.0.0` when starting server
- Check firewall allows port 8000
- Verify URL includes `http://` prefix

### Model Loading Fails

**Check paths:**

- Paths must be accessible from server's filesystem
- For Windows client → Linux server: use Linux paths
- Verify files exist on server

**Check dependencies:**

```bash
pip list | grep itkit
pip list | grep torch
pip list | grep onnxruntime
```

### Inference Fails

**CUDA out of memory:**

- Reduce patch size (e.g., `96,96,96` → `64,64,64`)
- Enable "Force CPU"
- Unload model and reload with smaller config

**Missing windowing parameters:**

- Server extracts from model metadata if available
- Manually override Window Level/Width in UI
- Add to model config or ONNX metadata

### requests Not Found in Slicer

```python
# In Slicer Python console:
import pip
pip.main(['install', 'requests'])
# Restart Slicer
```

## Architecture Details

### Client-Server Separation

**Why this architecture?**

1. **Avoid dependency hell**: PyTorch + Qt + Slicer = conflicts
2. **Flexible deployment**: Server on GPU machine, Slicer on workstation
3. **Independent updates**: Update ITKIT without rebuilding Slicer
4. **Standard pattern**: Same as MONAI Label, familiar to users

### Implementation Details

**Client (ITKITInference.py):**

- Qt-based UI in Slicer
- REST API calls via `requests`
- Minimal dependencies (SimpleITK, requests)
- Async inference to prevent UI freeze

**Server (itkit_server.py):**

- Flask REST API
- Single-model design (load replaces previous)
- Automatic LPI orientation alignment
- Window level/width preprocessing
- Temp file cleanup via `response.call_on_close()`

**Key Design Decisions:**

1. Single model: Simplified UX, matches typical workflow
2. Server-side preprocessing: Consistent results, reduce client complexity
3. Async client: Prevent 30% progress freeze
4. Direct ITKIT integration: No custom wrappers, use existing inferencer

### Data Flow

```
User loads volume in Slicer
    ↓
Volume saved to temp NIfTI
    ↓
POST /api/infer with file upload
    ↓
Server: Load image → LPI orientation → Windowing → Inference
    ↓
Server: Return segmentation as NIfTI
    ↓
Client: Load result → Convert to segmentation node
    ↓
Display in Slicer
```

## Advanced Usage

### Batch Processing Script

```python
import requests
import glob

server = "http://localhost:8000"

# Load model once
requests.post(f"{server}/api/model", json={
    "backend_type": "mmengine",
    "config_path": "/models/config.py",
    "model_path": "/models/checkpoint.pth"
})

# Process all images
for img_path in glob.glob("data/*.nii.gz"):
    with open(img_path, "rb") as f:
        response = requests.post(
            f"{server}/api/infer",
            files={"image": f}
        )

    out_path = img_path.replace(".nii.gz", "_seg.nii.gz")
    with open(out_path, "wb") as f:
        f.write(response.content)
    print(f"Processed: {img_path}")
```

### Custom Preprocessing

Modify server to add custom preprocessing:

```python
# In itkit_server.py, before inference:
def custom_preprocess(image_array):
    # Your preprocessing
    return processed_array

# Add to run_inference endpoint
image_array = custom_preprocess(image_array)
```

### Model Metadata

Add windowing to ONNX metadata:

```python
import onnx
model = onnx.load("model.onnx")
model.metadata_props.add(key="window_level", value="50")
model.metadata_props.add(key="window_width", value="350")
onnx.save(model, "model_with_metadata.onnx")
```

For MMEngine, add to config:

```python
# config.py
wl = 50
ww = 350
# or
model = dict(
    wl=50,
    ww=350
)
```

## Support

- **Issues**: <https://github.com/MGAMZ/ITKIT/issues>
- **Documentation**: <https://itkit.readthedocs.io/>
- **Email**: <312065559@qq.com>

## License

MIT License (same as ITKIT)

## Acknowledgments

- Architecture inspired by [MONAI Label](https://github.com/Project-MONAI/MONAILabel)
- Built on [ITKIT](https://github.com/MGAMZ/ITKIT) framework
- Uses [3D Slicer](https://www.slicer.org/) platform
