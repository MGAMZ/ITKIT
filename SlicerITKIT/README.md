# ITKIT 3D Slicer Extension

This extension provides integration between 3D Slicer and ITKIT for deep learning-based medical image segmentation using a **client-server architecture**.

## Architecture Overview

```
┌─────────────────┐         HTTP/REST API        ┌──────────────────┐
│                 │ ◄──────────────────────────► │                  │
│  3D Slicer      │                               │  ITKIT Server    │
│  (Client Plugin)│   - Upload images             │  (Standalone)    │
│                 │   - Download results          │                  │
│  Lightweight    │   - Model management          │  Full ITKIT      │
│  Dependencies   │                               │  Environment     │
└─────────────────┘                               └──────────────────┘
  • SimpleITK                                       • PyTorch
  • requests                                        • ITKIT
                                                    • MMEngine/ONNX
```

**Key Benefits:**
- ✅ **No dependency conflicts**: Slicer doesn't need PyTorch or ITKIT installed
- ✅ **Clean separation**: Server runs in its own Python environment
- ✅ **Flexible deployment**: Server can run locally, remotely, or in Docker
- ✅ **Same workflow as MONAI Label**: Familiar architecture for users

## Quick Start

### 1. Start the ITKIT Server

```bash
cd SlicerITKIT/server

# Install server dependencies (one time)
pip install -r requirements.txt
pip install itkit[advanced]  # or itkit[onnx] for ONNX backend

# Start server
python itkit_server.py --host 0.0.0.0 --port 8000
```

Server will start at `http://localhost:8000`

### 2. Install the Slicer Plugin

1. Open 3D Slicer
2. Go to Edit → Application Settings → Modules
3. Add path to `SlicerITKIT/ITKITInference` directory
4. Restart 3D Slicer

### 3. Use the Plugin

1. Open module: Modules → Segmentation → ITKIT Inference
2. Enter server URL: `http://localhost:8000`
3. Click "Connect to Server"
4. Load a model on the server
5. Run inference on your volumes

## Installation

### Server Installation

The server requires ITKIT and its dependencies:

```bash
cd SlicerITKIT/server

# Option 1: Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install ITKIT with desired backend
pip install itkit[advanced]  # For MMEngine (PyTorch)
pip install itkit[onnx]      # For ONNX
pip install itkit[advanced,onnx]  # For both
```

### Client (Slicer Plugin) Installation

The plugin only requires basic dependencies that are usually available in Slicer:

**Required:**
- SimpleITK (usually pre-installed in Slicer)
- requests (install if needed: `pip install requests` in Slicer's Python console)

**Installation Steps:**
1. Clone or download this repository
2. Open 3D Slicer
3. Go to Edit → Application Settings → Modules
4. Click "Add" in "Additional module paths"
5. Navigate to and select: `SlicerITKIT/ITKITInference`
6. Click "OK" and restart Slicer

The module will appear under: Modules → Segmentation → ITKIT Inference

## Usage

### Starting the Server

```bash
# Basic (localhost only)
python itkit_server.py

# Allow external connections
python itkit_server.py --host 0.0.0.0 --port 8000

# Enable debug mode
python itkit_server.py --debug
```

### In 3D Slicer

#### 1. Connect to Server

1. Open ITKIT Inference module
2. Enter server URL (default: `http://localhost:8000`)
3. Click "Connect to Server"
4. Status should show "Connected"

#### 2. Load a Model

1. Enter a model name (e.g., "abdomen_seg")
2. Select backend type (MMEngine or ONNX)
3. Browse and select:
   - Config file (.py) - for MMEngine only
   - Model file (.pth or .onnx)
4. Optional: Configure inference parameters
   - Patch size (e.g., `96,96,96`)
   - Patch stride (e.g., `48,48,48`)
   - FP16 precision
5. Click "Load Model on Server"
6. Wait for model to load (status will update)

#### 3. Run Inference

1. Load your medical image volume in Slicer
2. Select the input volume
3. Choose the loaded model
4. Optional: Enable "Force CPU" if needed
5. Click "Run Inference"
6. Results will appear as a segmentation node

### Model Management

**Load Multiple Models:**
- Load different models for different tasks
- Switch between models for comparison
- Models are listed in the "Loaded Models" section

**Unload Models:**
- Select a model from the list
- Click "Unload Model" to free GPU/CPU memory

## Server API

The server exposes a REST API for programmatic access. See [server/README.md](server/README.md) for full API documentation.

**Main Endpoints:**
- `GET /api/health` - Health check
- `GET /api/info` - Server and models info
- `POST /api/models` - Load a model
- `POST /api/infer` - Run inference
- `DELETE /api/models/<name>` - Unload a model

## Configuration

### Server Configuration

```bash
python itkit_server.py --help

Options:
  --host HOST        Host address (default: 127.0.0.1)
  --port PORT        Port number (default: 8000)
  --debug            Enable debug mode
  --model-dir DIR    Auto-load models from directory
```

### Inference Configuration

When loading a model, you can specify:

```json
{
  "patch_size": [96, 96, 96],     // Sliding window size
  "patch_stride": [48, 48, 48],   // Overlap between windows
  "fp16": false,                   // Use half-precision
  "accumulate_device": "cuda",     // Device for accumulation
  "forward_device": "cuda"         // Device for inference
}
```

## Deployment Options

### Local Development

```bash
python itkit_server.py
```

### Docker (Recommended for Production)

```bash
cd SlicerITKIT/server
docker build -t itkit-server .
docker run -p 8000:8000 --gpus all itkit-server
```

### Remote Server

1. Start server on a machine with GPU
2. Use `--host 0.0.0.0` to accept external connections
3. In Slicer, enter the remote server URL (e.g., `http://192.168.1.100:8000`)

### Cloud Deployment

Deploy to cloud platforms (AWS, Azure, GCP) for scalable inference:
- Use containerized deployment
- Configure load balancing for multiple users
- Set up authentication and SSL/TLS

## Troubleshooting

### Cannot Connect to Server

**Problem**: "Connection failed" in Slicer

**Solutions:**
1. Verify server is running: `curl http://localhost:8000/api/health`
2. Check server URL is correct
3. If using remote server, check firewall settings
4. Ensure server is running with `--host 0.0.0.0` for external access

### Server Won't Start

**Problem**: Port already in use

**Solution:**
```bash
# Use a different port
python itkit_server.py --port 8001
```

### Model Loading Fails

**Problem**: Error loading model on server

**Solutions:**
1. Check file paths are correct and accessible to server
2. Verify ITKIT is installed: `pip list | grep itkit`
3. For MMEngine: ensure config file is valid Python
4. Check server logs for detailed error messages

### Inference Fails

**Problem**: "CUDA out of memory" or inference error

**Solutions:**
1. Reduce patch size in model configuration
2. Enable "Force CPU" in Slicer
3. Unload other models to free GPU memory
4. Check input image format is supported (NIfTI recommended)

### Requests Library Not Found

**Problem**: "requests library is not installed" in Slicer

**Solution:**
```python
# In Slicer's Python console:
import pip
pip.main(['install', 'requests'])
# Restart Slicer
```

## Comparison with Previous Version

| Feature | Old (Monolithic) | New (Client-Server) |
|---------|------------------|---------------------|
| Architecture | All-in-one | Client-Server |
| Slicer Dependencies | PyTorch, ITKIT, MMEngine | SimpleITK, requests |
| Installation | Complex | Simple |
| Environment | Shared with Slicer | Separate |
| Deployment | Local only | Local, remote, cloud |
| Conflicts | Yes | No |
| Similar to | - | MONAI Label |

## Advanced Usage

### Batch Processing

Use the server API directly for batch processing:

```python
import requests

server_url = "http://localhost:8000"

# Load model once
requests.post(f"{server_url}/api/models", json={
    "name": "batch_model",
    "backend_type": "onnx",
    "model_path": "/path/to/model.onnx"
})

# Process multiple images
for image_path in image_list:
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{server_url}/api/infer",
            files={"image": f},
            data={"model_name": "batch_model"}
        )
    # Save result
    with open(f"seg_{image_path}", 'wb') as f:
        f.write(response.content)
```

### Custom Server

Extend the server for custom workflows:

```python
# Add custom endpoint
@app.route('/api/custom', methods=['POST'])
def custom_endpoint():
    # Your custom logic
    return jsonify({'status': 'success'})
```

## Support

- **GitHub Issues**: https://github.com/MGAMZ/ITKIT/issues
- **Documentation**: https://itkit.readthedocs.io/
- **Email**: 312065559@qq.com
- **Server API Docs**: See [server/README.md](server/README.md)

## License

This extension is released under the MIT License, consistent with the ITKIT framework.

## Acknowledgments

- Architecture inspired by [MONAI Label](https://github.com/Project-MONAI/MONAILabel)
- Built on top of [ITKIT](https://github.com/MGAMZ/ITKIT)
- Uses [3D Slicer](https://www.slicer.org/) platform
