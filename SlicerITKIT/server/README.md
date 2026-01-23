# ITKIT Inference Server

A lightweight REST API server for running ITKIT model inference. This server runs in its own Python environment, completely independent from 3D Slicer's environment.

## Features

- **Separate Environment**: Runs independently with all ITKIT dependencies
- **REST API**: Simple HTTP endpoints for inference
- **Multiple Backends**: Supports both MMEngine (PyTorch) and ONNX
- **Multiple Models**: Load and manage multiple models simultaneously
- **CUDA Support**: Automatic GPU detection and utilization
- **Memory Management**: Unload models to free GPU/CPU memory

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
cd SlicerITKIT/server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install server dependencies
pip install -r requirements.txt

# Install ITKIT with backends
# For MMEngine (PyTorch) backend:
pip install itkit[advanced]

# For ONNX backend:
pip install itkit[onnx]

# Or install both:
pip install itkit[advanced,onnx]
```

## Usage

### Starting the Server

```bash
# Basic usage (localhost only)
python itkit_server.py

# Allow external connections
python itkit_server.py --host 0.0.0.0 --port 8000

# Enable debug mode
python itkit_server.py --debug
```

The server will start and display:
```
============================================================
ITKIT Inference Server
============================================================
Host: 127.0.0.1
Port: 8000
CUDA Available: True
CUDA Devices: 1
============================================================
```

## API Endpoints

### 1. Health Check

```bash
GET /api/health
```

Response:
```json
{
  "status": "ok",
  "cuda_available": true,
  "cuda_device_count": 1
}
```

### 2. Server Info

```bash
GET /api/info
```

Response:
```json
{
  "name": "ITKIT Inference Server",
  "version": "1.0.0",
  "models": {
    "my_model": {
      "name": "my_model",
      "backend_type": "mmengine",
      "config_path": "/path/to/config.py",
      "model_path": "/path/to/checkpoint.pth",
      "loaded": true
    }
  },
  "cuda_available": true,
  "device_count": 1
}
```

### 3. Load Model

```bash
POST /api/models
Content-Type: application/json

{
  "name": "my_model",
  "backend_type": "mmengine",
  "config_path": "/path/to/config.py",
  "model_path": "/path/to/checkpoint.pth",
  "inference_config": {
    "patch_size": [96, 96, 96],
    "patch_stride": [48, 48, 48],
    "fp16": false
  }
}
```

For ONNX models:
```json
{
  "name": "my_onnx_model",
  "backend_type": "onnx",
  "model_path": "/path/to/model.onnx",
  "inference_config": {
    "patch_stride": [48, 48, 48]
  }
}
```

Response:
```json
{
  "status": "success",
  "model": {
    "name": "my_model",
    "backend_type": "mmengine",
    "loaded": true
  }
}
```

### 4. Run Inference

```bash
POST /api/infer
Content-Type: multipart/form-data

Form fields:
- image: (file) NIfTI image file
- model_name: (string) Name of the loaded model
- force_cpu: (string, optional) "true" or "false"
```

Response: Returns the segmentation as a NIfTI file download.

### 5. Unload Model

```bash
DELETE /api/models/my_model
```

Response:
```json
{
  "status": "success",
  "message": "Model my_model unloaded"
}
```

## Example Usage

### Using curl

```bash
# 1. Check server health
curl http://localhost:8000/api/health

# 2. Load a model
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "abdomen_seg",
    "backend_type": "mmengine",
    "config_path": "/path/to/config.py",
    "model_path": "/path/to/checkpoint.pth",
    "inference_config": {
      "patch_size": [128, 128, 128],
      "patch_stride": [64, 64, 64]
    }
  }'

# 3. Run inference
curl -X POST http://localhost:8000/api/infer \
  -F "image=@input.nii.gz" \
  -F "model_name=abdomen_seg" \
  -F "force_cpu=false" \
  -o segmentation.nii.gz

# 4. Unload model
curl -X DELETE http://localhost:8000/api/models/abdomen_seg
```

### Using Python

```python
import requests

# Server URL
server_url = "http://localhost:8000"

# 1. Load model
response = requests.post(
    f"{server_url}/api/models",
    json={
        "name": "my_model",
        "backend_type": "mmengine",
        "config_path": "/path/to/config.py",
        "model_path": "/path/to/checkpoint.pth",
        "inference_config": {
            "patch_size": [96, 96, 96],
            "patch_stride": [48, 48, 48]
        }
    }
)
print(response.json())

# 2. Run inference
with open("input.nii.gz", "rb") as f:
    response = requests.post(
        f"{server_url}/api/infer",
        files={"image": f},
        data={
            "model_name": "my_model",
            "force_cpu": "false"
        }
    )

# Save segmentation
with open("segmentation.nii.gz", "wb") as f:
    f.write(response.content)

print("Inference completed!")
```

## Configuration

### Inference Config Parameters

- `patch_size`: Size of sliding window patches (e.g., `[96, 96, 96]`)
- `patch_stride`: Stride between patches (e.g., `[48, 48, 48]`)
- `fp16`: Use half-precision (faster, less memory)
- `accumulate_device`: Device for accumulation (`"cuda"` or `"cpu"`)
- `forward_device`: Device for inference (`"cuda"` or `"cpu"`)

### Server Options

```bash
python itkit_server.py --help

Options:
  --host HOST        Host address (default: 127.0.0.1)
  --port PORT        Port number (default: 8000)
  --debug            Enable debug mode
  --model-dir DIR    Directory to auto-load models from
```

## Deployment

### Docker (Recommended for Production)

Create `Dockerfile`:
```dockerfile
FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install itkit[advanced]

# Copy server code
COPY itkit_server.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "itkit_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t itkit-server .
docker run -p 8000:8000 --gpus all itkit-server
```

### Systemd Service (Linux)

Create `/etc/systemd/system/itkit-server.service`:
```ini
[Unit]
Description=ITKIT Inference Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/SlicerITKIT/server
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python itkit_server.py --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable itkit-server
sudo systemctl start itkit-server
sudo systemctl status itkit-server
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process or use a different port
python itkit_server.py --port 8001
```

### CUDA Out of Memory

- Reduce `patch_size` in inference config
- Use `force_cpu=true` in inference requests
- Unload unused models to free GPU memory
- Enable `fp16` for lower memory usage

### Model Loading Fails

- Check that config and checkpoint paths are correct
- Ensure ITKIT is installed with correct backend: `pip install itkit[advanced]`
- Check server logs for detailed error messages

### Connection Refused

- Ensure server is running: `curl http://localhost:8000/api/health`
- Check firewall settings if connecting from another machine
- Use `--host 0.0.0.0` to accept external connections

## Security Considerations

**⚠️ Important**: This server is designed for local or trusted network use.

For production deployment:
- Use HTTPS with SSL/TLS certificates
- Implement authentication (API keys, OAuth, etc.)
- Add rate limiting to prevent abuse
- Run behind a reverse proxy (nginx, Apache)
- Use firewall rules to restrict access
- Validate all file uploads
- Set file size limits

## Performance Tips

1. **Pre-load Models**: Load models at server startup to avoid delays
2. **Use ONNX**: ONNX backend is optimized for inference
3. **Batch Processing**: Use appropriate patch sizes for GPU
4. **Memory Management**: Unload unused models regularly
5. **Threading**: Server uses Flask's threaded mode for concurrent requests

## Support

- GitHub Issues: https://github.com/MGAMZ/ITKIT/issues
- Documentation: https://itkit.readthedocs.io/
- Email: 312065559@qq.com
