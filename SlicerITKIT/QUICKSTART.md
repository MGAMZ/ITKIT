# ITKIT Slicer Plugin - Quick Start Guide

This guide will help you get started with the ITKIT 3D Slicer plugin using the new client-server architecture.

## 5-Minute Quick Start

### Step 1: Start the Server (2 minutes)

```bash
# Navigate to server directory
cd SlicerITKIT/server

# Create virtual environment (first time only)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (first time only)
pip install -r requirements.txt
pip install itkit[advanced]

# Start server
python itkit_server.py
```

You should see:
```
============================================================
ITKIT Inference Server
============================================================
Host: 127.0.0.1
Port: 8000
CUDA Available: True
============================================================
```

### Step 2: Install Slicer Plugin (1 minute)

1. Open 3D Slicer
2. Go to: Edit → Application Settings → Modules
3. Click "Add" under "Additional module paths"
4. Navigate to and select: `SlicerITKIT/ITKITInference`
5. Click OK and restart Slicer

### Step 3: Connect and Use (2 minutes)

1. In Slicer, go to: Modules → Segmentation → ITKIT Inference
2. Server URL should be pre-filled: `http://localhost:8000`
3. Click "Connect to Server" - Status should show "Connected"
4. Load a test model:
   - Model Name: `test_model`
   - Backend: ONNX or MMEngine
   - Select your model file
   - Click "Load Model on Server"
5. Run inference:
   - Load a volume in Slicer
   - Select input volume
   - Select model
   - Click "Run Inference"

## Detailed Setup

### Server Setup

#### Using Virtual Environment (Recommended)

```bash
cd SlicerITKIT/server

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install server dependencies
pip install -r requirements.txt

# Install ITKIT with backends
pip install itkit[advanced]     # For MMEngine (PyTorch)
# or
pip install itkit[onnx]          # For ONNX only
# or
pip install itkit[advanced,onnx] # For both

# Start server
python itkit_server.py --host 0.0.0.0 --port 8000
```

#### Using Conda

```bash
cd SlicerITKIT/server

# Create conda environment
conda create -n itkit-server python=3.10
conda activate itkit-server

# Install dependencies
pip install -r requirements.txt
pip install itkit[advanced]

# Start server
python itkit_server.py
```

### Client (Slicer Plugin) Setup

#### Check Dependencies

In Slicer's Python console, check if requests is installed:

```python
import requests
print("requests is installed")
```

If not installed:

```python
import pip
pip.main(['install', 'requests'])
# Restart Slicer
```

#### Install Plugin

1. Open 3D Slicer
2. Edit → Application Settings → Modules
3. Click "Add" under "Additional module paths"
4. Select `SlicerITKIT/ITKITInference` directory
5. Click OK
6. Restart Slicer

#### Verify Installation

1. Go to: Modules → Segmentation → ITKIT Inference
2. Module should load without errors
3. You should see "Server Connection" section

## First Inference

### Prepare a Model

You need one of:
- **MMEngine**: config.py + checkpoint.pth
- **ONNX**: model.onnx

### Load Model on Server

1. In Slicer's ITKIT Inference module:
   - Enter model name: e.g., `liver_seg`
   - Select backend: MMEngine or ONNX
   - Browse and select files:
     - Config file (MMEngine only)
     - Model/checkpoint file
   - (Optional) Set inference params:
     - Patch size: `128,128,128`
     - Patch stride: `64,64,64`
   - Click "Load Model on Server"
   
2. Wait for model to load (check status)

### Run Inference

1. Load a medical image:
   - File → Add Data
   - Or use Sample Data module

2. In ITKIT Inference module:
   - Select input volume
   - Select loaded model
   - Click "Run Inference"

3. View results:
   - Segmentation appears in slice views
   - Use Segment Editor to refine if needed

## Common Workflows

### Workflow 1: Single Local Inference

```
Start Server → Connect in Slicer → Load Model → Run Inference
```

### Workflow 2: Batch Processing

```
Start Server → Load Model via API → Process images via Python script
```

### Workflow 3: Remote Server

```
Deploy Server on GPU machine → Connect from Slicer on workstation → Run inference
```

## Example: Complete Workflow

```bash
# Terminal 1: Start server
cd SlicerITKIT/server
source venv/bin/activate
python itkit_server.py

# Terminal 2 (optional): Load model via API
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "abdomen_seg",
    "backend_type": "onnx",
    "model_path": "/path/to/model.onnx"
  }'
```

Then in Slicer:
1. Open ITKIT Inference module
2. Connect to http://localhost:8000
3. See "abdomen_seg" in loaded models
4. Load volume and run inference

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Can't connect | Check server is running: `curl http://localhost:8000/api/health` |
| Model won't load | Check file paths, verify files exist |
| Inference fails | Try smaller patch size or force CPU |
| Server won't start | Port in use - use different port: `--port 8001` |
| requests not found | Install in Slicer: `pip.main(['install', 'requests'])` |

## Next Steps

- **Deployment**: See [server/README.md](server/README.md) for Docker and production setup
- **API Usage**: Use REST API directly for batch processing
- **Custom Models**: Export your models to ONNX for faster inference
- **Remote Access**: Deploy server on GPU machine, access from multiple Slicer clients

## Support

- Issues: https://github.com/MGAMZ/ITKIT/issues
- Docs: https://itkit.readthedocs.io/
- Email: 312065559@qq.com

## Tips

💡 **Server Performance**
- Pre-load models to avoid delays
- Use ONNX for production (faster)
- Monitor GPU memory usage

💡 **Slicer Tips**
- Save server URL in settings (auto-remembered)
- Keep models loaded between inferences
- Use "Force CPU" if GPU OOM occurs

💡 **Development**
- Run server with `--debug` for detailed logs
- Use `--host 0.0.0.0` for external access
- Check server logs for errors
