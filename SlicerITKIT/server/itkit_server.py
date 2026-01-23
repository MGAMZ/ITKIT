"""
ITKIT Inference Server

A lightweight REST API server for running ITKIT inference.
This server runs independently with its own Python environment,
separate from 3D Slicer's environment.

Usage:
    python itkit_server.py --host 0.0.0.0 --port 8000

API Endpoints:
    GET  /api/info          - Get server info and available models
    POST /api/infer         - Run inference on uploaded image
    GET  /api/health        - Health check
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from itkit.mm.inference import (
    InferenceConfig,
    Inferencer_Seg3D,
    MMEngineInferBackend,
    ONNXInferBackend,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global storage for loaded models
loaded_models = {}


class ModelConfig:
    """Configuration for a loaded model."""
    
    def __init__(self, name: str, backend_type: str, config_path: Optional[str], 
                 model_path: str, inference_config: Optional[dict] = None):
        self.name = name
        self.backend_type = backend_type  # 'mmengine' or 'onnx'
        self.config_path = config_path
        self.model_path = model_path
        self.inference_config = inference_config or {}
        self.backend = None
        self.inferencer = None
    
    def load(self):
        """Load the model backend and inferencer."""
        logger.info(f"Loading model: {self.name}")
        
        # Parse inference config
        infer_cfg = InferenceConfig(**self.inference_config) if self.inference_config else InferenceConfig()
        
        # Initialize backend
        if self.backend_type.lower() == 'mmengine':
            if not self.config_path or not os.path.exists(self.config_path):
                raise ValueError(f"Config file required for MMEngine backend: {self.config_path}")
            
            self.backend = MMEngineInferBackend(
                cfg_path=self.config_path,
                ckpt_path=self.model_path,
                inference_config=infer_cfg,
                allow_tqdm=False
            )
        elif self.backend_type.lower() == 'onnx':
            self.backend = ONNXInferBackend(
                onnx_path=self.model_path,
                inference_config=infer_cfg,
                allow_tqdm=False
            )
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
        
        # Create inferencer
        self.inferencer = Inferencer_Seg3D(
            backend=self.backend,
            fp16=self.inference_config.get('fp16', False),
            allow_tqdm=False
        )
        
        logger.info(f"Model loaded successfully: {self.name}")
    
    def infer(self, image_array: np.ndarray, force_cpu: bool = False) -> tuple:
        """Run inference on image array.
        
        Args:
            image_array: Input image as numpy array (Z, Y, X)
            force_cpu: Force CPU accumulation
            
        Returns:
            Tuple of (seg_logits, sem_seg_map) as numpy arrays
        """
        if self.inferencer is None:
            raise RuntimeError(f"Model not loaded: {self.name}")
        
        # Update backend config if force_cpu is specified
        if force_cpu:
            self.backend.inference_config.accumulate_device = 'cpu'
        
        # Run inference
        seg_logits, sem_seg_map = self.inferencer.Inference_FromNDArray(image_array)
        
        # Convert to numpy
        seg_logits_np = seg_logits.cpu().numpy()
        sem_seg_map_np = sem_seg_map.cpu().numpy()
        
        return seg_logits_np, sem_seg_map_np
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'backend_type': self.backend_type,
            'config_path': self.config_path,
            'model_path': self.model_path,
            'inference_config': self.inference_config,
            'loaded': self.backend is not None
        }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get server information and loaded models."""
    models_info = {name: model.to_dict() for name, model in loaded_models.items()}
    
    return jsonify({
        'name': 'ITKIT Inference Server',
        'version': '1.0.0',
        'models': models_info,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })


@app.route('/api/models', methods=['POST'])
def load_model():
    """Load a new model.
    
    Request body:
    {
        "name": "my_model",
        "backend_type": "mmengine" or "onnx",
        "config_path": "/path/to/config.py" (optional for ONNX),
        "model_path": "/path/to/checkpoint.pth" or "/path/to/model.onnx",
        "inference_config": {
            "patch_size": [96, 96, 96],
            "patch_stride": [48, 48, 48],
            "fp16": false
        }
    }
    """
    try:
        data = request.json
        name = data.get('name')
        backend_type = data.get('backend_type')
        config_path = data.get('config_path')
        model_path = data.get('model_path')
        inference_config = data.get('inference_config', {})
        
        # Validate
        if not name:
            return jsonify({'error': 'Model name is required'}), 400
        if not backend_type:
            return jsonify({'error': 'Backend type is required'}), 400
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 400
        
        # Create and load model
        model = ModelConfig(name, backend_type, config_path, model_path, inference_config)
        model.load()
        
        # Store
        loaded_models[name] = model
        
        return jsonify({
            'status': 'success',
            'model': model.to_dict()
        })
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_name>', methods=['DELETE'])
def unload_model(model_name):
    """Unload a model to free memory."""
    if model_name in loaded_models:
        del loaded_models[model_name]
        torch.cuda.empty_cache()
        return jsonify({'status': 'success', 'message': f'Model {model_name} unloaded'})
    else:
        return jsonify({'error': f'Model not found: {model_name}'}), 404


@app.route('/api/infer', methods=['POST'])
def run_inference():
    """Run inference on uploaded image.
    
    Expects multipart form data with:
    - image: Image file (NIfTI format)
    - model_name: Name of the model to use
    - force_cpu: (optional) Force CPU accumulation
    """
    try:
        # Get parameters
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        model_name = request.form.get('model_name')
        force_cpu = request.form.get('force_cpu', 'false').lower() == 'true'
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        if model_name not in loaded_models:
            return jsonify({'error': f'Model not loaded: {model_name}'}), 404
        
        # Save uploaded file
        file = request.files['image']
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            file.save(tmp.name)
            tmp_input_path = tmp.name
        
        # Load image
        image = sitk.ReadImage(tmp_input_path)
        image_array = sitk.GetArrayFromImage(image)
        os.unlink(tmp_input_path)
        
        # Run inference
        model = loaded_models[model_name]
        logger.info(f"Running inference with model: {model_name}, shape: {image_array.shape}")
        seg_logits, sem_seg_map = model.infer(image_array, force_cpu=force_cpu)
        
        # Create segmentation image
        seg_map_squeezed = sem_seg_map.squeeze()  # Remove batch dimension
        seg_image = sitk.GetImageFromArray(seg_map_squeezed.astype(np.uint8))
        
        # Copy metadata from input image
        seg_image.SetOrigin(image.GetOrigin())
        seg_image.SetSpacing(image.GetSpacing())
        seg_image.SetDirection(image.GetDirection())
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp_output_path = tmp.name
        
        sitk.WriteImage(seg_image, tmp_output_path)
        
        logger.info(f"Inference completed. Unique labels: {np.unique(seg_map_squeezed)}")
        
        # Return the file
        return send_file(
            tmp_output_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='segmentation.nii.gz'
        )
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ITKIT Inference Server')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port number (default: 8000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Directory to auto-load models from')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("="*60)
    logger.info("ITKIT Inference Server")
    logger.info("="*60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
    logger.info("="*60)
    
    # Auto-load models if model directory specified
    if args.model_dir and os.path.exists(args.model_dir):
        logger.info(f"Auto-loading models from: {args.model_dir}")
        # TODO: Implement auto-loading logic
    
    # Start server
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
