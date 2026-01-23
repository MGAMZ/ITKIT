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

# Configure logging (will be updated in main() based on --debug flag)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global storage for the single loaded model
current_model: Optional["ModelConfig"] = None


class ModelConfig:
    """Configuration for a loaded model."""

    def __init__(
        self,
        name: str,
        backend_type: str,
        config_path: Optional[str],
        model_path: str,
        inference_config: Optional[dict] = None,
    ):
        self.name = name
        self.backend_type = backend_type  # 'mmengine' or 'onnx'
        self.config_path = config_path
        self.model_path = model_path
        self.inference_config = inference_config or {}
        self.backend = None
        self.inferencer = None

    def load(self):
        """Load the model backend and inferencer."""
        logger.info("Loading model: %s", self.name)
        infer_cfg = (
            InferenceConfig(**self.inference_config)
            if self.inference_config
            else InferenceConfig()
        )

        if self.backend_type.lower() == "mmengine":
            if not self.config_path or not os.path.exists(self.config_path):
                raise ValueError(f"Config file required: {self.config_path}")
            self.backend = MMEngineInferBackend(
                cfg_path=self.config_path,
                ckpt_path=self.model_path,
                inference_config=infer_cfg,
                allow_tqdm=False,
            )
        elif self.backend_type.lower() == "onnx":
            self.backend = ONNXInferBackend(
                onnx_path=self.model_path, inference_config=infer_cfg, allow_tqdm=False
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")

        self.inferencer = Inferencer_Seg3D(
            backend=self.backend,
            fp16=self.inference_config.get("fp16", False),
            allow_tqdm=False,
        )
        logger.info("Model loaded: %s", self.name)

    def infer(self, image_array: np.ndarray, force_cpu: bool = False) -> tuple:
        """Run inference on image array."""
        if force_cpu:
            self.backend.inference_config.accumulate_device = "cpu"
        seg_logits, sem_seg_map = self.inferencer.Inference_FromNDArray(image_array)
        return seg_logits.cpu().numpy(), sem_seg_map.cpu().numpy()

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "backend_type": self.backend_type,
            "config_path": self.config_path,
            "model_path": self.model_path,
            "inference_config": self.inference_config,
            "loaded": self.backend is not None,
        }


def _get_windowing_from_model(
    model: ModelConfig,
) -> tuple[Optional[float], Optional[float]]:
    """Try to read window level/width from backend metadata or config."""
    if model.backend_type.lower() == "mmengine":
        cfg = getattr(model.backend, "cfg", None)
        if cfg:
            wl = cfg.get("wl") or cfg.get("window_level")
            ww = cfg.get("ww") or cfg.get("window_width")
            if wl is None or ww is None:
                cfg_model = getattr(cfg, "model", None)
                if cfg_model:
                    wl = wl or cfg_model.get("wl") or cfg_model.get("window_level")
                    ww = ww or cfg_model.get("ww") or cfg_model.get("window_width")
            return wl, ww
    elif model.backend_type.lower() == "onnx":
        meta = model.backend.session.get_modelmeta().custom_metadata_map or {}
        return meta.get("window_level"), meta.get("window_width")
    return None, None


def _apply_windowing(image_array: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Apply windowing and normalize to [0,1]."""
    left = wl - ww / 2.0
    right = wl + ww / 2.0
    image_array = np.clip(image_array.astype(np.float32), left, right)
    image_array = (image_array - left) / ww
    return image_array


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        }
    )


@app.route("/api/info", methods=["GET"])
def get_info():
    """Get server information and loaded model."""
    model_info = current_model.to_dict() if current_model else None

    return jsonify(
        {
            "name": "ITKIT Inference Server",
            "version": "1.0.0",
            "model": model_info,
            "cuda_available": torch.cuda.is_available(),
            "device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        }
    )


@app.route("/api/model", methods=["POST"])
def load_model():
    """Load a new model (replaces any currently loaded model).

    Request body:
    {
        \"backend_type\": \"mmengine\" or \"onnx\",
        \"config_path\": \"/path/to/config.py\" (required for MMEngine),
        \"model_path\": \"/path/to/checkpoint.pth\" or \"/path/to/model.onnx\",
        \"inference_config\": {
            \"patch_size\": [96, 96, 96],
            \"patch_stride\": [48, 48, 48],
            \"fp16\": false
        }
    }
    """
    global current_model

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True) or {}
    logger.debug("/api/model payload: %s", data)
    backend_type = data.get("backend_type")
    config_path = data.get("config_path")
    model_path = data.get("model_path")
    inference_config = data.get("inference_config", {})

    if not backend_type:
        return jsonify({"error": "Backend type is required"}), 400
    if not model_path:
        return jsonify({"error": "Model path is required"}), 400
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model file not found: {model_path}"}), 400
    if backend_type.lower() == "mmengine" and (
        not config_path or not os.path.exists(config_path)
    ):
        return (
            jsonify({"error": f"Config file required for MMEngine: {config_path}"}),
            400,
        )

    if current_model:
        logger.info("Unloading current model")
        del current_model
        torch.cuda.empty_cache()

    name = Path(model_path).stem
    model = ModelConfig(name, backend_type, config_path, model_path, inference_config)
    model.load()
    current_model = model

    return jsonify({"status": "success", "model": model.to_dict()})


@app.route("/api/model", methods=["DELETE"])
def unload_model():
    """Unload the current model to free memory."""
    global current_model

    if not current_model:
        return jsonify({"error": "No model loaded"}), 404

    logger.info("Unloading model: %s", current_model.name)
    del current_model
    current_model = None
    torch.cuda.empty_cache()
    return jsonify({"status": "success"})


@app.route("/api/infer", methods=["POST"])
def run_inference():
    """Run inference on uploaded image.

    Expects multipart form data with:
    - image: Image file (NIfTI format)
    - force_cpu: (optional) Force CPU accumulation
    - window_level: (optional) Override window level
    - window_width: (optional) Override window width
    """
    if current_model is None:
        return jsonify({"error": "No model loaded"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    force_cpu = request.form.get("force_cpu", "false").lower() == "true"
    wl_override = request.form.get("window_level")
    ww_override = request.form.get("window_width")

    # Save and load image
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        file.save(tmp.name)
        tmp_input_path = tmp.name

    image = sitk.ReadImage(tmp_input_path)
    image_lpi = sitk.DICOMOrient(image, "LPI")
    image_array = sitk.GetArrayFromImage(image_lpi)
    os.unlink(tmp_input_path)

    # Resolve windowing
    wl = float(wl_override) if wl_override else None
    ww = float(ww_override) if ww_override else None
    if not wl or not ww:
        model_wl, model_ww = _get_windowing_from_model(current_model)
        wl = wl or model_wl
        ww = ww or model_ww
    if not wl or not ww:
        return jsonify({"error": "window_level/window_width required"}), 400

    image_array = _apply_windowing(image_array, wl, ww)
    logger.info(
        "Inference: model=%s shape=%s WL=%.1f WW=%.1f",
        current_model.name,
        image_array.shape,
        wl,
        ww,
    )

    seg_logits, sem_seg_map = current_model.infer(image_array, force_cpu=force_cpu)
    seg_map_squeezed = sem_seg_map.squeeze()

    # Create output
    seg_image = sitk.GetImageFromArray(seg_map_squeezed.astype(np.uint8))
    seg_image.SetOrigin(image_lpi.GetOrigin())
    seg_image.SetSpacing(image_lpi.GetSpacing())
    seg_image.SetDirection(image_lpi.GetDirection())

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_output_path = tmp.name
    sitk.WriteImage(seg_image, tmp_output_path)

    logger.info("Inference completed. Labels: %s", np.unique(seg_map_squeezed))

    response = send_file(
        tmp_output_path,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="segmentation.nii.gz",
    )
    response.call_on_close(
        lambda: os.path.exists(tmp_output_path) and os.unlink(tmp_output_path)
    )
    return response


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ITKIT Inference Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number (default: 8000)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("ITKIT Inference Server")
    logger.info("=" * 60)
    logger.info("Host: %s Port: %d Debug: %s", args.host, args.port, args.debug)
    logger.info(
        "CUDA: %s Devices: %d",
        torch.cuda.is_available(),
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
