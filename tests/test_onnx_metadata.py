import torch
import onnx
import json
import pytest

try:
    import onnx
    import onnxruntime
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from itkit.mm.inference import ONNXInferBackend

def create_dummy_onnx(path, inference_config_dict=None):
    # Create a simple model: y = x
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()
    # Using 5D input for 3D segmentation test
    dummy_input = torch.randn(1, 2, 32, 32, 32)
    torch.onnx.export(model, dummy_input, path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11)

    if inference_config_dict:
        onnx_model = onnx.load(path)
        meta = onnx_model.metadata_props.add()
        meta.key = 'inference_config'
        meta.value = json.dumps(inference_config_dict)
        onnx.save(onnx_model, path)

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
def test_onnx_infer_backend_metadata_parsing(tmp_path):
    onnx_path = str(tmp_path / "model_with_meta.onnx")
    config_dict = {
        "patch_size": [16, 16, 16],
        "patch_stride": [8, 8, 8],
        "forward_batch_windows": 4
    }

    create_dummy_onnx(onnx_path, config_dict)

    # Initialize backend without passing inference_config
    # It should pick it up from metadata
    backend = ONNXInferBackend(onnx_path, providers=['CPUExecutionProvider'])

    assert backend.inference_config.patch_size == (16, 16, 16)
    assert backend.inference_config.patch_stride == (8, 8, 8)
    assert backend.inference_config.forward_batch_windows == 4

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
def test_onnx_infer_backend_no_metadata_inference_logic(tmp_path):
    onnx_path = str(tmp_path / "model_no_meta.onnx")
    create_dummy_onnx(onnx_path)

    # Should use defaults or inferred values
    backend = ONNXInferBackend(onnx_path, providers=['CPUExecutionProvider'])

    # Injected dummy input was (1, 2, 32, 32, 32)
    # ONNXInferBackend parses spatial shape if patch_size is None
    assert backend.inference_config.patch_size == (32, 32, 32)
    # Stride defaults to patch_size // 2 if not set
    assert backend.inference_config.patch_stride == (16, 16, 16)
    assert backend.num_classes == 2
