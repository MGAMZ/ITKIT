from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .sliding_window import ArgmaxProcessor, InferenceConfig, slide_inference_3d


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, inference_config: InferenceConfig | dict | None = None, allow_tqdm: bool = True):
        """Initialize inference backend.

        Args:
            inference_config: Configuration for sliding-window inference.
            allow_tqdm: Whether to show progress bars.
        """
        if inference_config is None:
            self.inference_config = InferenceConfig()
        elif isinstance(inference_config, InferenceConfig):
            self.inference_config = inference_config
        elif isinstance(inference_config, dict):
            self.inference_config = InferenceConfig(**inference_config)
        else:
            raise TypeError(f'inference_config must be InferenceConfig, dict or None, but got {type(inference_config)}')

        # Default stride to half of patch_size if not set
        if self.inference_config.patch_size is not None and self.inference_config.patch_stride is None:
            self.inference_config.patch_stride = tuple(max(1, s // 2) for s in self.inference_config.patch_size)

        self.allow_tqdm = allow_tqdm
        self.argmax_processor = ArgmaxProcessor(self.inference_config)

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            inputs: Input tensor.

        Returns:
            Output logits tensor.
        """
        pass

    @abstractmethod
    def slide_inference(self, inputs: Tensor) -> Tensor:
        """Perform sliding-window inference.

        Args:
            inputs: Input tensor.

        Returns:
            Segmentation logits tensor.
        """
        pass


class MMEngineInferBackend(InferenceBackend):
    """Inference backend using MMEngine model loading."""

    def __init__(self,
                 cfg_path: str,
                 ckpt_path: str,
                 inference_config: InferenceConfig | dict | None = None,
                 allow_tqdm: bool = True):
        """Initialize MMEngine inference backend.

        Args:
            cfg_path: Path to config file.
            ckpt_path: Path to checkpoint file.
            inference_config: Configuration for sliding-window inference.
            allow_tqdm: Whether to show progress bars.
        """
        from mmengine.config import Config
        from mmengine.registry import MODELS
        from mmengine.runner import load_checkpoint

        self.cfg = Config.fromfile(cfg_path)
        base_infer_cfg = getattr(self.cfg.model, 'inference_config', None)
        if base_infer_cfg is None:
            base_cfg = InferenceConfig()
        elif isinstance(base_infer_cfg, InferenceConfig):
            base_cfg = base_infer_cfg
        elif isinstance(base_infer_cfg, dict):
            base_cfg = InferenceConfig(**base_infer_cfg)
        else:
            raise TypeError(f"inference_config must be InferenceConfig, dict or None, but got {type(base_infer_cfg)}")

        merged_infer_cfg = base_cfg.merge(inference_config)
        super().__init__(merged_infer_cfg, allow_tqdm)
        self.model = MODELS.build(self.cfg.model)
        load_checkpoint(self.model, ckpt_path, map_location='cpu')
        self.model.eval()
        self.model.cuda()
        self.model.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            inputs: Input tensor.

        Returns:
            Output logits tensor.
        """
        return self.model.backbone(inputs)

    @torch.inference_mode()
    def slide_inference(self, inputs: Tensor, force_cpu: bool = False) -> Tensor:
        """Perform sliding-window inference with overlapping sub-volumes.

        Args:
            inputs: Input tensor of shape (N, C, Z, Y, X).
            force_cpu: Whether to force accumulation on CPU to avoid GPU OOM.

        Returns:
            Tensor: Segmentation logits.
        """
        # Check if sliding window is enabled
        if self.inference_config.patch_size is None or self.inference_config.patch_stride is None:
            return self.forward(inputs)

        return slide_inference_3d(
            inputs=inputs,
            num_classes=self.model.num_classes,
            inference_config=self.inference_config,
            forward_func=self.forward,
            allow_tqdm=self.allow_tqdm,
            force_cpu=force_cpu,
            progress_desc="Slide Win. Infer."
        )


class ONNXInferBackend(InferenceBackend):
    """Inference backend using ONNX Runtime."""

    def __init__(self,
                 onnx_path: str,
                 inference_config: InferenceConfig | dict | None = None,
                 allow_tqdm: bool = True,
                 providers: list[str] | None = None):
        """Initialize ONNX inference backend.

        Args:
            onnx_path: Path to ONNX model file.
            inference_config: Configuration for sliding-window inference.
            allow_tqdm: Whether to show progress bars.
            providers: ONNX Runtime execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        import onnxruntime as ort  # pyright: ignore[reportMissingImports]
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Prefer model-embedded inference_config as base, then apply overrides.
        base_infer_cfg = None
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw_infer_cfg = meta.get('inference_config')
        if raw_infer_cfg:
            try:
                base_infer_cfg = InferenceConfig.model_validate_json(raw_infer_cfg)
            except Exception:
                pass

        if base_infer_cfg is None:
            base_cfg = InferenceConfig()
        else:
            base_cfg = base_infer_cfg

        merged_infer_cfg = base_cfg.merge(inference_config)
        super().__init__(merged_infer_cfg, allow_tqdm)

        # Get input/output names
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        output_info = self.session.get_outputs()[0]
        self.output_name = output_info.name

        # Shape is NCZYX
        input_shape = input_info.shape
        output_shape = output_info.shape
        self.num_classes = output_shape[1]

        # Parse ZYX as patch_size from input_shape (always use model input shape, ignore user config)
        spatial_shape = input_shape[2:]
        if len(spatial_shape) != 3:
            raise ValueError(f"Input shape must be 5D (N,C,Z,Y,X) for 3D segmentation, got {input_shape}.")
        self.inference_config.patch_size = tuple(spatial_shape)

        # Allow user to configure patch_stride, otherwise default to half of patch_size
        if self.inference_config.patch_stride is None:
            self.inference_config.patch_stride = tuple(s//2 for s in self.inference_config.patch_size)

        # Ensure batch size matches if fixed in model
        if isinstance(input_shape[0], int) and input_shape[0] > 0:
            if self.inference_config.forward_batch_windows != input_shape[0]:
                self.inference_config.forward_batch_windows = input_shape[0]

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the ONNX model.

        Args:
            inputs: Input tensor.

        Returns:
            Output logits tensor.
        """
        # Convert to numpy for ONNX Runtime
        inputs_np = inputs.cpu().numpy()
        outputs = self.session.run([self.output_name], {self.input_name: inputs_np})
        # Convert back to torch tensor
        return torch.from_numpy(outputs[0]).to(inputs.device)

    @torch.inference_mode()
    def slide_inference(self, inputs: Tensor, force_cpu: bool = False) -> Tensor:
        """Perform sliding-window inference with overlapping sub-volumes.

        Args:
            inputs: Input tensor of shape (N, C, Z, Y, X).
            force_cpu: Whether to force accumulation on CPU to avoid GPU OOM.

        Returns:
            Tensor: Segmentation logits.
        """
        # Check if sliding window is enabled
        if self.inference_config.patch_size is None or self.inference_config.patch_stride is None:
            return self.forward(inputs)

        return slide_inference_3d(
            inputs=inputs,
            num_classes=self.num_classes,
            inference_config=self.inference_config,
            forward_func=self.forward,
            allow_tqdm=self.allow_tqdm,
            force_cpu=force_cpu,
            progress_desc="Slide Win. Infer. (ONNX)"
        )


class Inferencer(ABC):
    """Base inferencer class that delegates to inference backends."""

    def __init__(self, backend: InferenceBackend, fp16: bool = False, allow_tqdm: bool = True):
        """Initialize inferencer with a backend.

        Args:
            backend: Inference backend to use.
            fp16: Whether to use FP16 (for input conversion).
            allow_tqdm: Whether to show progress bars.
        """
        self.backend = backend
        self.fp16 = fp16
        self.allow_tqdm = allow_tqdm

    @abstractmethod
    @torch.inference_mode()
    def Inference_FromNDArray(self, inputs: np.ndarray) -> tuple[Tensor, Tensor]:
        """Accept ndarray input and perform inference.

        Args:
            inputs (np.ndarray): Input image ndarray.

        Returns:
            tuple (Tensor, Tensor):
                - seg_logits (Tensor): Segmentation logits tensor.
                - sem_seg_map (Tensor): Segmentation map tensor.
        """
        pass


class Inferencer_Seg3D(Inferencer):
    """3D segmentation inferencer using backend modules."""

    def __init__(self, backend: InferenceBackend, fp16: bool = False, allow_tqdm: bool = True):
        super().__init__(backend=backend, fp16=fp16, allow_tqdm=allow_tqdm)

    @torch.inference_mode()
    def Inference_FromNDArray(self, inputs: np.ndarray) -> tuple[Tensor, Tensor]:
        """Perform 3D segmentation inference.

        Args:
            inputs: 3D numpy array with shape (Z, Y, X).

        Returns:
            tuple[Tensor, Tensor]:
                - seg_logits: Segmentation logits tensor with shape (N, C, Z, Y, X).
                - sem_seg_map: Segmentation map tensor with shape (N, Z, Y, X).
        """
        assert inputs.ndim == 3, f"Input image must be (Z, Y, X), got: {inputs.shape}."

        inputs = inputs.astype(np.float16 if self.fp16 else np.float32)
        tensor_input = torch.from_numpy(inputs[None, None])
        torch.cuda.empty_cache()

        with torch.autocast('cuda'):
            seg_logits = self.backend.slide_inference(tensor_input)  # [N,C,Z,Y,X]
        sem_seg_map = self._batched_argmax(seg_logits)  # [N,C,Z,Y,X] -> [N,Z,Y,X]

        return seg_logits, sem_seg_map

    def _batched_argmax(self, inputs: Tensor) -> Tensor:
        """Compute argmax in batches along Z-axis to avoid OOM.

        Args:
            inputs: Input tensor with shape (N, C, Z, Y, X).

        Returns:
            Tensor: Argmax result with shape (N, Z, Y, X).
        """
        argmax_batchsize: int = self.backend.inference_config.argmax_batchsize or 16
        forward_device: str = self.backend.inference_config.forward_device
        assert inputs.ndim == 5, f"Input tensor must be (N, C, Z, Y, X), got: {format(inputs.shape)}."
        N, C, Z, Y, X = inputs.shape

        sem_seg_map = torch.empty((N, Z, Y, X), dtype=torch.uint8)
        for start_z in tqdm(range(0, Z, argmax_batchsize),
                            desc="Batched ArgMax",
                            dynamic_ncols=True,
                            leave=False,
                            mininterval=1,
                            disable=not self.allow_tqdm):
            end_z = min(start_z + argmax_batchsize, Z)
            batch_sem_seg_map = inputs[:, :, start_z:end_z].to(device=forward_device).argmax(dim=1).to(torch.uint8)
            sem_seg_map[:, start_z:end_z].copy_(batch_sem_seg_map, non_blocking=True)

        return sem_seg_map  # [N, Z, Y, X]
