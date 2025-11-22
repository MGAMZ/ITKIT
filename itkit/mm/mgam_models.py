import logging
import pdb
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from mmengine.config import ConfigDict
from mmengine.dist import is_main_process
from mmengine.logging import print_log
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement, PixelData
from torch import Tensor
from tqdm import tqdm

from .mmseg_Dev3D import VolumeData


@dataclass
class InferenceConfig:
    """Configuration for sliding-window and device settings during inference.

    Attributes:
        patch_size (tuple | None): Sliding window size. None disables sliding window.
        patch_stride (tuple | None): Sliding window stride. None disables sliding window.
        accumulate_device (str): Device for accumulating window results, e.g., 'cpu' or 'cuda'.
        forward_device (str): Device to run the forward pass for each window.
    """
    patch_size: tuple | None = None
    patch_stride: tuple | None = None
    accumulate_device: str = 'cuda'
    forward_device: str = 'cuda'
    forward_batch_windows: int = 1
    # When accumulate and forward devices differ, a chunk size along the last
    # dimension must be provided to avoid OOM during argmax transfer.
    argmax_batchsize: int | None = None


class ArgmaxProcessor:
    """Device-aware argmax with optional chunking along the last dimension.

    Advantages:
    - Avoids OOM on device when handling large tensors.
    - ArgMax can utilize GPU acceleration instead of fully relying on CPU.

    Behavior:
    - Always compute argmax on forward_device.
    - If accumulate_device and forward_device are the same, perform argmax on
      the full tensor directly.
    - If they differ, require `argmax_batchsize` to chunk along the last
      dimension to avoid OOM; per-chunk results are transferred back and
      concatenated on accumulate_device.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def argmax(self, logits: Tensor, dim: int = 1, keepdim: bool = True) -> Tensor:
        acc_dev = torch.device(self.config.accumulate_device)
        fwd_dev = torch.device(self.config.forward_device)

        # Fast path: same device
        if acc_dev.type == fwd_dev.type:
            # Ensure tensor on forward/accum device (same type)
            t = logits.to(fwd_dev)
            preds = torch.argmax(t, dim=dim, keepdim=keepdim).to(torch.uint8)
            # Return on accumulate device (identical type)
            return preds.to(acc_dev)

        # Different devices: require chunk size
        chunk_size = self.config.argmax_batchsize
        if chunk_size is None or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(
                "When accumulate_device and forward_device differ, 'argmax_batchsize' "
                f"must be a positive int in InferenceConfig to enable chunked argmax. Got {chunk_size}"
            )

        # Chunk along the last dimension
        # Temp batch is transferred to forward device for argmax,
        # then results are transferred back to accumulate device.
        last_dim = logits.dim() - 1
        L = logits.shape[-1]
        chunks: list[Tensor] = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            slc = [slice(None)] * logits.dim()
            slc[last_dim] = slice(start, end)
            t_chunk = logits[tuple(slc)].to(fwd_dev)
            preds_chunk = torch.argmax(t_chunk, dim=dim, keepdim=keepdim).to(torch.uint8)
            chunks.append(preds_chunk.to(acc_dev))

        # Concatenate back along the last dimension on accumulate device
        pred = torch.cat(chunks, dim=last_dim if keepdim is False else last_dim)
        return pred


class mgam_Seg_Lite(BaseModel):
    def __init__(self,
                 backbone: ConfigDict,
                 criterion: ConfigDict|list[ConfigDict],
                 num_classes: int,
                 gt_sem_seg_key: str = 'gt_sem_seg',
                 use_half: bool = False,
                 binary_segment_threshold: float|None = None,
                 inference_config: InferenceConfig | dict | None = None,
                 allow_pbar: bool = False,
                 *args, **kwargs):
        """
        mgam_Seg_Lite is a Lite form of `mmseg` core model implementation,
        without decouple decoder_head, loss, neck design, allowing easier coding experiments.
        Meanwhile, it provides several args to support sliding window inference for large image/volume,
        especially useful for medical image segmentation tasks.

        Args:
            backbone (ConfigDict): Configuration of the backbone network, including the merged decode_head. This backbone should directly output the final segmentation logits.
            criterion (ConfigDict): Criterion for computing loss, such as Dice loss or cross-entropy loss.
            gt_sem_seg_key (str): The key name for the ground truth segmentation mask, default is 'gt_sem_seg'.
            use_half (bool): Whether to use half-precision (fp16) for the model, default is False.
            binary_segment_threshold (float | None): Threshold for binary segmentation. If the model outputs a single channel (binary), this must be provided; if the model outputs multiple channels (multi-class), this must be None.
            inference_config (InferenceConfig | dict | None): Inference configuration that groups sliding-window and device options. If dict, keys can be
                {'patch_size', 'patch_stride', 'accumulate_device', 'forward_device'}. If None, sliding-window is disabled.
        """
        super().__init__(*args, **kwargs)
        self.backbone = MODELS.build(backbone)
        self.criterion = [MODELS.build(c) for c in criterion] if isinstance(criterion, list) else [MODELS.build(criterion)]
        self.num_classes = num_classes
        self.gt_sem_seg_key = gt_sem_seg_key
        self.use_half = use_half
        self.binary_segment_threshold = binary_segment_threshold
        # Build and store inference configuration
        if inference_config is None:
            self.inference_config = InferenceConfig()
        elif isinstance(inference_config, InferenceConfig):
            self.inference_config = inference_config
        elif isinstance(inference_config, dict):
            # Accept both new and legacy key names for compatibility
            self.inference_config = InferenceConfig(**inference_config)
        else:
            raise TypeError(f'inference_config must be InferenceConfig, dict or None, but got {type(inference_config)}')
        self.allow_pbar = allow_pbar

        if use_half:
            self.half()

    def forward(self,
                inputs: Tensor,
                data_samples:Sequence[BaseDataElement]|None=None,
                mode:str='tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]) -> dict:
        ...

    @abstractmethod
    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        ...

    @abstractmethod
    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        ...


class mgam_Seg2D_Lite(mgam_Seg_Lite):
    def loss(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, H, W)
            data_samples (Sequence[BaseDataElement]): The seg data samples

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # Forward pass to get prediction logits
        seg_logits = self._forward(inputs, data_samples)

        # Extract ground truth masks from data_samples
        gt_segs = []
        for data_sample in data_samples:
            gt_segs.append(data_sample.get(self.gt_sem_seg_key).data)
        gt_segs = torch.stack(gt_segs, dim=0).squeeze(1)  # [N, H, W]

        return {'loss_' + cri.__class__.__name__: cri(seg_logits, gt_segs)
                for cri in self.criterion}

    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        """Predict results from a batch of inputs and data samples.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, H, W).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.
                It usually includes information such as `metainfo`.

        Returns:
            Sequence[BaseDataElement]: Segmentation results of the input images.
                Each SegDataSample usually contains:
                - pred_sem_seg (PixelData): Prediction of semantic segmentation.
                - seg_logits (PixelData): Predicted logits of semantic segmentation.
        """
        # Forward pass
        seg_logits = self.inference(inputs, data_samples) # [N, C, H, W]

        # Process outputs
        batch_size = inputs.shape[0]
        out_channels = seg_logits.shape[1]

        # Validate consistency of binary threshold and output channels
        if out_channels > 1 and self.binary_segment_threshold is not None:
            raise ValueError(
                f"Multi-class model (out_channels={out_channels}) should not set binary_segment_threshold; "
                f"current value is {self.binary_segment_threshold}, expected None"
            )
        if out_channels == 1 and self.binary_segment_threshold is None:
            raise ValueError(f"Binary model (out_channels={out_channels}) must set binary_segment_threshold; current value is None")

        if data_samples is None:
            data_samples = [BaseDataElement() for _ in range(batch_size)]

        for i in range(batch_size):
            # Process each sample
            i_seg_logits = seg_logits[i] # [C, H, W]

            # Generate prediction mask
            if out_channels > 1:  # 多分类情况
                # Argmax on forward_device with optional chunking
                argmax_proc = ArgmaxProcessor(self.inference_config)
                # We want keepdim=True to maintain [1, H, W]
                i_seg_pred = argmax_proc.argmax(i_seg_logits, dim=0, keepdim=True)
            else:  # 二分类情况
                assert self.binary_segment_threshold is not None, \
                    f"二分类模型(输出通道数={out_channels})必须设置binary_segment_threshold，" \
                    f"当前值为None"
                i_seg_logits_sigmoid = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits_sigmoid > self.binary_segment_threshold).to(i_seg_logits)

            # Store results into data_samples
            data_samples[i].seg_logits = PixelData(data=i_seg_logits)
            data_samples[i].pred_sem_seg = PixelData(data=i_seg_pred)

        return data_samples

    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, H, W).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.

        Returns:
            Tensor: Output tensor from backbone
        """
        return self.backbone(inputs)

    @torch.inference_mode()
    def inference(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """Perform inference, supporting sliding-window or full-image.

        Args:
            inputs (Tensor): Input tensor of shape (N, C, H, W).
            data_samples (Sequence[BaseDataElement], optional): Data samples.

        Returns:
            Tensor: Segmentation logits.
        """
        # 检查是否需要滑动窗口推理
        if self.inference_config.patch_size is not None and self.inference_config.patch_stride is not None:
            seg_logits = self.slide_inference(inputs, data_samples)
        else:
            # 整体推理
            seg_logits = self._forward(inputs, data_samples)

        return seg_logits

    def slide_inference(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """Perform sliding-window inference with overlapping patches.

        Args:
            inputs (Tensor): Input tensor of shape (N, C, H, W).
            data_samples (Sequence[BaseDataElement], optional): Data samples.

        Returns:
            Tensor: Segmentation logits.
        """
        # Retrieve sliding-window parameters
        assert self.inference_config.patch_size is not None and self.inference_config.patch_stride is not None, \
            f"Sliding-window inference requires patch_size({self.inference_config.patch_size}) and patch_stride({self.inference_config.patch_stride})"
        # Validate dimensionality for 2D
        if not (isinstance(self.inference_config.patch_size, (tuple, list)) and len(self.inference_config.patch_size) == 2):
            raise AssertionError(f"For 2D inference, patch_size must be a tuple/list of length 2, got {self.inference_config.patch_size}")
        if not (isinstance(self.inference_config.patch_stride, (tuple, list)) and len(self.inference_config.patch_stride) == 2):
            raise AssertionError(f"For 2D inference, patch_stride must be a tuple/list of length 2, got {self.inference_config.patch_stride}")
        h_stride, w_stride = self.inference_config.patch_stride
        h_crop, w_crop = self.inference_config.patch_size
        batch_size, _, h_img, w_img = inputs.size()
        h_img, w_img = int(h_img), int(w_img)

        # Check if padding is needed for small images
        need_padding = h_img < h_crop or w_img < w_crop
        if need_padding:
            # Compute padding sizes
            pad_h = max(h_crop - h_img, 0)
            pad_w = max(w_crop - w_img, 0)
            # padding: (left, right, top, bottom)
            pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            padded_inputs = torch.nn.functional.pad(inputs, pad, mode='replicate', value=0)
            h_padded, w_padded = padded_inputs.shape[2], padded_inputs.shape[3]
        else:
            padded_inputs = inputs
            h_padded, w_padded = h_img, w_img
            pad = None

        # Compute number of grids based on padded size
        h_grids = max(h_padded - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_padded - w_crop + w_stride - 1, 0) // w_stride + 1

        accumulate_device = torch.device(self.inference_config.accumulate_device)

        preds = torch.zeros(
            size=(batch_size, self.num_classes, h_padded, w_padded),
            dtype=torch.float32,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, h_padded, w_padded),
            dtype=torch.uint8,
            device=accumulate_device
        )

        # Sliding-window inference loop
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                h1 = h_idx * h_stride
                w1 = w_idx * w_stride
                h2 = min(h1 + h_crop, h_padded)
                w2 = min(w1 + w_crop, w_padded)
                h1 = max(h2 - h_crop, 0)
                w1 = max(w2 - w_crop, 0)

                # Extract patch
                crop_img = padded_inputs[:, :, h1:h2, w1:w2]

                # Run forward on patch (ensure forward device consistency)
                crop_seg_logit = self._forward(crop_img.to(self.inference_config.forward_device))

                # Move results to accumulation device and sum
                crop_seg_logit_on_device = crop_seg_logit.to(accumulate_device)
                preds[:, :, h1:h2, w1:w2] += crop_seg_logit_on_device
                count_mat[:, :, h1:h2, w1:w2] += 1

        # Verify all regions are covered by sliding windows
        assert torch.min(count_mat).item() > 0, "There are areas not covered by sliding windows"
        # Compute average logits
        seg_logits = preds / count_mat

        # Crop back to original size if padding was applied
        if need_padding:
            assert pad is not None, "Missing padding info, cannot crop back to original size"
            pad_left, pad_right, pad_top, pad_bottom = pad
            seg_logits = seg_logits[:, :, pad_top:h_padded-pad_bottom, pad_left:w_padded-pad_right]

        return seg_logits


class mgam_Seg3D_Lite(mgam_Seg_Lite):
    def loss(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, Z, Y, X)
            data_samples (Sequence[BaseDataElement]): The seg data samples

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # Forward pass to get prediction logits
        seg_logits = self._forward(inputs, data_samples)

        # Extract ground truth volumes from data_samples
        gt_segs = []
        for data_sample in data_samples:
            gt_segs.append(data_sample.get(self.gt_sem_seg_key).data)
        gt_segs = torch.stack(gt_segs, dim=0).squeeze(1)  # [N, Z, Y, X]

        return {'loss_' + cri.__class__.__name__: cri(seg_logits, gt_segs)
                for cri in self.criterion}

    @torch.inference_mode()
    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        """Predict results from a batch of inputs and data samples.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, Z, Y, X).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.
                It usually includes information such as `metainfo`.

        Returns:
            Sequence[BaseDataElement]: Segmentation results of the input images.
                Each SegDataSample usually contains:
                - pred_sem_seg (VolumeData): Prediction of semantic segmentation.
                - seg_logits (VolumeData): Predicted logits of semantic segmentation.
        """

        def _predict(force_cpu:bool=False):
            nonlocal data_samples

            seg_logits = self.inference(inputs, data_samples, force_cpu) # [N, C, Z, Y, X]

            batch_size = inputs.shape[0]
            out_channels = seg_logits.shape[1]

            if out_channels > 1 and self.binary_segment_threshold is not None:
                raise ValueError(
                    f"Multi-class model (out_channels={out_channels}) should not set binary_segment_threshold; "
                    f"current value is {self.binary_segment_threshold}, expected None"
                )
            if out_channels == 1 and self.binary_segment_threshold is None:
                raise ValueError(f"Binary model (out_channels={out_channels}) must set binary_segment_threshold; current value is None")

            if data_samples is None:
                data_samples = [BaseDataElement() for _ in range(batch_size)]

            for i in range(batch_size):
                # Process each sample
                i_seg_logits = seg_logits[i] # [C, Z, Y, X]

                # Generate prediction volume
                if out_channels > 1:  # Multi-class segmentation
                    # Argmax on forward_device with optional chunking
                    argmax_proc = ArgmaxProcessor(self.inference_config)
                    i_seg_pred = argmax_proc.argmax(i_seg_logits, dim=0, keepdim=True)
                else:  # Binary segmentation
                    assert self.binary_segment_threshold is not None, \
                        f"Binary segmentation model (out_channels={out_channels}) must set binary_segment_threshold，" \
                        f"currently it's {self.binary_segment_threshold}"
                    i_seg_logits_sigmoid = i_seg_logits.sigmoid()
                    i_seg_pred = (i_seg_logits_sigmoid > self.binary_segment_threshold).to(i_seg_logits)

                # Store results into data_samples
                data_samples[i].seg_logits = VolumeData(**{"data": i_seg_logits})
                data_samples[i].pred_sem_seg = VolumeData(**{"data": i_seg_pred})

            return data_samples

        try:
            return _predict()
        except torch.OutOfMemoryError as e:
            print_log("OOM during slide inference, trying cpu accumulate.", 'current', logging.WARNING)
            return _predict(force_cpu=True)

    @torch.inference_mode()
    def inference(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None, force_cpu:bool=False) -> Tensor:
        """Perform inference, supporting sliding-window or full-volume.

        Args:
            inputs (Tensor): Input tensor of shape (N, C, Z, Y, X).
            data_samples (Sequence[BaseDataElement], optional): Data samples.
            force_cpu (bool): Whether to force accumulation on CPU to avoid GPU OOM. Default is False.

        Returns:
            Tensor: Segmentation logits.
        """
        if self.inference_config.patch_size is not None and self.inference_config.patch_stride is not None:
            seg_logits = self.slide_inference(inputs, data_samples, force_cpu=force_cpu)

        else:
            seg_logits = self._forward(inputs, data_samples)

        return seg_logits

    @torch.inference_mode()
    def slide_inference(self,
                        inputs: Tensor,
                        data_samples: Sequence[BaseDataElement] | None = None,
                        force_cpu: bool = False,
                        forward_batch_windows: int | None = None) -> Tensor:
        """Perform sliding-window inference with overlapping sub-volumes.

        Args:
            inputs (Tensor): Input tensor of shape (N, C, Z, Y, X).
            data_samples (Sequence[BaseDataElement], optional): Data samples.
            force_cpu (bool): Whether to force accumulation on CPU to avoid GPU OOM. Default is False.
            batch_windows (int): Number of sub-volumes to process in a batch. Default is 1.

        Returns:
            Tensor: Segmentation logits.
        """
        # Retrieve sliding-window parameters
        assert self.inference_config.patch_size is not None and self.inference_config.patch_stride is not None, \
            f"When using sliding window, patch_size({self.inference_config.patch_size}) and patch_stride({self.inference_config.patch_stride}) must be set, " \
            f"elsewise, please set both to `None` to disable sliding window."
        z_stride, y_stride, x_stride = self.inference_config.patch_stride
        z_crop, y_crop, x_crop = self.inference_config.patch_size
        batch_windows = forward_batch_windows or self.inference_config.forward_batch_windows
        batch_size, _, z_img, y_img, x_img = inputs.size()
        assert batch_size == 1, "Currently only batch_size=1 is supported for 3D sliding-window inference"

        # Convert sizes to Python ints to avoid tensor-to-bool issues
        z_img = int(z_img)
        y_img = int(y_img)
        x_img = int(x_img)

        # Check if padding is needed for small volumes
        need_padding = z_img < z_crop or y_img < y_crop or x_img < x_crop
        if need_padding:
            # Compute padding sizes
            pad_z = max(z_crop - z_img, 0)
            pad_y = max(y_crop - y_img, 0)
            pad_x = max(x_crop - x_img, 0)
            # Apply symmetric padding: (left, right, top, bottom, front, back)
            pad = (pad_x // 2, pad_x - pad_x // 2,
                   pad_y // 2, pad_y - pad_y // 2,
                   pad_z // 2, pad_z - pad_z // 2)
            padded_inputs = torch.nn.functional.pad(inputs, pad, mode='replicate', value=0)
            z_padded, y_padded, x_padded = padded_inputs.shape[2], padded_inputs.shape[3], padded_inputs.shape[4]
        else:
            padded_inputs = inputs
            z_padded, y_padded, x_padded = z_img, y_img, x_img
            pad = None

        # Prepare accumulation and count tensors on target device
        accumulate_device = torch.device('cpu') if force_cpu else torch.device(self.inference_config.accumulate_device)
        if accumulate_device.type == 'cuda':
            # Clear CUDA cache if using GPU accumulation
            torch.cuda.empty_cache()

        # Create accumulation and count matrices on specified device
        preds = torch.zeros(
            size = (batch_size, self.num_classes, z_padded, y_padded, x_padded),
            dtype = torch.float16,
            device = accumulate_device,
            pin_memory = False
        )
        count_mat = torch.zeros(
            size = (batch_size, 1, z_padded, y_padded, x_padded),
            dtype = torch.uint8,
            device = accumulate_device,
            pin_memory = False
        )
        patch_cache = torch.empty(
            size = (batch_windows, self.num_classes, z_crop, y_crop, x_crop),
            dtype = torch.float16,
            device = accumulate_device,
            pin_memory = True if accumulate_device.type == 'cpu' else False
        )

        # calculate window slices
        window_slices = []
        z_grids = max(z_padded - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_padded - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_padded - x_crop + x_stride - 1, 0) // x_stride + 1
        for z_idx in range(z_grids):
            for y_idx in range(y_grids):
                for x_idx in range(x_grids):
                    z1 = z_idx * z_stride
                    y1 = y_idx * y_stride
                    x1 = x_idx * x_stride
                    z2 = min(z1 + z_crop, z_padded)
                    y2 = min(y1 + y_crop, y_padded)
                    x2 = min(x1 + x_crop, x_padded)
                    z1 = max(z2 - z_crop, 0)
                    y1 = max(y2 - y_crop, 0)
                    x1 = max(x2 - x_crop, 0)
                    window_slices.append((slice(z1, z2), slice(y1, y2), slice(x1, x2)))

        def _device_to_host_pinned_tensor(device_tensor: Tensor, non_blocking: bool = False) -> Tensor:
            """Inplace ops on pinned tensor for efficient transfer."""
            nonlocal patch_cache, preds
            device_tensor = device_tensor.to(preds.dtype) # NOTE Inconsistent dtype can SEVERLY impact tranfer speed.
            if device_tensor.shape == patch_cache.shape:
                # If the shape matches, copy directly to patch_cache
                patch_cache.copy_(device_tensor, non_blocking)
            else:
                # Otherwise, resize patch_cache to match the device tensor shape
                patch_cache.resize_(device_tensor.shape)
                patch_cache.copy_(device_tensor, non_blocking)
            return patch_cache

        # sliding window forward
        for i in tqdm(range(0, len(window_slices), batch_windows),
                      desc="Slide Win. Infer.",
                      disable=not (is_main_process() and self.allow_pbar),
                      dynamic_ncols=True,
                      leave=False):
            batch_slices = window_slices[i:i+batch_windows]

            # prepare inference batch
            batch_patches = []
            for (z_slice, y_slice, x_slice) in batch_slices:
                batch_patches.append(padded_inputs[:, :, z_slice, y_slice, x_slice])
            batch_patches = torch.cat(batch_patches, dim=0).to(self.inference_config.forward_device)  # [B, C, z_crop, y_crop, x_crop]

            # prevent crop_logits of previous patch inference from being overlapped by next patch copy
            # HACK NOT SURE IF THIS STILL HAPPEN, This is only observed when using `.copy(non_blocking=True)`.
            if torch.cuda.is_available() and torch.device(self.inference_config.forward_device).type == "cuda":
                torch.cuda.synchronize()
            patch_logits_on_device = self._forward(batch_patches)
            patch_cache = _device_to_host_pinned_tensor(patch_logits_on_device)

            # accumulate results
            for j, (z_slice, y_slice, x_slice) in enumerate(batch_slices):
                preds[:, :, z_slice, y_slice, x_slice] += patch_cache[j:j+1]
                count_mat[:, :, z_slice, y_slice, x_slice] += 1

        min_count = torch.min(count_mat)
        assert min_count.item() > 0, "There are areas not covered by sliding windows"
        seg_logits = (preds / count_mat).to(dtype=torch.float16)

        if need_padding:
            assert pad is not None, "Missing padding info, cannot crop back to original size"
            pad_x_left, pad_x_right, pad_y_top, pad_y_bottom, pad_z_front, pad_z_back = pad
            seg_logits = seg_logits[:, :,
                                   pad_z_front:z_padded-pad_z_back,
                                   pad_y_top:y_padded-pad_y_bottom,
                                   pad_x_left:x_padded-pad_x_right]

        return seg_logits

    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, Z, Y, X).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.

        Returns:
            Tensor: Output tensor from backbone
        """
        return self.backbone(inputs)
