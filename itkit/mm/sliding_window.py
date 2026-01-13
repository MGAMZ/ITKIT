from collections.abc import Callable

import torch
import torch.nn.functional as F
from mmengine.dist import is_main_process
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor
from tqdm import tqdm


class InferenceConfig(BaseModel):
    """Configuration for sliding-window and device settings during inference.

    Attributes:
        patch_size (tuple | None): Sliding window size. None disables sliding window.
        patch_stride (tuple | None): Sliding window stride. None disables sliding window.
        accumulate_device (str): Device for accumulating window results, e.g., 'cpu' or 'cuda'.
        forward_device (str): Device to run the forward pass for each window.
        forward_batch_windows (int): Number of sub-volumes to process in a batch.
        argmax_batchsize (int | None): Chunk size for argmax when devices differ.
    """
    patch_size: tuple[int, ...] | None = None
    patch_stride: tuple[int, ...] | None = None
    accumulate_device: str = 'cuda'
    forward_device: str = 'cuda'
    forward_batch_windows: int = 1
    # When accumulate and forward devices differ, a chunk size along the last
    # dimension must be provided to avoid OOM during argmax transfer.
    argmax_batchsize: int | None = None

    model_config = ConfigDict(extra='ignore')

    @field_validator('patch_size', 'patch_stride', mode='before')
    @classmethod
    def _coerce_tuple(cls, v):
        if v is None:
            return None
        if isinstance(v, tuple):
            return v
        if isinstance(v, list):
            return tuple(int(x) for x in v)
        raise ValueError(f"Invalid type for 'patch_size'/'patch_stride': expected tuple, list, or None, got {type(v)}.")


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


@torch.inference_mode()
def slide_inference_3d(
    inputs: Tensor,
    num_classes: int,
    inference_config: InferenceConfig,
    forward_func: Callable[[Tensor], Tensor],
    allow_tqdm: bool = True,
    force_cpu: bool = False,
    progress_desc: str = "Slide Win. Infer."
) -> Tensor:
    """Perform sliding-window inference with overlapping sub-volumes.

    Args:
        inputs (Tensor): Input tensor of shape (N, C, Z, Y, X).
        num_classes (int): Number of output classes.
        inference_config (InferenceConfig): Inference configuration.
        forward_func (Callable): Function to perform forward pass on a batch of patches.
        allow_tqdm (bool): Whether to show progress bars.
        force_cpu (bool): Whether to force accumulation on CPU to avoid GPU OOM.
        progress_desc (str): Description for progress bar.

    Returns:
        Tensor: Segmentation logits.
    """
    # Retrieve sliding-window parameters
    assert inference_config.patch_size is not None and inference_config.patch_stride is not None, \
        f"When using sliding window, patch_size({inference_config.patch_size}) and patch_stride({inference_config.patch_stride}) must be set."

    z_stride, y_stride, x_stride = inference_config.patch_stride
    z_crop, y_crop, x_crop = inference_config.patch_size
    batch_windows = inference_config.forward_batch_windows
    batch_size, _, z_img, y_img, x_img = inputs.size()
    assert batch_size == 1, "Currently only batch_size=1 is supported for 3D sliding-window inference"

    # Convert sizes to Python ints
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
        padded_inputs = F.pad(inputs, pad, mode='replicate')
        z_padded, y_padded, x_padded = padded_inputs.shape[2], padded_inputs.shape[3], padded_inputs.shape[4]
    else:
        padded_inputs = inputs
        z_padded, y_padded, x_padded = z_img, y_img, x_img
        pad = None

    # Prepare accumulation and count tensors on target device
    accumulate_device = torch.device('cpu') if force_cpu else torch.device(inference_config.accumulate_device)
    if accumulate_device.type == 'cuda':
        torch.cuda.empty_cache()

    # Create accumulation and count matrices on specified device
    preds = torch.zeros(
        size=(batch_size, num_classes, z_padded, y_padded, x_padded),
        dtype=torch.float16,
        device=accumulate_device,
        pin_memory=False
    )
    count_mat = torch.zeros(
        size=(batch_size, 1, z_padded, y_padded, x_padded),
        dtype=torch.uint8,
        device=accumulate_device,
        pin_memory=False
    )
    patch_cache = torch.empty(
        size=(batch_windows, num_classes, z_crop, y_crop, x_crop),
        dtype=torch.float16,
        device=accumulate_device,
        pin_memory=True if accumulate_device.type == 'cpu' else False
    )

    # Calculate window slices
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
        nonlocal patch_cache
        device_tensor = device_tensor.to(preds.dtype)
        if device_tensor.shape == patch_cache.shape:
            patch_cache.copy_(device_tensor, non_blocking)
        else:
            patch_cache.resize_(device_tensor.shape)
            patch_cache.copy_(device_tensor, non_blocking)
        return patch_cache

    # Sliding window forward
    for i in tqdm(range(0, len(window_slices), batch_windows),
                  desc=progress_desc,
                  disable=not (is_main_process() and allow_tqdm),
                  dynamic_ncols=True,
                  leave=False):
        batch_slices = window_slices[i:i+batch_windows]

        # Prepare inference batch
        batch_patches = []
        for (z_slice, y_slice, x_slice) in batch_slices:
            batch_patches.append(padded_inputs[:, :, z_slice, y_slice, x_slice])

        # Move batch to forward device
        batch_patches = torch.cat(batch_patches, dim=0).to(inference_config.forward_device)

        # CUDA synchronization to prevent race condition
        if torch.device(inference_config.forward_device).type == "cuda":
            torch.cuda.synchronize()

        # Forward pass
        patch_logits_on_device = forward_func(batch_patches)
        patch_cache = _device_to_host_pinned_tensor(patch_logits_on_device)

        # Accumulate results
        for j, (z_slice, y_slice, x_slice) in enumerate(batch_slices):
            preds[:, :, z_slice, y_slice, x_slice] += patch_cache[j:j+1]
            count_mat[:, :, z_slice, y_slice, x_slice] += 1

    min_count = torch.min(count_mat)
    assert min_count.item() > 0, "There are areas not covered by sliding windows"

    # Average and convert type
    seg_logits = (preds / count_mat).to(dtype=torch.float16)

    if need_padding:
        assert pad is not None, "Missing padding info, cannot crop back to original size"
        pad_x_left, pad_x_right, pad_y_top, pad_y_bottom, pad_z_front, pad_z_back = pad
        seg_logits = seg_logits[:, :,
                               pad_z_front:z_padded-pad_z_back,
                               pad_y_top:y_padded-pad_y_bottom,
                               pad_x_left:x_padded-pad_x_right]

    return seg_logits
