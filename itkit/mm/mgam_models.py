import pdb
import logging
import copy
from abc import abstractmethod
from tqdm import tqdm
from typing_extensions import Sequence

import torch
from torch import Tensor

from mmengine.logging import print_log
from mmengine.registry import MODELS
from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement, PixelData
from mmengine.model import BaseModel
from mmengine.dist import is_main_process

from .mmseg_Dev3D import VolumeData


class mgam_Seg_Lite(BaseModel):
    def __init__(self,
                 backbone: ConfigDict,
                 criterion: ConfigDict|list[ConfigDict],
                 num_classes: int,
                 gt_sem_seg_key: str='gt_sem_seg',
                 use_half: bool=False,
                 binary_segment_threshold: float|None=None,
                 inference_PatchSize: tuple|None=None,
                 inference_PatchStride: tuple|None=None,
                 inference_PatchAccumulateDevice: str='cuda',
                 inference_ForwardDevice: str='cuda',
                 allow_pbar: bool=False,
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
            inference_PatchSize (tuple | None): Size of the sliding window for inference. If None, sliding window inference is not used, default is None.
            inference_PatchStride (tuple | None): Stride of the sliding window for inference. If None, sliding window inference is not used, default is None.
            inference_PatchAccumulateDevice (str): Device to accumulate sliding window inference results, either 'cpu' or 'cuda'. For large images, 'cpu' can help avoid GPU OOM. Default is 'cuda'.
        """
        super().__init__(*args, **kwargs)
        self.backbone = MODELS.build(backbone)
        self.criterion = [MODELS.build(c) for c in criterion] if isinstance(criterion, list) else [MODELS.build(criterion)]
        self.num_classes = num_classes
        self.gt_sem_seg_key = gt_sem_seg_key
        self.use_half = use_half
        self.binary_segment_threshold = binary_segment_threshold
        self.inference_PatchSize = inference_PatchSize
        self.inference_PatchStride = inference_PatchStride
        self.inference_PatchAccumulateDevice = inference_PatchAccumulateDevice
        self.inference_ForwardDevice = inference_ForwardDevice
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
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
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
        if self.inference_PatchSize is not None and self.inference_PatchStride is not None:
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
        assert self.inference_PatchSize is not None and self.inference_PatchStride is not None, \
            f"Sliding-window inference requires inference_PatchSize({self.inference_PatchSize}) and inference_PatchStride({self.inference_PatchStride})"
        h_stride, w_stride = self.inference_PatchStride
        h_crop, w_crop = self.inference_PatchSize
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

        accumulate_device = torch.device(self.inference_PatchAccumulateDevice)

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

                # Run forward on patch
                crop_seg_logit = self._forward(crop_img)

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
                    i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
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

    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, Z, Y, X).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.
            
        Returns:
            Tensor: Output tensor from backbone
        """
        return self.backbone(inputs)

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
        if self.inference_PatchSize is not None and self.inference_PatchStride is not None:
            seg_logits = self.slide_inference(inputs, data_samples, force_cpu=force_cpu)

        else:
            seg_logits = self._forward(inputs, data_samples)
            
        return seg_logits

    def slide_inference(self,
                        inputs: Tensor,
                        data_samples: Sequence[BaseDataElement]|None=None,
                        force_cpu: bool=False,
                        batch_windows: int=1) -> Tensor:
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
        assert self.inference_PatchSize is not None and self.inference_PatchStride is not None, \
            f"When using sliding window, inference_PatchSize({self.inference_PatchSize}) and inference_PatchStride({self.inference_PatchStride}) must be set," \
            f"elsewise, please set both to `None` to disable sliding window."
        z_stride, y_stride, x_stride = self.inference_PatchStride
        z_crop, y_crop, x_crop = self.inference_PatchSize
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
        accumulate_device = torch.device('cpu') if force_cpu else torch.device(self.inference_PatchAccumulateDevice)
        if accumulate_device.type == 'cuda':
            # Clear CUDA cache if using GPU accumulation
            torch.cuda.empty_cache()

        # Create accumulation and count matrices on specified device
        preds = torch.zeros(
            size=(batch_size, self.num_classes, z_padded, y_padded, x_padded),
            dtype=torch.float16,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, z_padded, y_padded, x_padded),
            dtype=torch.uint8,
            device=accumulate_device
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
                    
        for i in tqdm(range(0, len(window_slices), batch_windows),
                      desc="Slide Win. Infer.",
                      disable=not (is_main_process() and self.allow_pbar),
                      dynamic_ncols=True,
                      leave=False):
            batch_slices = window_slices[i:i+batch_windows]
            
            # prepare inference batch
            batch_patches = []
            for (z_slice, y_slice, x_slice) in batch_slices:
                crop_img = padded_inputs[:, :, z_slice, y_slice, x_slice]
                batch_patches.append(crop_img.to(self.inference_ForwardDevice, non_blocking=True))
            batch_patches = torch.cat(batch_patches, dim=0)  # [B, C, z_crop, y_crop, x_crop]
            
            # run forward
            batch_seg_logits = self._forward(batch_patches)

            # accumulate results
            for j, (z_slice, y_slice, x_slice) in enumerate(batch_slices):
                preds[:, :, z_slice, y_slice, x_slice] += batch_seg_logits[j:j+1].to(accumulate_device, non_blocking=True)
                count_mat[:, :, z_slice, y_slice, x_slice] += 1

        # 使用tensor操作进行断言检查，避免tensor到boolean转换
        min_count = torch.min(count_mat)
        assert min_count.item() > 0, "There are areas not covered by sliding windows"
        # 计算平均值
        seg_logits = (preds / count_mat).to(dtype=torch.float16)
        
        # Crop back to original size if padding was applied
        if need_padding:
            assert pad is not None, "Missing padding info, cannot crop back to original size"
            pad_x_left, pad_x_right, pad_y_top, pad_y_bottom, pad_z_front, pad_z_back = pad
            seg_logits = seg_logits[:, :, 
                                   pad_z_front:z_padded-pad_z_back,
                                   pad_y_top:y_padded-pad_y_bottom,
                                   pad_x_left:x_padded-pad_x_right]
        
        return seg_logits


class MomentumAvgModel(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 100,
                 interval: int = 1,
                 device: torch.device|None = None,
                 update_buffers: bool = False) -> None:
        super().__init__()
        
        # Check distributed environment
        self.is_distributed = hasattr(model, 'module')
        self.is_deepspeed = hasattr(model, 'module') and hasattr(model.module, 'deepspeed')
        
        # For DeepSpeed, get the full underlying model parameters
        if self.is_deepspeed:
            with model.module.summon_full_params():
                self.module = copy.deepcopy(model.module).requires_grad_(False)
        else:
            target_model = model.module if self.is_distributed else model
            self.module = copy.deepcopy(target_model).requires_grad_(False)
            
        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)
            
        self.register_buffer('steps', torch.tensor(0, dtype=torch.long, device=device))
                           
        self.update_buffers = update_buffers
        if update_buffers:
            state_dict = self.module.state_dict()
            self.avg_parameters = {
                k: v for k, v in state_dict.items() 
                if v.numel() > 0
            }
        else:
            params = dict(self.module.named_parameters())
            self.avg_parameters = {k: v for k, v in params.items() 
                                   if v.numel() > 0}
            
        # Validate momentum parameter range
        assert 0.0 < momentum < 1.0, f'momentum must be in range (0.0, 1.0) but got {momentum}'
        if momentum > 0.5:
            print_log('The value of momentum in EMA is usually a small number,'
                      'which is different from the conventional notion of '
                      f'momentum but got {momentum}. Please make sure the '
                      f'value is correct.',
                      logger='current', 
                      level=logging.WARNING)
        self.momentum = momentum
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def _get_current_param(self):
        if self.update_buffers:
            return self.module.state_dict()
        else:
            return dict(self.module.named_parameters())
    
    def update_parameters(self, model: torch.nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:  # type: ignore
            for k, p_avg in self.avg_parameters.items():
                # NOTE handle deepspeed model shred issue, p_avg may be empty here.
                if p_avg.dtype.is_floating_point and p_avg.shape==src_parameters[k].data.shape:
                    device = p_avg.device
                    self.avg_func(p_avg.data,
                                  src_parameters[k].data.to(device),
                                  self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1  # type: ignore

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum,
                       self.gamma / (self.gamma + self.steps.item()))
        averaged_param.lerp_(source_param, momentum)
