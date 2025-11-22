import pdb
from abc import abstractmethod

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from torch import Tensor
from tqdm import tqdm

from ..mm.mgam_models import mgam_Seg3D_Lite


class Inferencer:
    def __init__(self, cfg_path, ckpt_path, fp16:bool=False, allow_tqdm:bool=True):
        self.fp16 = fp16
        self.allow_tqdm = allow_tqdm
        self.cfg = Config.fromfile(cfg_path)
        self.model: torch.nn.Module = MODELS.build(self.cfg.model)
        load_checkpoint(self.model, ckpt_path, map_location='cpu')
        self.model.eval()
        self.model.cuda()
        self.model.requires_grad_(False)

    @abstractmethod
    @torch.inference_mode()
    def Inference_FromNDArray(self, inputs:np.ndarray) -> tuple[Tensor, Tensor]:
        """ Accept ndarray input and perform inference.
        
        Args:
            inputs (np.ndarray): Input image ndarray.
        
        Returns:
            tuple (Tensor, Tensor):
                - seg_logits (Tensor): Segmentation logits tensor with shape (N, C, Z, Y, X).
                - sem_seg_map (Tensor): Segmentation map tensor with shape (N, Z, Y, X).
        """


class Inferencer_Seg3D(Inferencer):
    ARGMAX_BATCH_SIZE = 16
    assert torch.cuda.is_available(), "CUDA is required for 3D segmentation inference."
    
    @torch.inference_mode()
    def Inference_FromNDArray(self, inputs:np.ndarray) -> tuple[Tensor, Tensor]:
        self.model: mgam_Seg3D_Lite
        assert inputs.ndim == 3, f"Input image must be (Z, Y, X), got: {inputs.shape}."

        inputs = inputs.astype(np.float16 if self.fp16 else np.float32)
        tensor_input = torch.from_numpy(inputs[None, None])
        torch.cuda.empty_cache()

        with torch.autocast('cuda'):
            seg_logits = self.model.slide_inference(tensor_input) # [N,C,Z,Y,X]
        sem_seg_map = self._batched_argmax(seg_logits) # [N,C,Z,Y,X] -> [N,Z,Y,X]

        return seg_logits, sem_seg_map

    def _batched_argmax(self, inputs:Tensor) -> Tensor:
        assert inputs.ndim == 5, f"Input tensor must be (N, C, Z, Y, X), got: {format(inputs.shape)}."
        N, C, Z, Y, X = inputs.shape
        
        sem_seg_map = torch.empty((N, Z, Y, X), dtype=torch.uint8)
        for start_z in tqdm(range(0, Z, Inferencer_Seg3D.ARGMAX_BATCH_SIZE),
                            desc="Batched ArgMax",
                            dynamic_ncols=True,
                            leave=False,
                            mininterval=1,
                            disable=not self.allow_tqdm):
            end_z = min(start_z + Inferencer_Seg3D.ARGMAX_BATCH_SIZE, Z)
            batch_sem_seg_map = inputs[:, :, start_z:end_z].cuda().argmax(dim=1).to(torch.uint8)
            sem_seg_map[:, start_z:end_z].copy_(batch_sem_seg_map, non_blocking=True)

        return sem_seg_map # [N, Z, Y, X]
