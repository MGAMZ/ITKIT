import pdb, gc
import torch
import numpy as np
from .base import BaseTransform


class TypeConvert(BaseTransform):
    def __init__(self, key:str|list[str], dtype):
        self.key = key if isinstance(key, list) else [key]
        self.dtype = dtype
    
    def __call__(self, sample:dict) -> dict:
        for k in self.key:
            sample[k] = sample[k].astype(self.dtype)
        return sample


class ToOneHot(BaseTransform):
    def __init__(self, key:str|list[str], num_classes:int):
        self.key = key if isinstance(key, list) else [key]
        self.num_classes = num_classes
    
    def __call__(self, sample:dict) -> dict:
        for k in self.key:
            assert sample[k].shape[0] == 1, "ToOneHot transform expects the first dimension to be the channel dimension."
            one_hot = np.eye(self.num_classes)[sample[k][0]]
            sample[k] = np.moveaxis(one_hot, -1, 0)
        return sample


class ToTensor(BaseTransform):
    def __init__(self, key:str|list[str]):
        self.key = key if isinstance(key, list) else [key]
    
    def __call__(self, sample:dict) -> dict:
        for k in self.key:
            sample[k] = torch.from_numpy(sample[k])
        return sample


class GCCollect(BaseTransform):
    """强制GC回收"""
    def __call__(self, sample:dict) -> dict:
        gc.collect()
        return sample
