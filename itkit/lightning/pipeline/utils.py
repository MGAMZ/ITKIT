import pdb, gc
import torch
import numpy as np
from .base import BaseTransform


class TypeConvert(BaseTransform):
    def __init__(self, key: str | list[str], dtype):
        self.key = key if isinstance(key, list) else [key]
        self.dtype = dtype

    def __call__(self, sample: dict) -> dict:
        for k in self.key:
            sample[k] = sample[k].astype(self.dtype)
        return sample


class ToOneHot(BaseTransform):
    def __init__(self, input_key: str, output_key: str, num_classes: int):
        self.input_key = input_key
        self.output_key = output_key
        self.num_classes = num_classes
        assert num_classes <= 255, (
            "ToOneHot transform currently supports up to 255 classes, "
            "otherwise, please be careful with the dtype of `np.eye`."
        )

    def __call__(self, sample: dict) -> dict:
        v: np.ndarray = sample[self.input_key]
        if v.ndim == 4:
            assert (
                v.shape[0] == 1
            ), "ToOneHot transform expects the first dimension to be the channel dimension."
            v = v[0]  # remove channel dim
        else:
            assert (
                v.ndim == 3
            ), f"ToOneHot transform expects input to be 3D or 4D array, got shape {v.shape}."

        one_hot = np.eye(self.num_classes, dtype=np.uint8)[v]
        sample[self.output_key] = np.moveaxis(one_hot, -1, 0)
        return sample


class ToTensor(BaseTransform):
    def __init__(self, key: str | list[str]):
        self.key = key if isinstance(key, list) else [key]

    def __call__(self, sample: dict) -> dict:
        for k in self.key:
            sample[k] = torch.from_numpy(sample[k])
        return sample


class GCCollect(BaseTransform):
    """强制GC回收"""

    def __call__(self, sample: dict) -> dict:
        gc.collect()
        return sample
