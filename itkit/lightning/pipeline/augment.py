import random
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from ...process.GeneralPreProcess import RandomRotate3D as mm_RandomRotate3D
from .base import BaseTransform


class RandomPatch3D(BaseTransform):
    def __init__(
        self,
        patch_size: Sequence[int],
        keys: list[str] = ["image", "label"]
    ):
        self.patch_size = patch_size
        self.keys = keys

    def __call__(self, sample:dict[str, np.ndarray]):
        img = sample.get("image")
        if img is None:
            raise KeyError("`img` key is required for RandomPatch3D")

        c, z, y, x = img.shape
        pz, py, px = self.patch_size
        if any(dim < p for dim, p in zip((z, y, x), (pz, py, px))):
            raise ValueError(f"Patch size {self.patch_size} exceeds image shape {(z, y, x)}")
        z1 = random.randint(0, z - pz)
        y1 = random.randint(0, y - py)
        x1 = random.randint(0, x - px)

        for key in self.keys:
            sample[key] = sample[key][..., z1:z1+pz, y1:y1+py, x1:x1+px]

        return sample


class RandomPatch3DIndexing(BaseTransform):
    """
    Generates random patch indices but does not slice the data.
    The slicing is intended to be done on the GPU in the training step.
    """
    def __init__(
        self,
        patch_size: Sequence[int],
        num_patches: int,
    ):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, sample:dict[str, np.ndarray]):
        img = sample.get("image")
        if img is None:
            raise KeyError("`image` key is required for RandomPatch3DIndexing")

        # img shape is [C, Z, Y, X]
        c, z, y, x = img.shape
        pz, py, px = self.patch_size
        if any(dim < p for dim, p in zip((z, y, x), (pz, py, px))):
            raise ValueError(f"Patch size {self.patch_size} exceeds image shape {(z, y, x)}")

        indices = []
        for _ in range(self.num_patches):
            z1 = random.randint(0, z - pz)
            y1 = random.randint(0, y - py)
            x1 = random.randint(0, x - px)
            indices.append((z1, y1, x1))

        sample['patch_indices'] = np.array(indices, dtype=np.int32)
        return sample


class RandomRotate3D(mm_RandomRotate3D):
    def __init__(self, keys:list[str], interp_orders:list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys if isinstance(keys, list) else [keys]
        self.interp_orders = interp_orders if isinstance(interp_orders, list) else [interp_orders]

    def __call__(self, sample: dict) -> dict:
        if np.random.rand() < self.prob:
            rot = self._sample_rotation_matrix()
            for key, interp_order in zip(self.keys, self.interp_orders):
                sample[key] = self._rotate_volume(sample[key], rot, interp_order)
        return sample


class AutoPad(BaseTransform):
    def __init__(
        self,
        size: Sequence[int],
        dim: Literal["1d", "2d", "3d"],
        pad_val: int = 0,
        pad_label_val: int = 0,
    ):
        self.dim = dim
        self.dim_map = {"1d": 1, "2d": 2, "3d": 3}
        if len(size) != self.dim_map[dim]:
            raise ValueError(f"Size tuple length {len(size)} does not match dim {dim}")
        self.size = size
        self.pad_val = pad_val
        self.pad_label_val = pad_label_val

    def _get_pad_params(self, current_shape: tuple):
        pad_params = []
        # 只处理最后n个维度，n由dim决定
        dims_to_pad = self.dim_map[self.dim]

        # 确保current_shape维度足够
        if len(current_shape) < dims_to_pad:
            raise ValueError(f"Input shape {current_shape} has fewer dimensions than required {dims_to_pad}")

        # 处理不需要padding的前置维度
        for _ in range(len(current_shape) - dims_to_pad):
            pad_params.append((0, 0))

        # 处理需要padding的维度
        for target_size, curr_size in zip(self.size, current_shape[-dims_to_pad:]):
            if curr_size >= target_size:
                pad_params.append((0, 0))
            else:
                pad = target_size - curr_size
                pad_1 = pad // 2
                pad_2 = pad - pad_1
                pad_params.append((pad_1, pad_2))

        return pad_params

    def __call__(self, sample: dict):
        pad_params = self._get_pad_params(sample["image"].shape)

        if any(p[0] > 0 or p[1] > 0 for p in pad_params):
            sample["image"] = np.pad(
                sample["image"],
                pad_params,
                mode="constant",
                constant_values=self.pad_val,
            )

            if "label" in sample:
                label_do_not_have_channel_dimension = (sample["image"].ndim - sample["label"].ndim) == 1
                sample["label"] = np.pad(
                    sample["label"],
                    pad_params[1:] if label_do_not_have_channel_dimension else pad_params,
                    mode="constant",
                    constant_values=self.pad_label_val,
                )

        return sample


class BatchAugment(BaseTransform):
    """
    NOTE
    The reason to do SampleWiseInTimeAugment is the time comsumption
    for IO of an entire sample is too expensive, so it's better
    to augment the sample in time, thus accquiring multiple trainable sub-samples.
    """
    def __init__(self, num_samples:int, pipeline: list[Callable]|Callable):
        self.num_samples = num_samples
        self.pipeline = pipeline if isinstance(pipeline, list) else [pipeline]

    def get_one_sample(self, sample: dict):
        for t in self.pipeline:
            sample = t(sample)
        return sample

    def __call__(self, sample: dict) -> list[dict]:
        samples = []
        for _ in range(self.num_samples):
            samples.append(self.get_one_sample(sample.copy()))
        return samples
