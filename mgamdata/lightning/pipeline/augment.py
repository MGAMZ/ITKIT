from collections.abc import Callable, Sequence
import random
import numpy as np
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

