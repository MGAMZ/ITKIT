import numpy as np
from .base import BaseTransform


class WindowNorm(BaseTransform):
    def __init__(self, window_level: int = 0, window_width: int = 300):
        super().__init__()
        self.window_level = window_level
        self.window_width = window_width
    
    def __call__(self, sample: dict) -> dict:
        if 'image' in sample:
            image = sample['image']
            image = np.clip(image, self.window_level - self.window_width // 2, self.window_level + self.window_width // 2)
            image = (image - (self.window_level - self.window_width // 2)) / self.window_width
            sample['image'] = image
        return sample
