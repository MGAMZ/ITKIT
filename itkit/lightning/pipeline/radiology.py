import numpy as np
import SimpleITK as sitk

from ...io.sitk_toolkit import sitk_resample_to_size, sitk_resample_to_spacing
from .base import BaseTransform


class WindowNorm(BaseTransform):
    def __init__(self, window_level:int, window_width:int):
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


class ITKResample(BaseTransform):
    def __init__(self,
                 size: tuple,
                 spacing: tuple,
                 key: str = 'image',
                 itk_resample_kwargs: dict = {}
    ):
        super().__init__()
        self.size = size
        self.spacing = spacing
        self.key = key
        self.itk_resample_kwargs = itk_resample_kwargs

    def __call__(self, sample: dict) -> dict:
        if self.key not in sample:
            return sample

        image_itk = sample[self.key]
        img_dim = image_itk.GetDimension()

        # 验证每个维度的互斥性
        if len(self.spacing) == img_dim and len(self.size) == img_dim:
            for i in range(img_dim):
                if self.spacing[i] != -1 and self.size[i] != -1:
                    raise ValueError(f"Dimension {i} cannot have both spacing and size specified for resampling.")

        # --- 阶段一：Spacing 重采样 ---
        orig_spacing = image_itk.GetSpacing()[::-1]
        effective_spacing = list(orig_spacing)
        needs_spacing_resample = False
        if len(self.spacing) == img_dim:
            for i in range(img_dim):
                if self.spacing[i] != -1:
                    effective_spacing[i] = self.spacing[i]
                    needs_spacing_resample = True

        image_after_spacing = image_itk
        if needs_spacing_resample and not np.allclose(effective_spacing, orig_spacing):
            image_after_spacing = sitk_resample_to_spacing(
                image_itk, effective_spacing, **self.itk_resample_kwargs
            )
        if not isinstance(image_after_spacing, sitk.Image):
            raise RuntimeError(f"Resampled image is not a SimpleITK Image, got {image_after_spacing}")

        # --- 阶段二：Size 重采样 ---
        current_size = image_after_spacing.GetSize()[::-1]
        effective_size = list(current_size)
        needs_size_resample = False
        if len(self.size) == img_dim:
            for i in range(img_dim):
                if self.size[i] != -1:
                    effective_size[i] = self.size[i]
                    needs_size_resample = True

        image_resampled = image_after_spacing
        if needs_size_resample and effective_size != list(current_size):
            image_resampled = sitk_resample_to_size(
                image_after_spacing, effective_size, **self.itk_resample_kwargs
            )

        sample[self.key] = image_resampled
        return sample
