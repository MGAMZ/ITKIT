import numpy as np
import SimpleITK as sitk
from .base import BaseTransform
from itkit.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size


class LoadMHAFile(BaseTransform):
    def __init__(self,
                 size: tuple = (None, None, None),
                 spacing: tuple = (None, None, None),):
        super().__init__()
        self.size = size
        self.spacing = spacing

    def _resample_itk(self, image_itk, **kwargs):
        img_dim = image_itk.GetDimension()

        # 验证每个维度的互斥性
        if len(self.spacing) == img_dim and len(self.size) == img_dim:
            for i in range(img_dim):
                if self.spacing[i] is not None and self.size[i] is not None:
                    raise ValueError(f"Dimension {i} cannot have both spacing and size specified for resampling.")

        # --- 阶段一：Spacing 重采样 ---
        orig_spacing = image_itk.GetSpacing()[::-1]
        effective_spacing = list(orig_spacing)
        needs_spacing_resample = False
        for i in range(img_dim):
            if self.spacing[i] is not None:
                effective_spacing[i] = self.spacing[i]
                needs_spacing_resample = True

        image_after_spacing = image_itk
        if needs_spacing_resample and not np.allclose(effective_spacing, orig_spacing):
            image_after_spacing = sitk_resample_to_spacing(
                image_itk, effective_spacing, **kwargs
            )

        # --- 阶段二：Size 重采样 ---
        current_size = image_after_spacing.GetSize()[::-1]
        effective_size = list(current_size)
        needs_size_resample = False
        for i in range(img_dim):
            if self.size[i] is not None:
                effective_size[i] = self.size[i]
                needs_size_resample = True

        image_resampled = image_after_spacing
        if needs_size_resample and effective_size != list(current_size):
            image_resampled = sitk_resample_to_size(
                image_after_spacing, effective_size, **kwargs
            )
        
        return image_resampled

    def __call__(self, sample: dict) -> dict:
        if 'image_mha_path' in sample:
            image_mha = sitk.ReadImage(sample['image_mha_path'])
            image_mha = self._resample_itk(image_mha, field='image')
            image_mha = sitk.DICOMOrient(image_mha, 'LPI')
            sample['image'] = sitk.GetArrayFromImage(image_mha)[None].astype(np.int16)
        
        if 'label_mha_path' in sample:
            label_mha = sitk.ReadImage(sample['label_mha_path'])
            label_mha = self._resample_itk(label_mha, field='label')
            label_mha = sitk.DICOMOrient(label_mha, 'LPI')
            sample['label'] = sitk.GetArrayFromImage(label_mha).astype(np.uint8)
        
        return sample
