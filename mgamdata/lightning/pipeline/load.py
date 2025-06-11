import numpy as np
import SimpleITK as sitk
from .base import BaseTransform


class LoadMHAFile(BaseTransform):
    def __call__(self, sample: dict) -> dict:
        if 'image_mha_path' in sample:
            image_mha = sitk.ReadImage(sample['image_mha_path'])
            sample['image'] = sitk.GetArrayFromImage(image_mha)[None].astype(np.int16)
        
        if 'label_mha_path' in sample:
            label_mha = sitk.ReadImage(sample['label_mha_path'])
            sample['label'] = sitk.GetArrayFromImage(label_mha)[None].astype(np.uint8)
        
        return sample
