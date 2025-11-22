import pdb
from collections.abc import Sequence
from typing import Literal
from warnings import deprecated

import cv2
import numpy as np
import SimpleITK as sitk
from mmcv.transforms import BaseTransform

from itkit.io.sitk_toolkit import sitk_resample_to_size, sitk_resample_to_spacing

"""
General Rule:
Before entering the neural network,
the channel dimension order should align with
[Z,Y,X] or [D,H,W]
"""


class BaseLoadBiomedicalData(BaseTransform):
    def _label_remap(self, mask:np.ndarray, label_map:dict):
        mask_copy = mask.copy()
        for old_id, new_id in label_map.items():
            mask[mask_copy == old_id] = new_id
        return mask


class LoadImgFromOpenCV(BaseLoadBiomedicalData):
    """
    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    """

    def transform(self, results: dict) -> dict:
        img_path = results["img_path"]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        results["img"] = img
        results["img_shape"] = img.shape[-2:]
        results["ori_shape"] = img.shape[-2:]
        return results


class LoadAnnoFromOpenCV(BaseLoadBiomedicalData):
    """
    Required Keys:

    - seg_map_path

    Modified Keys:

    - gt_seg_map
    - seg_fields
    """

    def transform(self, results: dict) -> dict:
        if "seg_map_path" in results:
            mask_path = results["seg_map_path"]
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"File not found: {mask_path}")
            if results.get("label_map", None) is not None:
                mask = self._label_remap(mask, results["label_map"])

            results["gt_seg_map"] = mask
            results["seg_fields"].append("gt_seg_map")
        return results


class LoadFromMHA(BaseLoadBiomedicalData):
    def __init__(self, resample_spacing=None, resample_size=None, debug:bool=False):
        self.resample_spacing = resample_spacing
        self.resample_size = resample_size
        self.debug = debug

    def _process_mha(self, mha, field:Literal["image", "label"]):
        if self.resample_spacing is not None:
            mha = sitk_resample_to_spacing(mha, self.resample_spacing, field, interp_method=sitk.sitkLinear if field == "image" else None)
        if self.resample_size is not None:
            mha = sitk_resample_to_size(mha, self.resample_size, field, interp_method=sitk.sitkLinear if field == "image" else None)
        # mha.GetSize(): [X, Y, Z]
        if not isinstance(mha, sitk.Image):
            raise TypeError(f"Error during `sitk_resample`: {mha}")
        mha_array = sitk.GetArrayFromImage(mha)  # [Z, Y, X]
        return mha_array


class LoadImageFromMHA(LoadFromMHA):
    """
    Required Keys:

    - img_path

    Modified Keys:

    - img
    - sitk_image
    """

    def transform(self, results):
        img_path = results["img_path"]
        img_mha = sitk.ReadImage(img_path)
        img_mha = sitk.DICOMOrient(img_mha, "LPI")
        img = self._process_mha(img_mha, "image")

        results["img"] = img  # output: [Z, Y, X]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        if self.debug:
            print(f"[LoadImageMHA] `{img_path}` shape: {img.shape}")
        return results


class LoadMaskFromMHA(LoadFromMHA):
    """
    Required Keys:

    - label_path
    - sitk_image

    Modified Keys:

    - gt_seg_map
    """

    def transform(self, results):
        if "seg_map_path" in results:
            mask_path = results["seg_map_path"]
            mask_mha = sitk.ReadImage(mask_path)
            mask_mha = sitk.DICOMOrient(mask_mha, "LPI")
            mask = self._process_mha(mask_mha, "label")
            if results.get("label_map", None) is not None:
                mask = self._label_remap(mask, results["label_map"])

            results["gt_seg_map"] = mask # output: [Z, Y, X]
            results["seg_fields"].append("gt_seg_map")
            if self.debug:
                print(f"[LoadMaskMHA] `{mask_path}` shape: {mask.shape}")

        return results
