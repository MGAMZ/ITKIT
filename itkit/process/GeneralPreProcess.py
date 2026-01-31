"""
General Rule:
Before entering the neural network,
the channel dimension order should align with
[Z,Y,X] or [D,H,W]
"""

import math
import random
from collections.abc import Sequence
from functools import partial
from numbers import Number
from typing import Any, Literal, cast

import albumentations as A
import cv2
import numpy as np
import torch
from colorama import Fore, Style
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F


def SetWindow(array:np.ndarray|torch.Tensor, window_width:int, window_level:int):
    window_left = window_level - window_width // 2
    window_right = window_level + window_width // 2
    if isinstance(array, np.ndarray):
        array = np.clip(array, window_left, window_right)
    elif isinstance(array, torch.Tensor):
        array = torch.clamp(array, window_left, window_right)
    else:
        raise TypeError(f"Unsupported type {type(array)}. Expected np.ndarray or torch.Tensor.")
    array = (array - window_left) / window_width
    return array # range: [0, 1]


class AutoPad(BaseTransform):
    def __init__(
        self,
        size: tuple[int, ...],
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

    def _get_pad_params(self, current_shape: tuple) -> tuple[tuple[int, int], ...]:
        pad_params = []
        # Only handle the last n dimensions, where n is determined by `dim`
        dims_to_pad = self.dim_map[self.dim]

        # Ensure current_shape has enough dimensions
        if len(current_shape) < dims_to_pad:
            raise ValueError(f"Input shape {current_shape} has fewer dimensions than required {dims_to_pad}")

        # Handle leading dimensions that don't need padding
        for _ in range(len(current_shape) - dims_to_pad):
            pad_params.append((0, 0))

        # Handle dimensions that need padding
        for target_size, curr_size in zip(self.size, current_shape[-dims_to_pad:]):
            if curr_size >= target_size:
                pad_params.append((0, 0))
            else:
                pad = target_size - curr_size
                pad_1 = pad // 2
                pad_2 = pad - pad_1
                pad_params.append((pad_1, pad_2))

        return tuple(pad_params)

    def transform(self, results: dict):
        img = results["img"]
        pad_params = self._get_pad_params(img.shape)

        if any(p[0] > 0 or p[1] > 0 for p in pad_params):
            results["img"] = np.pad(
                img,
                pad_params,
                mode="constant",
                constant_values=self.pad_val,
            )

            for seg_field in results['seg_fields']:
                results[seg_field] = np.pad(
                    results[seg_field],
                    pad_params,
                    mode="constant",
                    constant_values=self.pad_label_val,
                )

        return results


class CropSlice_Foreground(BaseTransform):
    """
    Required Keys:

    - img
    - gt_seg_map_index
    - gt_seg_map_channel

    Modified Keys:

    - img
    - gt_seg_map_index
    - gt_seg_map_channel
    """

    def __init__(self, num_slices: int, ratio: float = 0.9):
        self.num_slices = num_slices
        self.ratio = ratio  # With a certain probability, this operation will be applied

    def _locate_possible_start_slice_with_non_background(self, mask):
        assert (
            mask.ndim == 3
        ), f"Invalid Mask Shape: Expected [D,H,W], but got {mask.shape}"
        # locate non-background slices
        slices_not_pure_background = np.argwhere(np.any(mask, axis=(1, 2)))
        start_slice, end_slice = (
            slices_not_pure_background.min(),
            slices_not_pure_background.max(),
        )
        non_background_slices = np.arange(start_slice, end_slice, dtype=np.uint32)

        # locate the range of possible start slice
        # which ensures the selected slices are not entirely background
        min_possible_start_slice = max(
            0, non_background_slices[0] - self.num_slices + 1
        )
        max_possible_start_slice = max(
            0, min(non_background_slices[-1], mask.shape[0] - self.num_slices)
        )

        return (min_possible_start_slice, max_possible_start_slice)

    def transform(self, results):
        if np.random.rand(1) > self.ratio:
            return results

        mask = results["gt_seg_map_index"]
        min_start_slice, max_start_slice = (
            self._locate_possible_start_slice_with_non_background(mask)
        )
        selected_slices = np.arange(
            min_start_slice, max_start_slice + self.num_slices - 1
        )

        results["img"] = np.take(results["img"], selected_slices, axis=0)
        results["gt_seg_map_channel"] = np.take(
            results["gt_seg_map_channel"], selected_slices, axis=1
        )
        results["gt_seg_map_index"] = np.take(
            results["gt_seg_map_index"], selected_slices, axis=0
        )
        return results


class WindowSet(BaseTransform):
    """
    Required Keys:

    - img

    Modified Keys:

    - img
    """

    def __init__(self, level, width):
        self.clip_range = (level - width // 2, level + width // 2)
        self.level = level
        self.width = width

    def _window_norm(self, img: np.ndarray):
        img = np.clip(img, self.clip_range[0], self.clip_range[1])  # Window Clip
        img = img - self.clip_range[0]  # HU bias to positive
        img = img / self.width  # Zero-One Normalization
        return img.astype(np.float32)

    def transform(self, results):
        results["img"] = self._window_norm(results["img"])
        return results


class TypeConvert(BaseTransform):
    """
    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map
    """
    def __init__(self, key:str|list[str], dtype:str):
        self.key = key if isinstance(key, list) else [key]
        self.dtype = dtype

    def transform(self, results):
        for k in self.key:
            results[k] = results[k].astype(self.dtype)
        return results


class RandomRoll(BaseTransform):
    """
    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map
    """

    def __init__(
        self,
        axis: int | list[int],
        gap: int | list[int],
        erase: bool = False,
        pad_val: int = 0,
        seg_pad_val: int = 0,
    ):
        """
        Perform random roll along specified axes.

        :param axis: Axis or axes to roll along.
        :param gap: Maximum shift range for the corresponding axes.
        :param erase: Whether to erase the rolled-over region.
        :param pad_val: Padding value used when erasing image regions.
        :param seg_pad_val: Padding value used when erasing segmentation regions.
        """
        if isinstance(axis, int):
            axis = [axis]
        if isinstance(gap, int):
            gap = [gap]

        assert len(axis) == len(
            gap
        ), f"axis ({len(axis)}) and gap ({len(gap)}) should have the same length"

        self.axis = axis
        self.gap = gap
        self.erase = erase
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    @staticmethod
    def _roll(results, gap, axis):
        if "img" in results:
            results["img"] = np.roll(results["img"], shift=gap, axis=axis)
        if "gt_seg_map" in results:
            results["gt_seg_map"] = np.roll(results["gt_seg_map"], shift=gap, axis=axis)
        return results

    def _erase_part(self, results, gap, axis):
        slicer = [slice(None)] * results["img"].ndim
        if gap > 0:
            slicer[axis] = slice(0, gap)
        else:
            slicer[axis] = slice(gap, None)

        if "img" in results:
            results["img"][tuple(slicer)] = self.pad_val
        if "gt_seg_map" in results:
            results["gt_seg_map"][tuple(slicer)] = self.seg_pad_val

        return results

    def transform(self, results):
        for axis, max_gap in zip(self.axis, self.gap):
            gap = random.randint(-max_gap, max_gap)
            results = self._roll(results, gap, axis)
            if self.erase:
                results = self._erase_part(results, gap, axis)
        return results


class InstanceNorm(BaseTransform):
    """
    Required Keys:

    - img

    Modified Keys:

    - img
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def transform(self, results):
        ori_dtype = results["img"].dtype
        img = results["img"]
        img = img - img.min()
        img = img / (img.std() + self.eps)
        results["img"] = img.astype(ori_dtype)
        return results


class ExpandOneHot(BaseTransform):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        inplace: bool = False
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.inplace = inplace

    def transform(self, results):
        mask = results["gt_seg_map"]  # [...]
        # NOTE The ignored index is remapped to the last class.
        if self.ignore_index is not None:
            mask[mask == self.ignore_index] = self.num_classes
        # eye: Identity Matrix [num_classes+1, num_classes+1]
        mask_channel = np.eye(self.num_classes + 1)[mask]
        mask_channel = np.moveaxis(mask_channel, -1, 0).astype(np.uint8)

        if self.inplace:
            results["gt_seg_map"] = mask_channel[:-1]
        else:
            results["gt_seg_map_one_hot"] = mask_channel[:-1]  # [num_classes, ...]

        return results


class GaussianBlur(BaseTransform):
    def __init__(
        self,
        field: list[Literal["image", "label"]],
        kernel_size: int,
        sigma: float,
        amplify: float = 1.0,
    ):
        self.field = field if isinstance(field, list) else [field]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amplify = amplify
        self.blur = partial(
            cv2.GaussianBlur,
            ksize=(self.kernel_size, self.kernel_size),
            sigmaX=sigma)

    def transform(self, results: dict):
        if "image" in self.field and "img" in results:
            results["img"] = self.blur(results["img"])
        if "label" in self.field and "gt_seg_map" in results:
            results["gt_seg_map"] = self.blur(results["gt_seg_map"].astype(np.float32)) * self.amplify
        return results


class GaussianBlur3D(BaseTransform):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def transform(self, results: dict):
        results["img"] = gaussian_filter(results["img"], sigma=self.sigma)
        return results


class RandomGaussianBlur3D(BaseTransform):
    def __init__(self, max_sigma: float, prob: float = 1.0):
        self.sigma = max_sigma
        self.prob = prob

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            sigma = np.random.uniform(0, self.sigma)
            results["img"] = gaussian_filter(results["img"], sigma=sigma)
        return results


class RandomFlip(BaseTransform):
    def __init__(self, axis: Literal[0, 1, 2], prob: float = 0.5):
        self.axis = axis
        self.prob = prob

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            if "img" in results:
                results["img"] = np.flip(results["img"], axis=self.axis).copy()
            if "gt_seg_map" in results:
                results["gt_seg_map"] = np.flip(
                    results["gt_seg_map"], axis=self.axis
                ).copy()
        return results


class RandomCrop3D(BaseTransform):
    """Random crop the 3D volume & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int, int]]):  Expected size after cropping
            with the format of (d, h, w). If set to an integer, then cropping
            depth, width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    CROP_RETRY = 32

    def __init__(
        self,
        crop_size: int | tuple[int, int, int],
        cat_max_ratio: float = 1.0,
        std_threshold: float|None = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        if isinstance(crop_size, Sequence):
            assert (
                len(crop_size) == 3
            ), f"The expected crop_size containing 3 integers, but got {crop_size}"
        elif isinstance(crop_size, int):
            crop_size = (crop_size, crop_size, crop_size)
        else:
            raise TypeError(f"Unsupported crop size: {crop_size}")

        assert min(crop_size) > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.std_threshold = std_threshold
        self.ignore_index = ignore_index

    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped volume.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input volume.

            Returns:
                tuple: Coordinates of the cropped volume.
            """

            margin_z = max(img.shape[0] - self.crop_size[0], 0)
            margin_y = max(img.shape[1] - self.crop_size[1], 0)
            margin_x = max(img.shape[2] - self.crop_size[2], 0)
            offset_z = np.random.randint(0, margin_z + 1)
            offset_y = np.random.randint(0, margin_y + 1)
            offset_x = np.random.randint(0, margin_x + 1)
            crop_z1, crop_z2 = offset_z, offset_z + self.crop_size[0]
            crop_y1, crop_y2 = offset_y, offset_y + self.crop_size[1]
            crop_x1, crop_x2 = offset_x, offset_x + self.crop_size[2]

            return crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2

        img = results["img"]
        ann = results["gt_seg_map"]

        ccm_check_ = None
        std_check_ = None

        # crop the volume
        for _ in range(self.CROP_RETRY):
            crop_bbox = generate_crop_bbox(img)

            # crop check: category max ratio
            if self.cat_max_ratio is not None and self.cat_max_ratio < 1.0:
                seg_temp = self.crop(ann, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (len(cnt) <= 1) or ((np.max(cnt) / np.sum(cnt)) > self.cat_max_ratio):
                    ccm_check_ = np.max(cnt) / np.sum(cnt)
                    continue

            # crop check: std threshold
            if self.std_threshold is not None:
                img_temp = self.crop(img, crop_bbox)
                if img_temp.std() < self.std_threshold:
                    std_check_ = img_temp.std()
                    continue

            # when pass all check
            return crop_bbox

        else:
            raise RuntimeError(
                Fore.YELLOW + \
                f"Cannot find a valid crop bbox after {self.CROP_RETRY+1} trials. " + \
                f"Last check result: ccm_check={ccm_check_}, std_check={std_check_}." + \
                Style.RESET_ALL
            )

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input volume.
            crop_bbox (tuple): Coordinates of the cropped volume.

        Returns:
            np.ndarray: The cropped volume.
        """

        crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop volumes, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results["img"]
        crop_bbox = self.crop_bbox(results)

        # crop the volume
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        results["img"] = img
        results["img_shape"] = img.shape[:3]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


class RandomAxis(BaseTransform):
    def __init__(
        self, axis: tuple[Literal[0, 1, 2], Literal[0, 1, 2]], prob: float = 0.5
    ):
        self.axis = axis
        self.prob = prob

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            results["img"] = np.moveaxis(results["img"], self.axis[0], self.axis[1])
            if "gt_seg_map" in results:
                results["gt_seg_map"] = np.moveaxis(
                    results["gt_seg_map"], self.axis[0], self.axis[1]
                )
        return results


class NewAxis(BaseTransform):
    def __init__(self, axis: int, keys:list[str]=['img']):
        self.axis = axis
        self.keys = keys

    def transform(self, results: dict):
        for k in self.keys:
            results[k] = np.expand_dims(results[k], axis=self.axis)
        return results


class RandomContinuousErase(BaseTransform):
    def __init__(
        self,
        max_size: list[int] | int,
        pad_val: float | int,
        seg_pad_val=0,
        prob: float = 0.5,
    ):
        self.max_size = max_size if isinstance(max_size, (Sequence)) else [max_size]
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.prob = prob

    def _random_area(self, image_size: list[int], area_size:list[int]|None=None
                     ) -> tuple[list[int], list[int]]:
        assert len(image_size) == len(self.max_size)
        dim = len(image_size)
        selected_size = [
                np.random.randint(1, i) for i in self.max_size
            ] if area_size is None else area_size

        start_cord = [np.random.randint(0, image_size[i] - selected_size[i])
                      for i in range(dim)]
        end_cord = [start_cord[i] + selected_size[i]
                    for i in range(dim)]
        return start_cord, end_cord

    def _erase_area(self, array: np.ndarray, start_cord: list[int], end_cord: list[int]):
        """Erase the information in the selected area, supports any dim"""
        _area = [slice(start_cord[i], end_cord[i])
                 for i in range(len(start_cord))]
        array[tuple(_area)] = self.pad_val
        return array

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            cord = self._random_area(results["img"].squeeze().shape)
            results["img"] = self._erase_area(results["img"], cord[0], cord[1])
            if "gt_seg_map" in results:
                results["gt_seg_map"] = self._erase_area(results["gt_seg_map"], cord[0], cord[1])
        return results


class RandomAlter(RandomContinuousErase):
    def _alter_area(self,
                    array: np.ndarray,
                    source_area: tuple[list[int], list[int]],
                    target_area: tuple[list[int], list[int]]):
        """Exchange the information between two local area, supports any dim"""
        source_start, source_end = source_area
        target_start, target_end = target_area
        _source_area = [slice(source_start[i], source_end[i])
                        for i in range(len(source_start))]
        _target_area = [slice(target_start[i], target_end[i])
                        for i in range(len(target_start))]

        source = array[tuple(_source_area)]
        target = array[tuple(_target_area)]
        array[tuple(_target_area)] = source
        array[tuple(_source_area)] = target
        return array

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            # The location is always randomly determined,
            # but the size is only determined when source_cord is calculated.
            # Nevertheless, the two area can't alter.
            source_cord = self._random_area(results["img"].squeeze().shape)
            dest_cord = self._random_area(results["img"].squeeze().shape,
                                          area_size=[source_cord[1][i] - source_cord[0][i]
                                                     for i in range(len(self.max_size))])
            results["img"] = self._alter_area(results["img"], source_cord, dest_cord)
            if "gt_seg_map" in results:
                results["gt_seg_map"] = self._alter_area(results["gt_seg_map"], source_cord, dest_cord)
        return results


class RandomDiscreteErase(BaseTransform):
    """
    Args:
        max_ratio (float): The maximum ratio of the erased area.
        keys_pad_vals (Sequence[tuple[str, Number]]): The keys and values to be padded.
        min_ratio (float): The minimum ratio of the erased area.
        prob (float): The probability of performing this transformation.

    Modified Keys:
        Specified by `keys_pad_vals`

    Added Keys:
        ori_img (np.ndarray): The original image before erasing.
        erase_mask (np.ndarray): The mask of the erased area.
    """

    def __init__(
        self,
        max_ratio: float,
        keys_pad_vals: Sequence[tuple[str, Number]],
        min_ratio: float = 0.,
        prob: float = 0.5,
    ):
        assert 0 < max_ratio <= 1
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prob = prob
        self.keys_pad_vals = keys_pad_vals

    def _generate_mask(self, array_shape: tuple, ratio: float) -> np.ndarray:
        total_elements = np.prod(array_shape)
        num_erase = int(total_elements * ratio)
        mask = np.zeros(total_elements, dtype=bool)
        erase_indices = np.random.choice(total_elements, num_erase, replace=False)
        mask[erase_indices] = True
        mask = mask.reshape(array_shape)
        return mask

    def _apply_mask(self, array: np.ndarray, mask: np.ndarray, pad_value) -> np.ndarray:
        if array.ndim > mask.ndim:
            mask = mask[..., None] # channel dim
        array[mask] = pad_value
        return array

    def transform(self, results: dict):
        results["ori_img"] = results["img"].copy()
        results["erase_mask"] = np.zeros_like(results["img"].squeeze())

        if np.random.rand() < self.prob:
            erase_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            img_shape = results["img"].squeeze().shape
            mask = self._generate_mask(img_shape, erase_ratio)

            results["erase_mask"] = mask
            for key, pad_val in self.keys_pad_vals:
                results[key] = self._apply_mask(results[key], mask, pad_val)

        return results


class Identity(BaseTransform):
    def transform(self, results: dict):
        return results


class Resample(BaseTransform):
    def __init__(self, size: list[float], mode: str = "bilinear", field: str = "img"):
        self.size = size
        self.mode = mode
        self.field = field

    def transform(self, results: dict):
        results[self.field] = F.interpolate(
            results[self.field][None, None],
            size=[int(round(dim)) for dim in self.size],
            mode=self.mode,
        ).squeeze()
        return results


class device_to(BaseTransform):
    def __init__(self, key:str|list[str], device:str, non_blocking:bool=False):
        self.key = key if isinstance(key, list) else [key]
        self.device = torch.device(device)
        self.non_blocking = non_blocking

    def transform(self, results: dict):
        for key in self.key:
            d = results[key]
            if isinstance(d, torch.Tensor):
                results[key] = d.to(self.device, non_blocking=self.non_blocking)
            elif isinstance(d, np.ndarray):
                results[key] = torch.from_numpy(d).to(self.device, non_blocking=self.non_blocking)
            else:
                raise ValueError(f"Unsupported type {type(d)}")
        return results


class SampleAugment(BaseTransform):
    """
    NOTE
    The reason to do SampleWiseInTimeAugment is the time comsumption
    for IO of an entire sample is too expensive, so it's better
    to augment the sample in time, thus accquiring multiple trainable sub-samples.
    """
    def __init__(self, num_samples:int, pipeline: list[dict]):
        self.num_samples = num_samples
        self.transforms = [TRANSFORMS.build(t) for t in pipeline]

    def get_one_sample(self, results: dict):
        for t in self.transforms:
            results = t(results)
        return results

    def transform(self, results: dict):
        samples = []
        for _ in range(self.num_samples):
            samples.append(self.get_one_sample(results.copy()))
        return samples


class RandomRotate3D(BaseTransform):
    """Free random 3D rotation"""

    def __init__(self,
                 degree: float,
                 prob: float = 1.0,
                 interp_order: int = 0,
                 pad_val: float = 0,
                 resample_prefilter: bool = False,
                 crop_to_valid_region: bool = False,
                 img_keys: list[str] = ["img"],
    ):
        self.degree = degree
        self.prob = prob
        self.interp_order = interp_order
        self.pad_val = pad_val
        self.resample_prefilter = resample_prefilter
        self.crop_to_valid_region = crop_to_valid_region
        self.img_keys = img_keys
        # Precompute cosine of the maximum rotation angle
        self.cos_theta = np.cos(np.deg2rad(degree))

    def _sample_rotation_matrix(self):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(-self.degree, self.degree)
        return R.from_rotvec(np.deg2rad(angle) * axis).as_matrix()

    def _rotate_volume(self, array: np.ndarray, rot: np.ndarray, order: int):
        if array.ndim not in (3, 4):
            raise ValueError(f"Unsupported input array ndim={array.ndim}. Expected 3 or 4.")
        if array.ndim == 3:
            z, y, x = array.shape
        else:  # array.ndim == 4
            c, z, y, x = array.shape

        # Coordinate transformation
        center = np.array([z / 2.0, y / 2.0, x / 2.0], dtype=np.float32)
        dz, dy, dx = np.indices((z, y, x))
        coords = np.stack([dz, dy, dx], axis=0).reshape(3, -1).astype(np.float32)  # (3, N)
        coords_centered = coords.T - center  # (N, 3)
        coords_rotated = (rot @ coords_centered.T).T + center  # (N, 3)
        coords_list = [coords_rotated[:, 0], coords_rotated[:, 1], coords_rotated[:, 2]]

        def _map_single_channel(vol3: np.ndarray) -> np.ndarray:
            mapped = map_coordinates(
                vol3,
                tuple(coords_list),
                order=cast(Literal[0, 1, 2, 3, 4, 5], order),
                mode="constant",
                cval=self.pad_val,
                prefilter=self.resample_prefilter,
            ).reshape(z, y, x)
            # Try to preserve original dtype (e.g., labels are integers)
            try:
                return mapped.astype(vol3.dtype, copy=False)
            except Exception:
                return mapped

        if array.ndim == 3:
            return _map_single_channel(array)
        else:
            # 4D: per-channel map
            out = np.empty_like(array)
            for ch in range(array.shape[0]):
                out[ch] = _map_single_channel(array[ch])
            return out

    def _center_crop(self, array: np.ndarray, bounds):
        zmin, zmax, ymin, ymax, xmin, xmax = bounds
        return array[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

    def transform(self, results):
        if np.random.rand() < self.prob:
            rot = self._sample_rotation_matrix()
            for key in self.img_keys:
                results[key] = self._rotate_volume(results[key], rot, self.interp_order)
            for key in results.get("seg_fields", []):
                results[key] = self._rotate_volume(results[key], rot, 0)
        return results


class RandomRotate3D_GPU:
    def __init__(self, angle_range:Sequence[float]):
        self.angle_range = angle_range # degrees

    def _gen_grid(self, sample):
        N, C, Z, Y, X = sample.shape
        device = sample.device

        # Random Euler angles shared across the batch (radians; order [Z, Y, X])
        ang_z = math.radians(random.uniform(-self.angle_range[0], self.angle_range[0]))
        ang_y = math.radians(random.uniform(-self.angle_range[1], self.angle_range[1]))
        ang_x = math.radians(random.uniform(-self.angle_range[2], self.angle_range[2]))

        # Rotation matrix
        cz, sz = math.cos(ang_z), math.sin(ang_z)
        cy, sy = math.cos(ang_y), math.sin(ang_y)
        cx, sx = math.cos(ang_x), math.sin(ang_x)
        R = torch.tensor([
            [cz * cy,                          cz * sy * sx - sz * cx,  cz * sy * cx + sz * sx],
            [sz * cy,                          sz * sy * sx + cz * cx,  sz * sy * cx - cz * sx],
            [-sy,                               cy * sx,                 cy * cx               ],
        ], device=device, dtype=torch.float32)

        # Map R to the normalized affine theta for affine_grid
        # Note: With align_corners=True and identical input/output sizes, rotation
        # around voxel centers corresponds to rotation around the origin in
        # normalized coordinates. Therefore the translation term should be 0,
        # and theta = A @ R @ A^{-1} (no extra translation).
        # Also, grid_sample expects grid order (x, y, z) and scale vector [X, Y, Z].
        A_diag    = torch.tensor([2.0 / (X - 1), 2.0 / (Y - 1), 2.0 / (Z - 1)], device=device, dtype=torch.float32)
        Ainv_diag = torch.tensor([(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0], device=device, dtype=torch.float32)
        M = R * Ainv_diag                  # scale columns (A^{-1})
        M = A_diag.view(3, 1) * M          # scale rows (A)
        t = torch.zeros(3, device=device, dtype=torch.float32)
        theta = torch.cat([M, t.view(3, 1)], dim=1)  # [3,4]
        theta = theta.unsqueeze(0).expand(N, 3, 4).contiguous()

        return F.affine_grid(theta, size=[N, C, Z, Y, X], align_corners=True)  # [N,Z,Y,X,3]

    def warp(self, x:torch.Tensor, grid:torch.Tensor, interp_mode:str):
        if x.ndim != 5:
            raise ValueError(f"image and label must be [N,C,Z,Y,X], got {x.shape}")
        dtype_in = x.dtype
        return F.grid_sample(x.to(torch.float32), grid, mode=interp_mode, padding_mode="border", align_corners=True).to(dtype=dtype_in)


class CenterCrop3D(BaseTransform):
    def __init__(
        self,
        size: list[int],
        keys: list[str] = ["img", "gt_seg_map"]
    ):
        self.size = size
        self.keys = keys

    def transform(self, results):
        for key in self.keys:
            shape = results[key].shape
            center = np.array(shape) // 2
            half_size = np.array(self.size) // 2
            mins = center - half_size
            maxs = center + half_size
            results[key] = results[key][mins[0]:maxs[0],
                                        mins[1]:maxs[1],
                                        mins[2]:maxs[2]]

        if "img_shape" in results:
            results["img_shape"] = self.size

        return results


class RandomPatch3D(BaseTransform):
    def __init__(
        self,
        patch_size: list[int],
        keys: list[str] = ["img", "gt_seg_map"]
    ):
        self.patch_size = patch_size
        self.keys = keys

    def transform(self, results:dict[str, Any]):
        """Randomly crop a patch from the 3D volume

        Args:
            results (dict): Result dict from loading pipeline.
                - img: The image to be cropped, shape [Z, Y, X].
                - gt_seg_map: The segmentation map to be cropped, shape[(Optional) C, Z, Y, X].
                - img_shape: Original shape of the image.

        Returns:
            results (dict): The cropped results.
                - img: The cropped image, shape [pz, py, px].
                - gt_seg_map: The cropped segmentation map, shape[(Optional) C, pz, py, px].
                - img_shape: The shape of the cropped image.
        """
        # Obtain the source image
        img = results.get("img")
        if img is None:
            raise KeyError("`img` key is required for RandomPatch3D")
        # Original volume dimensions
        z, y, x = img.shape[:3]
        pz, py, px = self.patch_size
        # Ensure patch fits within the volume
        if any(dim < p for dim, p in zip((z, y, x), (pz, py, px))):
            raise ValueError(f"Patch size {self.patch_size} exceeds image shape {(z, y, x)}")
        # Random start indices for cropping
        z1 = random.randint(0, z - pz)
        y1 = random.randint(0, y - py)
        x1 = random.randint(0, x - px)
        # Crop each specified key
        for key in self.keys:
            results[key] = results[key][z1:z1+pz, y1:y1+py, x1:x1+px, ...]
        # Update the shape in results
        results["img_shape"] = (pz, py, px)
        return results


class RandomBrightnessContrast(BaseTransform):
    def __init__(self,
                 brightness_limit: tuple[float, float] = (-0.5, 0.5),
                 contrast_limit: tuple[float, float] = (-0.5, 0.5),
                 prob: float = 0.5):
        self.fn = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=prob
        )

    def transform(self, results: dict):
        results["img"] = self.fn(image=results["img"])["image"]
        return results


class RandomGamma(BaseTransform):
    def __init__(self,
                 gamma_limit: tuple[int, int] = (50, 150),
                 prob: float = 0.5):
        self.fn = A.RandomGamma(gamma_limit=gamma_limit, p=prob)

    def transform(self, results: dict):
        results["img"] = self.fn(image=results["img"])["image"]
        return results


class MONAINoise(BaseTransform):
    def __init__(self,
                 method: Literal['gibbs', 'gaussian', 'kspace', 'rician'],
                 prob: float = 0.5,
                 **kwargs):
        from monai.transforms import RandGaussianNoise, RandGibbsNoise, RandKSpaceSpikeNoise, RandRicianNoise
        if method == 'gibbs':
            self.noise_fn = RandGibbsNoise(prob, **kwargs)
        elif method == 'gaussian':
            self.noise_fn = RandGaussianNoise(prob, **kwargs)
        elif method == 'kspace':
            self.noise_fn = RandKSpaceSpikeNoise(prob, **kwargs)
        elif method == 'rician':
            self.noise_fn = RandRicianNoise(prob, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported noise method: {method}")

    def transform(self, results: dict):
        min_HU = results["img"].min()
        results["img"] = self.noise_fn(results["img"] - min_HU) + min_HU
        return results
