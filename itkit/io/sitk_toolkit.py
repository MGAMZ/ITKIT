import os
import warnings
from typing import Literal

import numpy as np
import pydicom
import SimpleITK as sitk
from colorama import Fore, Style

STANDARD_DIRECTION = [1, 0, 0, 0, 1, 0, 0, 0, 1]
STANDARD_ORIGIN = [0, 0, 0]
PIXEL_TYPE = lambda field: sitk.sitkInt16 if field == "image" else sitk.sitkUInt8
INTERPOLATOR = lambda field: sitk.sitkLinear if field == "image" else sitk.sitkNearestNeighbor


def sitk_resample_to_spacing(mha:sitk.Image,
                             spacing: list[float],
                             field: Literal["image", "label"],
                             interp_method=None):
    """Resample an image to a new spacing.

    Args:
        mha (sitk.Image): input image.
        spacing (tuple[float,float,float]): new spacing at [Z,Y,X] order.
        field (str, optional):
            Processing field, Literal['image', 'label', 'mask'].
            The parameter will affect the interpolation method and the output format.
        standardize (bool, optional): Whether to reset origin and direction. Defaults to False.

    Returns:
        sitk.Image: The resampled image.
    """
    assert field in ["image", "label"], "field must be one of ['image', 'label']"
    assert len(spacing) == 3, f"Spacing must be a 3-tuple, got {spacing}"

    # Calculate the resampled spacing.
    # The `None` values will be filled with original spacing.
    spacing = spacing[::-1]
    original_spacing = mha.GetSpacing()
    for i in range(3):
        if spacing[i] == -1:
            spacing[i] = original_spacing[i]
        else:
            assert spacing[i] > 0, f"Spacing must be positive or -1 (Not Changed), but got {spacing}"

    # Skip if original spacing is equal to target spacing
    if original_spacing == spacing:
        return mha

    # Calculate the resampled size, required by `sitk.Resample` API.
    original_size = mha.GetSize()
    spacing_ratio = [original_spacing[i] / spacing[i] for i in range(3)]
    resampled_size = [int(original_size[i] * spacing_ratio[i]) for i in range(3)]

    # Execute
    try:
        return sitk.Resample(
            image1=mha,
            size=resampled_size,  # type: ignore[arg-type]
            interpolator=interp_method or INTERPOLATOR(field),
            outputSpacing=spacing,
            outputPixelType=PIXEL_TYPE(field),
            outputOrigin=mha.GetOrigin(),
            outputDirection=mha.GetDirection(),
            transform=sitk.Transform(),
        )
    except Exception as e:
        return {
            "error": f"Failed to resample image: {e}",
            "original_size": original_size,
            "original_spacing": original_spacing,
            "spacing": spacing,
            "resampled_size": resampled_size,
            "mha": str(mha),
            "field": field,
        }


def sitk_resample_to_image(
    image: sitk.Image,
    reference_image: sitk.Image,
    field: Literal["image", "label"],
    default_value=0.0,
    interp_method=None
):
    """Resample a sitk.Image to align with a reference sitk.Image.

    Args:
        image (sitk.Image): Input image.
        reference_image (sitk.Image): Reference image to align to.
        field (str, optional): Processing target, choose from 'image', 'label', 'mask'. Determines interpolation and output pixel type.
        default_value (float, optional): Fill value used during resampling. Defaults to 0.

    Returns:
        sitk.Image: Resampled sitk.Image.
    """
    return sitk.Resample(
        image1=image,
        size=reference_image.GetSize(),
        interpolator=interp_method or INTERPOLATOR(field),
        outputSpacing=reference_image.GetSpacing(),
        outputPixelType=PIXEL_TYPE(field),
        outputOrigin=reference_image.GetOrigin(),
        outputDirection=reference_image.GetDirection(),
        defaultPixelValue=default_value,
    )


def sitk_resample_to_size(
    image,
    new_size: list[float],
    field: Literal["image", "label"],
    interp_method=None
):
    """Resample a sitk.Image to a new size.

    Args:
        image (sitk.Image): Input image.
        new_size (list[float]): New size [Z, Y, X]; use -1 to keep original for a dimension.
        field (str, optional): Processing target, choose from 'image', 'label', 'mask'. Determines interpolation and output pixel type.
        interp_method: Custom interpolation method (overrides default).

    Returns:
        sitk.Image: Resampled sitk.Image.
    """
    assert len(new_size) == 3, f"Size must be a 3-tuple, got {new_size}"

    new_size = new_size[::-1]
    original_size = image.GetSize()
    for i in range(3):
        if new_size[i] == -1:
            new_size[i] = original_size[i]
        else:
            assert new_size[i] > 0, f"Size must be positive or -1 (Not Changed), but got {new_size}"

    if new_size == original_size:
        return image

    original_spacing = image.GetSpacing()
    new_spacing = np.divide(original_spacing, np.divide(new_size, original_size))

    return sitk.Resample(
        image1=image,
        size=new_size,  # type: ignore[arg-type]
        interpolator=interp_method or INTERPOLATOR(field),
        outputSpacing=new_spacing,
        outputPixelType=PIXEL_TYPE(field),
        outputOrigin=image.GetOrigin(),
        outputDirection=image.GetDirection(),
        transform=sitk.Transform(),
    )


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.0):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float32).T * default_value, isVector=False)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def nii_to_sitk(
    nii_path: str,
    field: Literal["image", "label"],
    value_offset: int | float | None = None,
) -> sitk.Image:
    try:
        sitk_img = sitk.ReadImage(nii_path, outputPixelType=sitk.sitkInt16 if field == "image" else sitk.sitkUInt8)
        if value_offset is not None:
            sitk_img_new = sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk_img) + value_offset)
            sitk_img_new.CopyInformation(sitk_img)
            sitk_img = sitk_img_new

    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file: {nii_path}.") from e

    return sitk_img


def LoadDcmAsSitkImage(dcm_paths, read_workers=8):
    dcms = []

    for dcm_path in dcm_paths:
        ds = pydicom.dcmread(dcm_path, force=True)
        if (0x20, 0x32) not in ds:  # (0020, 0032) Image Position (Patient)
            warnings.warn(
                Fore.YELLOW
                + f"ImagePosition Missing, Deprecating: {dcm_paths}"
                + Style.RESET_ALL
            )
            return False
        dcms.append((dcm_path, ds[0x20, 0x32].value[-1]))
    else:
        dcms = sorted(dcms, key=lambda x: x[1], reverse=False)

    sorted_dcm_paths = [dcm[0] for dcm in dcms]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_dcm_paths)
    reader.SetNumberOfWorkUnits(read_workers)
    sitk_image: sitk.Image = reader.Execute()

    return sitk_image


def merge_masks(mhas: list[str]|list[sitk.Image], dtype=np.uint8) -> sitk.Image:
    """
    Merge all class-wise binary masks into a single multi-class mask and return as a SimpleITK image.

    :param mhas: List of mask file paths or sitk.Image objects.
    :param dtype: Output data type.
    :return: Merged SimpleITK image (voxel value = class index, background=0).
    """
    # 初始化一个空的掩码图像
    mask = None
    merged_mask = None

    # 遍历mha文件路径列表中的每个文件
    for class_index, mha in enumerate(mhas):
        if isinstance(mha, str) and os.path.isfile(mha):
            mask = sitk.ReadImage(mha)
        elif isinstance(mha, sitk.Image):
            mask = mha
        else:
            raise NotImplementedError(f"Unsupported type: {type(mha)}")

        mask_array = sitk.GetArrayFromImage(mask)
        if merged_mask is None:
            merged_mask = np.zeros_like(mask_array)
        if np.any(merged_mask[mask_array == 1]):
            print(f"Warning: Overlapping masks detected for class {class_index + 1}.")
        merged_mask[mask_array == 1] = class_index + 1

    if merged_mask is None:
        raise ValueError("No mask found in the provided paths")

    # 将合并后的掩码转换为SimpleITK图像
    merged_mask_image = sitk.GetImageFromArray(merged_mask.astype(dtype))
    merged_mask_image.CopyInformation(mask)
    return merged_mask_image


def split_image_label_pairs_to_2d(image: sitk.Image, label: sitk.Image):
    """
    Split a 3D image and its label volume into 2D slice pairs along the first (Z) axis.

    Args:
        image (sitk.Image): Image volume.
        label (sitk.Image): Label volume.

    Yields:
        tuple[np.ndarray, np.ndarray]: (image_slice, label_slice)
    """
    # Consistency checks
    assert (
        image.GetSize() == label.GetSize()
    ), f"Image size {image.GetSize()} != Label size {label.GetSize()}"
    assert (
        image.GetSpacing() == label.GetSpacing()
    ), f"Image spacing {image.GetSpacing()} != Label spacing {label.GetSpacing()}"
    assert (
        image.GetOrigin() == label.GetOrigin()
    ), f"Image origin {image.GetOrigin()} != Label origin {label.GetOrigin()}"
    # assert image.GetDirection() == label.GetDirection(), f"Image direction {image.GetDirection()} != Label direction {label.GetDirection()}"

    # 将SimpleITK图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label)

    # Z轴切片
    for i in range(len(image_array)):
        image_slice: np.ndarray = image_array[i]
        label_slice: np.ndarray = label_array[i]
        yield image_slice, label_slice
