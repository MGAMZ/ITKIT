from collections.abc import Iterator
from typing import Any

import monai.data as monai_data
import monai.transforms as monai_transforms
import numpy as np
import SimpleITK as sitk
import torch

from .base import SeriesVolumeDataset


class mgam_MONAI_Patched_Structure(SeriesVolumeDataset):
    def __init__(self,
                 data_root: str,
                 samples_per_volume: int = 10,
                 patch_size: tuple = (96, 96, 96),
                 *args, **kwargs) -> None:
        self.samples_per_volume = samples_per_volume
        self.patch_size = patch_size
        self.patch_iter = None
        super().__init__(data_root=data_root, *args, **kwargs)

    def prepare_data(self, idx) -> Any:  # idx unused, required by interface
        if self.patch_iter is None:
            self.patch_iter = self._get_subset_iterator()

        try:
            patch_data = next(self.patch_iter)
        except StopIteration:
            # Infinite queue: restart when exhausted
            self.patch_iter = self._get_subset_iterator()
            patch_data = next(self.patch_iter)

        data_info = patch_data['mm_meta'].copy()
        data_info['img'] = patch_data['image'].detach().cpu().numpy().squeeze(0)

        if 'label' in patch_data:
            data_info['gt_seg_map'] = patch_data['label'].detach().cpu().numpy().squeeze(0)
            data_info['seg_fields'] = ['gt_seg_map']

        data_info['img_shape'] = data_info['img'].shape
        data_info['ori_shape'] = data_info['img'].shape

        # Remove keys with None values to avoid default_collate errors
        if data_info.get('label_map') is None:
            del data_info['label_map']

        return self.pipeline(data_info)

    def load_data_list(self) -> list[dict]:
        raw_data_list = super().load_data_list()

        # Convert to MONAI format, preserve mmseg metadata
        self.monai_data_list = []
        for item in raw_data_list:
            new_item = {"image": item['img_path'], "mm_meta": item}
            if item.get('seg_map_path'):
                new_item["label"] = item['seg_map_path']
            self.monai_data_list.append(new_item)

        # Volume loading transforms (using custom loader)
        self.volume_transforms = monai_transforms.Compose([
            monai_transforms.Lambda(self._load_volume),
            # Pad to ensure volume >= patch_size to avoid size mismatch
            monai_transforms.SpatialPadd(keys=["image", "label"], spatial_size=self.patch_size,
                                          allow_missing_keys=True),
        ])

        # Random patch sampler
        self.patch_sampler = monai_transforms.RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=self.patch_size,
            num_samples=self.samples_per_volume,
            random_center=True,
            random_size=False,
            allow_missing_keys=True
        )

        # Create dummy PatchDataset for length calculation
        tmp_volume_ds = monai_data.Dataset(data=self.monai_data_list, transform=self.volume_transforms)
        self.monai_patch_ds = monai_data.PatchDataset(
            data=tmp_volume_ds,  # pyright: ignore[reportArgumentType]
            patch_func=self.patch_sampler,
            samples_per_image=self.samples_per_volume
        )

        return [{} for _ in range(len(self.monai_patch_ds))]

    def _get_subset_iterator(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        subset_data = self.monai_data_list if worker_info is None else \
                      self.monai_data_list[worker_info.id::worker_info.num_workers]

        if not subset_data:
            raise RuntimeError(f"Worker {worker_info.id if worker_info else 'main'} has no data. "
                               f"Reduce num_workers (dataset size: {len(self.monai_data_list)}).")

        _volume_ds = monai_data.Dataset(data=subset_data, transform=self.volume_transforms)
        _patch_ds = monai_data.PatchDataset(
            data=_volume_ds,  # pyright: ignore[reportArgumentType]
            patch_func=self.patch_sampler,
            samples_per_image=self.samples_per_volume
        )
        return iter(_patch_ds)

    def _load_volume(self, data_dict):
        """Custom loader using SimpleITK instead of MONAI LoadImaged"""
        sitk_img = sitk.ReadImage(data_dict['image'])
        img_array = sitk.GetArrayFromImage(sitk_img)  # Returns (Z, Y, X)
        img_array = np.expand_dims(img_array, axis=0)  # Add channel dim -> (1, Z, Y, X)

        result = {'image': torch.from_numpy(img_array), 'mm_meta': data_dict['mm_meta']}

        if 'label' in data_dict:
            sitk_label = sitk.ReadImage(data_dict['label'])
            label_array = sitk.GetArrayFromImage(sitk_label)
            label_array = np.expand_dims(label_array, axis=0)
            result['label'] = torch.from_numpy(label_array)

        return result
