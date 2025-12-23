import logging
from typing import Any, Literal

import torch
import torchio as tio
from mmengine.logging import MMLogger, print_log

from .base import mgam_SemiSup_3D_Mha


class mgam_TorchIO_Patched_Structure(mgam_SemiSup_3D_Mha):
    def __init__(self,
                 data_root: str,
                 queue_max_length: int = 300,
                 samples_per_volume: int = 10,
                 patch_size: int | tuple = (96, 96, 96),
                 sampler_type: Literal['uniform', 'weighted', 'label'] = 'uniform',
                 queue_num_workers: int = 2,
                 sampler_parameters: dict | None = None,
                 *args, **kwargs) -> None:
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume
        self.patch_size = patch_size
        self.sampler_type = sampler_type
        self.queue_num_workers = queue_num_workers
        self.sampler_parameters = sampler_parameters or {}

        self.subjects: list[tio.Subject] = []
        self.subjects_dataset: tio.SubjectsDataset | None = None
        self.tio_queue: tio.Queue | None = None

        super().__init__(data_root=data_root, *args, **kwargs)

    def load_data_list(self) -> list[dict]:
        # 1. Load original data list (paths)
        # This calls mgam_BaseSegDataset.load_data_list -> sample_iterator
        raw_data_list = super().load_data_list()

        # 2. Build TorchIO Subjects
        self.subjects = []
        for item in raw_data_list:
            subject_dict: dict[str, Any] = {
                'image': tio.ScalarImage(item['img_path']),
            }
            if item.get('seg_map_path'):
                subject_dict['label'] = tio.LabelMap(item['seg_map_path'])

            # Store original metadata
            subject = tio.Subject(**subject_dict)
            subject['mm_meta'] = item
            self.subjects.append(subject)

        self.subjects_dataset = tio.SubjectsDataset(self.subjects)

        # 3. Return dummy list with correct length
        # Length = num_subjects * samples_per_volume
        return [{} for _ in range(len(self.subjects) * self.samples_per_volume)]

    def _initialize_queue(self):
        # 1. Determine subset of subjects for this worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Main process or single worker
            subset_subjects = self.subjects
        else:
            # Split subjects among workers
            subset_subjects = self.subjects[worker_info.id::worker_info.num_workers]
            if not subset_subjects:
                 raise RuntimeError(f"Worker {worker_info.id} has no data. "
                                   f"Reduce num_workers (dataset size: {len(self.subjects)}).")

        # Create a dataset specifically for this queue (might be subset)
        queue_dataset = tio.SubjectsDataset(subset_subjects)

        # 2. Build Sampler
        if self.sampler_type == 'uniform':
            sampler = tio.data.UniformSampler(self.patch_size)
        elif self.sampler_type == 'weighted':
            sampler = tio.data.WeightedSampler(self.patch_size, **self.sampler_parameters)
        elif self.sampler_type == 'label':
            sampler = tio.data.LabelSampler(self.patch_size, **self.sampler_parameters)
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

        # 3. Build Queue
        print_log(f"Initializing TorchIO Queue with {len(subset_subjects)} subjects (subset), "
                  f"max_len={self.queue_max_length}, workers={self.queue_num_workers}",
                  MMLogger.get_current_instance(),
                  logging.DEBUG)

        self.tio_queue = tio.Queue(
            queue_dataset,
            self.queue_max_length,
            self.samples_per_volume,
            sampler,
            num_workers=self.queue_num_workers,
            shuffle_subjects=True,
            shuffle_patches=True
        )

    def prepare_data(self, idx) -> Any:
        if self.tio_queue is None:
             self._initialize_queue()

        if self.tio_queue is None: # Should not happen
            raise RuntimeError("Queue initialization failed.")

        patch = self.tio_queue[0]
        data_info = patch.get('mm_meta', {}).copy()

        data_info['img'] = patch['image'][tio.DATA].squeeze(0).numpy()  # pyright: ignore[reportAttributeAccessIssue]
        if 'label' in patch:
            data_info['gt_seg_map'] = patch['label'][tio.DATA].squeeze(0).numpy()  # pyright: ignore[reportAttributeAccessIssue]
            data_info['seg_fields'] = ['gt_seg_map']
        data_info['img_shape'] = patch['image'][tio.DATA].shape[1:]
        data_info['ori_shape'] = patch['image'][tio.DATA].shape[1:]
        data_info['patch_location'] = patch[tio.LOCATION]

        return self.pipeline(data_info)
