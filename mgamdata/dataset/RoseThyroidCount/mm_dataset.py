import os
import pdb
import json
from typing_extensions import Literal, Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from ..base import mgam_BaseSegDataset
from .meta import CLASS_INDEX_MAP


class RoseThyroidCount_Precrop_Npz(mgam_BaseSegDataset):
    SPLIT_RATIO = [0.7, 0.3]
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))
    
    def __init__(self, data_root:str, include_clustered:bool=True, *args, **kwargs):
        self.include_clustered = include_clustered
        with open(os.path.join(data_root, "slide_processing_log.json"), 'r') as f:
            self.meta:dict = json.load(f)
        self.slide_uids = list(self.meta.keys())
        super().__init__(data_root=data_root, *args, **kwargs)

    def _split(self):
        # delete slides that have no valid patches
        for slide_uid in self.meta.keys():
            slide_meta = self.meta[slide_uid]
            if not any(patch_meta['info'] == 'success' and
                       (self.include_clustered or patch_meta["clustered"] == 1)
                       for patch_meta in slide_meta.values()):
                self.slide_uids.remove(slide_uid)
        
        if self.split == 'train':
            return self.slide_uids[:int(len(self.slide_uids) * self.SPLIT_RATIO[0])]
        elif self.split == 'val' or self.split == 'test':
            return self.slide_uids[int(len(self.slide_uids) * self.SPLIT_RATIO[0]):]
        elif self.split == 'all':
            return self.slide_uids
        else:
            raise ValueError(f"Invalid split: {self.split}. Expected one of ['train', 'val', 'test']")
    
    def sample_iterator(self):
        for slide_uid in self._split():
            slide_meta:dict = self.meta[slide_uid]
            for patch_seriesUID, patch_meta in slide_meta.items():
                if patch_meta['info'] != 'success':
                    continue
                # NOTE 0 代表成团， 1 代表不成团，是倒过来的
                if self.include_clustered or patch_meta["clustered"] == 1:
                    yield (os.path.join(self.data_root, slide_uid, patch_seriesUID+'.npz'),
                           os.path.join(self.data_root, slide_uid, patch_seriesUID+'.npz'))


class LoadRoseThyroidSampleFromNpz(BaseTransform):
    """
    Required Keys:

    - img_path
    - seg_map_path

    Modified Keys:

    - img
    - gt_seg_map
    - seg_fields
    """
    VALID_LOAD_FIELD = Literal["img", "anno"]

    def __init__(self, load_type: VALID_LOAD_FIELD | Sequence[VALID_LOAD_FIELD]):
        self.load_type = load_type if isinstance(load_type, Sequence) else [load_type]
        assert all([load_type in ["img", "anno"] for load_type in self.load_type])

    def transform(self, results):
        assert (results["img_path"] == results["seg_map_path"]), \
            f"img_path: {results['img_path']}, seg_map_path: {results['seg_map_path']}"
        sample_path = results["img_path"]
        sample = np.load(sample_path)

        try:
            if "img" in self.load_type:
                results["img"] = sample["img"] # shape: (Y, X, C)
                results["img_shape"] = results["img"].shape[:-1]
                results["ori_shape"] = results["img"].shape[:-1]

            if "anno" in self.load_type:
                results["points"] = sample["point_map"]
                results["gt_label"] = sample["clustered_cls"]
                
            return results
        
        except Exception as e:
            raise FileNotFoundError(f"Error when loading {sample_path}") from e


class GenPointMap(BaseTransform):
    def transform(self, results:dict):
        size = results["img_shape"]
        points = results.get("points", [])
        point_mask = np.zeros(size, dtype=np.uint8)
        
        if len(points) > 0:
            for point in points:
                x, y = point
                x = np.clip(np.round(x, 0), 0, size[1]-1).astype(int)
                y = np.clip(np.round(y, 0), 0, size[0]-1).astype(int)
                point_mask[y, x] += 1
        
        results["gt_seg_map"] = point_mask # shape: (Y, X)
        results["seg_fields"].append("gt_seg_map")
        return results
