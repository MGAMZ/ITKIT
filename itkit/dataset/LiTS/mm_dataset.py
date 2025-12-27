from ..base import PatchedDataset, SeriesVolumeDataset
from . import CLASS_INDEX_MAP


class LiTS_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class LiTS_Patch(LiTS_base, PatchedDataset):
    pass

class LiTS_Mha(LiTS_base, SeriesVolumeDataset):
    pass
