from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class AbdomenCT_1K_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class AbdomenCT_1K_Mha(AbdomenCT_1K_base, SeriesVolumeDataset):
    ...

class AbdomenCT_1K_Patch(AbdomenCT_1K_base, PatchedDataset):
    ...
