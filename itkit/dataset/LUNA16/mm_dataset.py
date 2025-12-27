from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class LUNA16_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class LUNA16_Mha(LUNA16_base, SeriesVolumeDataset):
    pass

class LUNA16_Patch(LUNA16_base, PatchedDataset):
    pass
