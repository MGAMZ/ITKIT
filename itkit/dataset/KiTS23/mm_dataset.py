from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class KiTS23_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class KiTS23_Mha(KiTS23_base, SeriesVolumeDataset):
    pass

class KiTS23_Patch(KiTS23_base, PatchedDataset):
    pass
