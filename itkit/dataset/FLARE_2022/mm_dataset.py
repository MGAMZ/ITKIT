from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class FLARE_2022_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class FLARE_2022_Mha(FLARE_2022_base, SeriesVolumeDataset):
    pass

class FLARE_2022_Patch(FLARE_2022_base, PatchedDataset):
    pass
