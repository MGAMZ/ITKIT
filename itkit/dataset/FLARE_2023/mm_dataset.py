from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class FLARE_2023_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class FLARE_2023_Patch(FLARE_2023_base, PatchedDataset):
    pass

class FLARE_2023_Mha(FLARE_2023_base, SeriesVolumeDataset):
    pass
