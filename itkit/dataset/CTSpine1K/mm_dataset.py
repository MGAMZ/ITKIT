from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class CTSpine1K_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class CTSpine1K_Mha(CTSpine1K_base, SeriesVolumeDataset):
    ...

class CTSpine1K_Patch(CTSpine1K_base, PatchedDataset):
    ...
