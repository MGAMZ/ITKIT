from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class ImageTBAD_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class TBAD_Mha(ImageTBAD_base, SeriesVolumeDataset):
    ...


class TBAD_Patch(ImageTBAD_base, PatchedDataset):
    ...
