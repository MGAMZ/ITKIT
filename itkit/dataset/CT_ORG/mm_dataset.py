from ..base import PatchedDataset, SeriesVolumeDataset
from .meta import CLASS_INDEX_MAP


class CT_ORG_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class CT_ORG_Mha(CT_ORG_base, SeriesVolumeDataset):
    pass

class CT_ORG_Patch(CT_ORG_base, PatchedDataset):
    pass
