from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class LUNA16_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class LUNA16_Mha(LUNA16_base, mgam_SemiSup_3D_Mha):
    pass


class LUNA16_Patch(LUNA16_base, mgam_SeriesPatched_Structure):
    pass
