from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class KiTS23_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class KiTS23_Mha(KiTS23_base, mgam_SemiSup_3D_Mha):
    pass


class KiTS23_Patch(KiTS23_base, mgam_SeriesPatched_Structure):
    pass
