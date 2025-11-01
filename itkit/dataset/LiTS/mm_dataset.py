from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from . import CLASS_INDEX_MAP

class LiTS_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class LiTS_Patch(LiTS_base, mgam_SeriesPatched_Structure):
    pass

class LiTS_Mha(LiTS_base, mgam_SemiSup_3D_Mha):
    pass
