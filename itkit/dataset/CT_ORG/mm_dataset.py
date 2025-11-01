from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class CT_ORG_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class CT_ORG_Mha(CT_ORG_base, mgam_SemiSup_3D_Mha):
    pass


class CT_ORG_Patch(CT_ORG_base, mgam_SeriesPatched_Structure):
    pass
