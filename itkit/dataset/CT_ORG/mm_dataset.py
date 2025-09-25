from ..base import mgam_SemiSup_3D_Mha, mgam_SemiSup_Precropped_Npz
from .meta import CLASS_INDEX_MAP



class CT_ORG_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class CT_ORG_Precrop_Npz(CT_ORG_base, mgam_SemiSup_Precropped_Npz):
    pass


class CT_ORG_Mha(CT_ORG_base, mgam_SemiSup_3D_Mha):
    pass