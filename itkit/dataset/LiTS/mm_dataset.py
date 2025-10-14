from ..base import mgam_SemiSup_3D_Mha, mgam_SemiSup_Precropped_Npz
from . import CLASS_INDEX_MAP

class LiTS_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class LiTS_Precrop_Npz(LiTS_base, mgam_SemiSup_Precropped_Npz):
    pass


class LiTS_Mha(LiTS_base, mgam_SemiSup_3D_Mha):
    pass