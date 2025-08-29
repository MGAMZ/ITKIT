from ..base import mgam_SemiSup_3D_Mha, mgam_SemiSup_Precropped_Npz
from .meta import LIVER_CLASS_INDEX_MAP, LIVER_TUMOR_CLASS_INDEX_MAP


# 二分类：# 背景+肝脏

class LiverSeg_base:
    METAINFO = dict(classes=list(LIVER_CLASS_INDEX_MAP.keys()))


class LiverSeg_Precrop_Npz(LiverSeg_base, mgam_SemiSup_Precropped_Npz):
    pass


class LiverSeg_Semi_Mha(LiverSeg_base, mgam_SemiSup_3D_Mha):
    pass


# 三分类：背景+肝脏+肝肿瘤

class LiverTumorSeg_base:
    METAINFO = dict(classes=list(LIVER_TUMOR_CLASS_INDEX_MAP.keys()))


class LiverTumorSeg_Precrop_Npz(LiverTumorSeg_base, mgam_SemiSup_Precropped_Npz):
    pass


class LiverTumorSeg_Semi_Mha(LiverTumorSeg_base, mgam_SemiSup_3D_Mha):
    pass
