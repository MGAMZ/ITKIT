from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class FLARE_2022_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class FLARE_2022_Mha(FLARE_2022_base, mgam_SemiSup_3D_Mha):
    pass


class FLARE_2022_Patch(FLARE_2022_base, mgam_SeriesPatched_Structure):
    pass
