from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class FLARE_2023_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class FLARE_2023_Patch(FLARE_2023_base, mgam_SeriesPatched_Structure):
    pass

class FLARE_2023_Mha(FLARE_2023_base, mgam_SemiSup_3D_Mha):
    pass
