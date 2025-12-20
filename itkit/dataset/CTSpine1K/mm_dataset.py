from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class CTSpine1K_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

class CTSpine1K_Mha(CTSpine1K_base, mgam_SemiSup_3D_Mha):
    ...

class CTSpine1K_Patch(CTSpine1K_base, mgam_SeriesPatched_Structure):
    ...
