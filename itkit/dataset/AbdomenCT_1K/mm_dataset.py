from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class AbdomenCT_1K_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class AbdomenCT_1K_Mha(AbdomenCT_1K_base, mgam_SemiSup_3D_Mha): ...


class AbdomenCT_1K_Patch(AbdomenCT_1K_base, mgam_SeriesPatched_Structure): ...
