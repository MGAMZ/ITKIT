from ..base import mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure
from .meta import CLASS_INDEX_MAP


class ImageTBAD_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class TBAD_Mha(ImageTBAD_base, mgam_SemiSup_3D_Mha): ...


class TBAD_Patch(ImageTBAD_base, mgam_SeriesPatched_Structure): ...
