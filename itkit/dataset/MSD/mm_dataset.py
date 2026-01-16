import json
from pathlib import Path
from typing import Any

from ..base import PatchedDataset, SeriesVolumeDataset

JSON_PATH = Path(__file__).parent / "dataset.json"

def _load_metadata() -> dict[str, Any]:
    if not JSON_PATH.exists():
        return {}
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)

MSD_METADATA = _load_metadata()

def _get_classes(task_id: str) -> list[str]:
    labels = MSD_METADATA.get(task_id, {}).get("labels", {})
    return [labels[k] for k in sorted(labels.keys(), key=int)]


# Task01_BrainTumour
class Task01_BrainTumour_base:
    METAINFO = {"classes": _get_classes("Task01_BrainTumour")}

class Task01_BrainTumour_Mha(Task01_BrainTumour_base, SeriesVolumeDataset):
    pass

class Task01_BrainTumour_Patch(Task01_BrainTumour_base, PatchedDataset):
    pass


# Task02_Heart
class Task02_Heart_base:
    METAINFO = {"classes": _get_classes("Task02_Heart")}

class Task02_Heart_Mha(Task02_Heart_base, SeriesVolumeDataset):
    pass

class Task02_Heart_Patch(Task02_Heart_base, PatchedDataset):
    pass


# Task03_Liver
class Task03_Liver_base:
    METAINFO = {"classes": _get_classes("Task03_Liver")}

class Task03_Liver_Mha(Task03_Liver_base, SeriesVolumeDataset):
    pass

class Task03_Liver_Patch(Task03_Liver_base, PatchedDataset):
    pass


# Task04_Hippocampus
class Task04_Hippocampus_base:
    METAINFO = {"classes": _get_classes("Task04_Hippocampus")}

class Task04_Hippocampus_Mha(Task04_Hippocampus_base, SeriesVolumeDataset):
    pass

class Task04_Hippocampus_Patch(Task04_Hippocampus_base, PatchedDataset):
    pass


# Task05_Prostate
class Task05_Prostate_base:
    METAINFO = {"classes": _get_classes("Task05_Prostate")}

class Task05_Prostate_Mha(Task05_Prostate_base, SeriesVolumeDataset):
    pass

class Task05_Prostate_Patch(Task05_Prostate_base, PatchedDataset):
    pass


# Task06_Lung
class Task06_Lung_base:
    METAINFO = {"classes": _get_classes("Task06_Lung")}

class Task06_Lung_Mha(Task06_Lung_base, SeriesVolumeDataset):
    pass

class Task06_Lung_Patch(Task06_Lung_base, PatchedDataset):
    pass


# Task07_Pancreas
class Task07_Pancreas_base:
    METAINFO = {"classes": _get_classes("Task07_Pancreas")}

class Task07_Pancreas_Mha(Task07_Pancreas_base, SeriesVolumeDataset):
    pass

class Task07_Pancreas_Patch(Task07_Pancreas_base, PatchedDataset):
    pass


# Task08_HepaticVessel
class Task08_HepaticVessel_base:
    METAINFO = {"classes": _get_classes("Task08_HepaticVessel")}

class Task08_HepaticVessel_Mha(Task08_HepaticVessel_base, SeriesVolumeDataset):
    pass

class Task08_HepaticVessel_Patch(Task08_HepaticVessel_base, PatchedDataset):
    pass


# Task09_Spleen
class Task09_Spleen_base:
    METAINFO = {"classes": _get_classes("Task09_Spleen")}

class Task09_Spleen_Mha(Task09_Spleen_base, SeriesVolumeDataset):
    pass

class Task09_Spleen_Patch(Task09_Spleen_base, PatchedDataset):
    pass


# Task10_Colon
class Task10_Colon_base:
    METAINFO = {"classes": _get_classes("Task10_Colon")}

class Task10_Colon_Mha(Task10_Colon_base, SeriesVolumeDataset):
    pass

class Task10_Colon_Patch(Task10_Colon_base, PatchedDataset):
    pass
