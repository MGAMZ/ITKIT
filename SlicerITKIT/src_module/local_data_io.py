import logging
import os

import slicer

LOGGER = logging.getLogger(__name__)


def scan_image_label_pairs(image_folder: str, label_folder: str) -> list[dict[str, str]]:
    """Find image-label pairs whose filenames match exactly in both folders."""
    if not image_folder:
        raise ValueError("Image folder is required")
    if not label_folder:
        raise ValueError("Label folder is required")
    if not os.path.isdir(image_folder):
        raise ValueError(f"Image folder does not exist: {image_folder}")
    if not os.path.isdir(label_folder):
        raise ValueError(f"Label folder does not exist: {label_folder}")

    image_files = {
        entry.name: os.path.join(image_folder, entry.name)
        for entry in os.scandir(image_folder)
        if entry.is_file()
    }
    label_files = {
        entry.name: os.path.join(label_folder, entry.name)
        for entry in os.scandir(label_folder)
        if entry.is_file()
    }

    matched_names = sorted(set(image_files) & set(label_files))
    LOGGER.info("Found %d matched image-label pairs", len(matched_names))

    return [
        {
            "series_name": file_name,
            "image_path": image_files[file_name],
            "label_path": label_files[file_name],
        }
        for file_name in matched_names
    ]


def import_labelmap_to_segmentation(
    label_path: str,
    reference_volume,
    output_segmentation=None,
    segmentation_name: str | None = None,
):
    """Load a labelmap file and import it into a segmentation node."""
    label_map_node = slicer.util.loadLabelVolume(label_path)
    if label_map_node is None:
        raise RuntimeError(f"Failed to load label volume: {label_path}")

    created_new_segmentation = False
    if output_segmentation is None:
        output_segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        created_new_segmentation = True

    try:
        if segmentation_name:
            output_segmentation.SetName(segmentation_name)
        elif created_new_segmentation:
            output_segmentation.SetName(reference_volume.GetName() + "_Segmentation")

        output_segmentation.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            label_map_node, output_segmentation
        )
    except Exception:
        if created_new_segmentation:
            slicer.mrmlScene.RemoveNode(output_segmentation)
        raise
    finally:
        slicer.mrmlScene.RemoveNode(label_map_node)

    return output_segmentation


def load_image_label_pair(
    image_path: str,
    label_path: str,
    series_name: str,
) -> tuple[slicer.vtkMRMLScalarVolumeNode, slicer.vtkMRMLSegmentationNode]:
    """Load a local image-label pair and convert the label to a segmentation."""
    if not os.path.isfile(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    if not os.path.isfile(label_path):
        raise ValueError(f"Label file not found: {label_path}")

    image_node = slicer.util.loadVolume(image_path)
    if image_node is None:
        raise RuntimeError(f"Failed to load image volume: {image_path}")
    image_node.SetName(series_name)

    try:
        segmentation_node = import_labelmap_to_segmentation(
            label_path=label_path,
            reference_volume=image_node,
            output_segmentation=None,
            segmentation_name=series_name + "_Segmentation",
        )
    except Exception:
        slicer.mrmlScene.RemoveNode(image_node)
        raise

    app_logic = slicer.app.applicationLogic()
    selection_node = app_logic.GetSelectionNode()
    selection_node.SetActiveVolumeID(image_node.GetID())
    app_logic.PropagateVolumeSelection()

    LOGGER.info("Loaded local pair: %s", series_name)
    return image_node, segmentation_node
