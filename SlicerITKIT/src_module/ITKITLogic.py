import logging
import os
import tempfile
from typing import Optional

import slicer
from ITKITClient import ITKITClient
from local_data_io import import_labelmap_to_segmentation, load_image_label_pair, scan_image_label_pairs
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

LOGGER = logging.getLogger(__name__)


class ITKITLogic(ScriptedLoadableModuleLogic):
    """Client-side orchestration logic for ITKIT inference workflows."""

    def __init__(self) -> None:
        super().__init__()
        self.client = ITKITClient()

    def scan_image_label_pairs(self, image_folder: str, label_folder: str) -> list[dict[str, str]]:
        """Find image-label pairs whose filenames match exactly in both folders."""
        return scan_image_label_pairs(image_folder=image_folder, label_folder=label_folder)

    def load_image_label_pair(
        self, image_path: str, label_path: str, series_name: str
    ) -> tuple[slicer.vtkMRMLScalarVolumeNode, slicer.vtkMRMLSegmentationNode]:
        """Load a local image-label pair and convert the label to a segmentation."""
        return load_image_label_pair(
            image_path=image_path,
            label_path=label_path,
            series_name=series_name,
        )

    def get_server_info(self, server_url: str) -> dict | None:
        """Get server information."""
        return self.client.get_server_info(server_url=server_url)

    def load_model(
        self,
        server_url: str,
        backend_type: str,
        config_path: Optional[str],
        model_path: str,
        inference_config: dict,
    ) -> bool:
        """Load a model on the server."""
        try:
            return self.client.load_model(
                server_url=server_url,
                backend_type=backend_type,
                config_path=config_path,
                model_path=model_path,
                inference_config=inference_config,
            )
        except Exception as exc:
            LOGGER.error("Failed to load model: %s", exc)
            raise

    def unload_model(self, server_url: str) -> bool:
        """Unload the current model from the server."""
        return self.client.unload_model(server_url=server_url)

    def run_inference(
        self,
        server_url: str,
        input_volume,
        output_segmentation,
        force_cpu: bool = False,
        window_level: float | None = None,
        window_width: float | None = None,
    ) -> slicer.vtkMRMLSegmentationNode:
        """Run inference on the server and return output segmentation node."""
        import time

        start_time = time.time()

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_input_path = tmp.name
        LOGGER.debug("Temp input path: %s", tmp_input_path)

        slicer.util.saveNode(input_volume, tmp_input_path)
        LOGGER.info("Input volume saved to temp file")

        try:
            LOGGER.info("Sending inference request to %s", server_url)
            response_content = self.client.run_inference(
                server_url=server_url,
                input_image_path=tmp_input_path,
                force_cpu=force_cpu,
                window_level=window_level,
                window_width=window_width,
            )

            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp_output_path = tmp.name
                tmp.write(response_content)
            LOGGER.debug("Temp output path: %s", tmp_output_path)

            output_segmentation = import_labelmap_to_segmentation(
                label_path=tmp_output_path,
                reference_volume=input_volume,
                output_segmentation=output_segmentation,
                segmentation_name=(
                    None
                    if output_segmentation is not None
                    else input_volume.GetName() + "_Segmentation"
                ),
            )
            LOGGER.info("Imported labelmap to segmentation node")

            os.unlink(tmp_output_path)
            LOGGER.debug("Cleaned up temporary output resources")

            stop_time = time.time()
            LOGGER.info("Inference completed in %.2f seconds", stop_time - start_time)

            return output_segmentation

        finally:
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
                LOGGER.debug("Cleaned up temporary input file")
