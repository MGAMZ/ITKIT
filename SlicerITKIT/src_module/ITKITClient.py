import logging
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


class ITKITClient:
    """HTTP client for interacting with the ITKIT inference server."""

    def get_server_info(self, server_url: str) -> dict | None:
        """Get server information; return None when connection fails."""
        try:
            LOGGER.debug("GET %s/api/info", server_url)
            response = requests.get(f"{server_url}/api/info", timeout=5)
            response.raise_for_status()
            LOGGER.info("Server info retrieved")
            return response.json()
        except Exception as exc:
            LOGGER.error("Failed to get server info: %s", exc)
            return None

    def load_model(
        self,
        server_url: str,
        backend_type: str,
        config_path: Optional[str],
        model_path: str,
        inference_config: dict,
    ) -> bool:
        """Load model on server and return True on success."""
        data = {
            "backend_type": backend_type,
            "model_path": model_path,
            "inference_config": inference_config,
        }

        if config_path:
            data["config_path"] = config_path

        LOGGER.debug("POST %s/api/model payload keys: %s", server_url, list(data.keys()))
        response = requests.post(f"{server_url}/api/model", json=data, timeout=120)
        if not response.ok:
            try:
                error_payload = response.json()
                error_msg = error_payload.get("error", response.text)
            except Exception:
                error_msg = response.text
            LOGGER.error("Model load failed (%s): %s", response.status_code, error_msg)
            raise RuntimeError(f"Server error {response.status_code}: {error_msg}")

        result = response.json()
        LOGGER.info("Model load response: %s", result.get("status"))
        return result.get("status") == "success"

    def unload_model(self, server_url: str) -> bool:
        """Unload model from server and return True on success."""
        try:
            LOGGER.debug("DELETE %s/api/model", server_url)
            response = requests.delete(f"{server_url}/api/model", timeout=10)
            response.raise_for_status()
            result = response.json()
            LOGGER.info("Model unload response: %s", result.get("status"))
            return result.get("status") == "success"
        except Exception as exc:
            LOGGER.error("Failed to unload model: %s", exc)
            return False

    def run_inference(
        self,
        server_url: str,
        input_image_path: str,
        force_cpu: bool = False,
        window_level: float | None = None,
        window_width: float | None = None,
    ) -> bytes:
        """Run inference and return raw segmentation bytes from server response."""
        with open(input_image_path, "rb") as image_file:
            files = {"image": image_file}
            data = {"force_cpu": str(force_cpu).lower()}
            if window_level is not None:
                data["window_level"] = str(window_level)
            if window_width is not None:
                data["window_width"] = str(window_width)

            LOGGER.debug("POST %s/api/infer with force_cpu=%s", server_url, force_cpu)
            response = requests.post(
                f"{server_url}/api/infer",
                files=files,
                data=data,
                timeout=600,
            )
            response.raise_for_status()
            LOGGER.info("Inference response received: %d bytes", len(response.content))
            return response.content
