# pyright: reportUndefinedVariable=false
"""
3D Slicer Extension for ITKIT Inference (Client)

This module provides a lightweight 3D Slicer interface for running ITKIT inference.
It communicates with an ITKIT Inference Server via REST API, eliminating the need
to install ITKIT and PyTorch in Slicer's Python environment.

Architecture:
- This plugin is a lightweight client (only needs SimpleITK, requests)
- ITKIT Server runs separately with all dependencies (PyTorch, ITKIT, etc.)
- Communication via REST API over HTTP

Usage:
1. Start ITKIT Server: python SlicerITKIT/server/itkit_server.py
2. Load this module in 3D Slicer
3. Connect to server and run inference
"""

import logging
import os
import tempfile
from typing import Optional

import ctk
import qt
import requests
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# Module logger
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)


class ITKITInference(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ITKIT"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["ITKIT Team"]
        self.parent.helpText = """
This module provides a client interface for running ITKIT inference on 3D medical images.
It communicates with an ITKIT Inference Server via REST API.

Setup:
1. Start ITKIT Server: python SlicerITKIT/server/itkit_server.py
2. Enter server URL (default: http://localhost:8000)
3. Load a model on the server
4. Run inference on loaded volumes

Architecture:
- Lightweight client (no heavy dependencies in Slicer)
- Server runs in separate Python environment
- Clean separation via REST API
"""
        self.parent.acknowledgementText = """
This module was developed using the ITKIT framework.
For more information, visit: https://github.com/MGAMZ/ITKIT
"""

        # Add settings
        slicer.app.connect("startupCompleted()", self.initializeAfterStartup)

    def initializeAfterStartup(self):
        """Initialize settings after application startup."""
        if not slicer.app.commandOptions().noMainWindow:
            # Register settings
            settings = qt.QSettings()
            if not settings.contains("ITKITInference/serverUrl"):
                settings.setValue("ITKITInference/serverUrl", "http://localhost:8000")
            LOGGER.info("ITKITInference settings initialized")


class ITKITInferenceWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class."""

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Create logic
        self.logic = ITKITInferenceLogic()
        LOGGER.info("ITKITInference UI setup started")

        # Server Connection Section
        serverCollapsibleButton = ctk.ctkCollapsibleButton()
        serverCollapsibleButton.text = "Server Connection"
        self.layout.addWidget(serverCollapsibleButton)
        serverFormLayout = qt.QFormLayout(serverCollapsibleButton)

        # Server URL
        self.serverUrlLineEdit = qt.QLineEdit()
        settings = qt.QSettings()
        self.serverUrlLineEdit.setText(
            settings.value("ITKITInference/serverUrl", "http://localhost:8000")
        )
        self.serverUrlLineEdit.setToolTip("URL of the ITKIT Inference Server")
        serverFormLayout.addRow("Server URL:", self.serverUrlLineEdit)

        # Connect button
        self.connectButton = qt.QPushButton("Connect to Server")
        self.connectButton.toolTip = "Test connection to ITKIT server"
        self.connectButton.clicked.connect(self.onConnectButton)
        serverFormLayout.addRow(self.connectButton)

        # Server status
        self.serverStatusLabel = qt.QLabel("Not connected")
        self.serverStatusLabel.setStyleSheet("color: gray;")
        serverFormLayout.addRow("Status:", self.serverStatusLabel)

        # Model Management Section
        modelCollapsibleButton = ctk.ctkCollapsibleButton()
        modelCollapsibleButton.text = "Model Configuration"
        self.layout.addWidget(modelCollapsibleButton)
        modelFormLayout = qt.QFormLayout(modelCollapsibleButton)

        # Backend type selector
        self.backendSelector = qt.QComboBox()
        self.backendSelector.addItem("MMEngine")
        self.backendSelector.addItem("ONNX")
        self.backendSelector.setToolTip("Select the inference backend type")
        self.backendSelector.currentIndexChanged.connect(self.onBackendChanged)
        modelFormLayout.addRow("Backend Type:", self.backendSelector)

        # Config file path (for MMEngine)
        self.configPathLineEdit = qt.QLineEdit()
        self.configPathLineEdit.setToolTip("Path to the model configuration file (.py)")
        self.configBrowseButton = qt.QPushButton("Browse...")
        self.configBrowseButton.clicked.connect(self.onConfigBrowseClicked)
        configLayout = qt.QHBoxLayout()
        configLayout.addWidget(self.configPathLineEdit)
        configLayout.addWidget(self.configBrowseButton)
        modelFormLayout.addRow("Config File:", configLayout)

        # Model file path
        self.modelPathLineEdit = qt.QLineEdit()
        self.modelPathLineEdit.setToolTip(
            "Path to checkpoint (.pth) or ONNX model (.onnx)"
        )
        self.modelBrowseButton = qt.QPushButton("Browse...")
        self.modelBrowseButton.clicked.connect(self.onModelBrowseClicked)
        modelLayout = qt.QHBoxLayout()
        modelLayout.addWidget(self.modelPathLineEdit)
        modelLayout.addWidget(self.modelBrowseButton)
        modelFormLayout.addRow("Model File:", modelLayout)

        # FP16 checkbox
        self.fp16CheckBox = qt.QCheckBox()
        self.fp16CheckBox.setToolTip("Use FP16 precision for inference")
        self.fp16CheckBox.setChecked(False)
        modelFormLayout.addRow("Use FP16:", self.fp16CheckBox)

        # Load/Unload buttons
        buttonLayout = qt.QHBoxLayout()
        self.loadModelButton = qt.QPushButton("Load Model")
        self.loadModelButton.toolTip = (
            "Load the model on the server (replaces any currently loaded model)"
        )
        self.loadModelButton.clicked.connect(self.onLoadModelButton)
        self.loadModelButton.enabled = False
        buttonLayout.addWidget(self.loadModelButton)

        self.unloadModelButton = qt.QPushButton("Unload Model")
        self.unloadModelButton.toolTip = "Unload the current model from server"
        self.unloadModelButton.clicked.connect(self.onUnloadModelButton)
        self.unloadModelButton.enabled = False
        buttonLayout.addWidget(self.unloadModelButton)
        modelFormLayout.addRow(buttonLayout)

        # Current model status
        self.modelStatusLabel = qt.QLabel("No model loaded")
        self.modelStatusLabel.setStyleSheet("color: gray;")
        modelFormLayout.addRow("Current Model:", self.modelStatusLabel)

        # Inference Section
        inferenceCollapsibleButton = ctk.ctkCollapsibleButton()
        inferenceCollapsibleButton.text = "Run Inference"
        self.layout.addWidget(inferenceCollapsibleButton)
        inferenceFormLayout = qt.QFormLayout(inferenceCollapsibleButton)

        # Input volume selector
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Pick the input volume for segmentation")
        inferenceFormLayout.addRow("Input Volume:", self.inputSelector)

        # Patch parameters
        self.patchSizeLineEdit = qt.QLineEdit()
        self.patchSizeLineEdit.setPlaceholderText("e.g., 96,96,96 (optional)")
        self.patchSizeLineEdit.setToolTip("Sliding window patch size")
        inferenceFormLayout.addRow("Patch Size:", self.patchSizeLineEdit)

        self.patchStrideLineEdit = qt.QLineEdit()
        self.patchStrideLineEdit.setPlaceholderText("e.g., 48,48,48 (optional)")
        self.patchStrideLineEdit.setToolTip("Sliding window stride")
        inferenceFormLayout.addRow("Patch Stride:", self.patchStrideLineEdit)

        # Force CPU checkbox
        self.forceCpuCheckBox = qt.QCheckBox()
        self.forceCpuCheckBox.setToolTip("Force CPU accumulation on server")
        self.forceCpuCheckBox.setChecked(False)
        inferenceFormLayout.addRow("Force CPU:", self.forceCpuCheckBox)

        # Windowing overrides
        self.windowLevelLineEdit = qt.QLineEdit()
        self.windowLevelLineEdit.setPlaceholderText("e.g., 50 (optional)")
        self.windowLevelLineEdit.setToolTip(
            "Override window level (WL) for preprocessing"
        )
        inferenceFormLayout.addRow("Window Level (WL):", self.windowLevelLineEdit)

        self.windowWidthLineEdit = qt.QLineEdit()
        self.windowWidthLineEdit.setPlaceholderText("e.g., 350 (optional)")
        self.windowWidthLineEdit.setToolTip(
            "Override window width (WW) for preprocessing"
        )
        inferenceFormLayout.addRow("Window Width (WW):", self.windowWidthLineEdit)

        # Output segmentation selector
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.showHidden = False
        self.outputSelector.showChildNodeTypes = False
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("Pick the output segmentation node")
        self.outputSelector.baseName = "ITKIT_Segmentation"
        inferenceFormLayout.addRow("Output Segmentation:", self.outputSelector)

        # Apply Button
        self.applyButton = qt.QPushButton("Run Inference")
        self.applyButton.toolTip = "Run the inference on the server"
        self.applyButton.enabled = False
        inferenceFormLayout.addRow(self.applyButton)

        # Progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.hide()
        inferenceFormLayout.addRow(self.progressBar)

        # Status label
        self.statusLabel = qt.QLabel()
        inferenceFormLayout.addRow(self.statusLabel)

        # Connections
        self.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.serverUrlLineEdit.textChanged.connect(self.onServerUrlChanged)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Initial state
        self.onBackendChanged()
        LOGGER.info("ITKITInference UI setup completed")

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        pass

    def onServerUrlChanged(self):
        """Called when server URL changes."""
        # Save to settings
        settings = qt.QSettings()
        settings.setValue("ITKITInference/serverUrl", self.serverUrlLineEdit.text)
        LOGGER.info("Server URL updated")

        # Reset connection status
        self.serverStatusLabel.setText("Not connected")
        self.serverStatusLabel.setStyleSheet("color: gray;")
        self.loadModelButton.enabled = False
        self.applyButton.enabled = False

    def onConnectButton(self):
        """Called when connect button is clicked."""
        server_url = self.serverUrlLineEdit.text.strip()
        LOGGER.info("Connecting to server: %s", server_url)

        try:
            self.statusLabel.setText("Connecting to server...")
            slicer.app.processEvents()

            # Test connection
            info = self.logic.get_server_info(server_url)

            if info:
                LOGGER.info("Connected to server: %s", info.get("name", "Unknown"))
                LOGGER.debug("Server info payload: %s", info)
                self.serverStatusLabel.setText(
                    f"Connected - {info.get('name', 'Unknown')}"
                )
                self.serverStatusLabel.setStyleSheet("color: green;")
                self.loadModelButton.enabled = True
                self.statusLabel.setText(
                    f"Server: {info.get('name')}, CUDA: {info.get('cuda_available')}"
                )

                # Update model status
                self.updateModelStatus(info.get("model"))
            else:
                LOGGER.warning("Connection failed for server: %s", server_url)
                self.serverStatusLabel.setText("Connection failed")
                self.serverStatusLabel.setStyleSheet("color: red;")
                self.statusLabel.setText("Failed to connect to server")

        except Exception as e:
            LOGGER.exception("Connect to server failed")
            self.serverStatusLabel.setText("Connection failed")
            self.serverStatusLabel.setStyleSheet("color: red;")
            self.statusLabel.setText(f"Error: {str(e)}")
            slicer.util.errorDisplay(f"Failed to connect to server: {str(e)}")

    def updateModelStatus(self, model_info):
        """Update the current model status display."""
        LOGGER.debug("Updating model status: %s", model_info)

        if model_info and model_info.get("loaded"):
            model_name = model_info.get("name", "Unknown")
            backend_type = model_info.get("backend_type", "Unknown")
            self.modelStatusLabel.setText(f"{model_name} ({backend_type})")
            self.modelStatusLabel.setStyleSheet("color: green;")
            self.unloadModelButton.enabled = True
            self.applyButton.enabled = True
        else:
            self.modelStatusLabel.setText("No model loaded")
            self.modelStatusLabel.setStyleSheet("color: gray;")
            self.unloadModelButton.enabled = False
            self.applyButton.enabled = False

    def onBackendChanged(self):
        """Called when backend type is changed."""
        backend = self.backendSelector.currentText
        LOGGER.info("Backend changed to: %s", backend)
        # Config file is only needed for MMEngine
        if backend == "MMEngine":
            self.configPathLineEdit.setEnabled(True)
            self.configBrowseButton.setEnabled(True)
        else:
            self.configPathLineEdit.setEnabled(False)
            self.configBrowseButton.setEnabled(False)

    def onConfigBrowseClicked(self):
        """Called when config browse button is clicked."""
        LOGGER.debug("Opening config file dialog")
        filePath = qt.QFileDialog.getOpenFileName(
            self.parent,
            "Select Configuration File",
            "",
            "Python Files (*.py);;All Files (*)",
        )
        if filePath:
            LOGGER.info("Selected config path: %s", filePath)
            self.configPathLineEdit.setText(filePath)

    def onModelBrowseClicked(self):
        """Called when model browse button is clicked."""
        backend = self.backendSelector.currentText
        LOGGER.debug("Opening model file dialog for backend: %s", backend)
        if backend == "MMEngine":
            fileFilter = "PyTorch Checkpoint (*.pth *.pt);;All Files (*)"
        else:
            fileFilter = "ONNX Model (*.onnx);;All Files (*)"

        filePath = qt.QFileDialog.getOpenFileName(
            self.parent, "Select Model File", "", fileFilter
        )
        if filePath:
            LOGGER.info("Selected model path: %s", filePath)
            self.modelPathLineEdit.setText(filePath)

    def onLoadModelButton(self):
        """Load a model on the server."""
        try:
            # Get parameters
            server_url = self.serverUrlLineEdit.text.strip()
            backend = self.backendSelector.currentText
            config_path = (
                self.configPathLineEdit.text.strip() if backend == "MMEngine" else None
            )
            model_path = self.modelPathLineEdit.text.strip()

            # Validate
            if not model_path:
                LOGGER.warning("Model load aborted: empty model path")
                slicer.util.errorDisplay("Please select a model file path")
                return

            if backend == "MMEngine" and not config_path:
                LOGGER.warning("Model load aborted: empty config path for MMEngine")
                slicer.util.errorDisplay(
                    "Please select a config file for MMEngine backend"
                )
                return

            # Parse inference config
            inference_config = {"fp16": self.fp16CheckBox.isChecked()}

            LOGGER.info(
                "Loading model (backend=%s, fp16=%s)", backend, inference_config["fp16"]
            )
            LOGGER.debug("Model paths: config=%s model=%s", config_path, model_path)
            LOGGER.debug("Inference config: %s", inference_config)

            # Load model on server
            self.statusLabel.setText("Loading model on server...")
            self.progressBar.show()
            self.progressBar.setValue(50)
            slicer.app.processEvents()

            result = self.logic.load_model(
                server_url=server_url,
                backend_type=backend.lower(),
                config_path=config_path,
                model_path=model_path,
                inference_config=inference_config,
            )

            self.progressBar.setValue(100)
            self.progressBar.hide()

            if result:
                LOGGER.info("Model loaded successfully")
                self.statusLabel.setText("Model loaded successfully")
                # Refresh model status
                self.onConnectButton()
            else:
                LOGGER.warning("Model load failed")
                self.statusLabel.setText("Failed to load model")

        except Exception as e:
            LOGGER.exception("Load model failed")
            self.progressBar.hide()
            self.statusLabel.setText(f"Error: {str(e)}")
            slicer.util.errorDisplay(f"Failed to load model: {str(e)}")

    def onUnloadModelButton(self):
        """Unload the current model from the server."""
        try:
            server_url = self.serverUrlLineEdit.text.strip()

            result = self.logic.unload_model(server_url)

            if result:
                LOGGER.info("Model unloaded")
                self.statusLabel.setText("Model unloaded")
                # Refresh model status
                self.onConnectButton()
            else:
                LOGGER.warning("Model unload failed")
                self.statusLabel.setText("Failed to unload model")

        except Exception as e:
            LOGGER.exception("Unload model failed")
            slicer.util.errorDisplay(f"Failed to unload model: {str(e)}")

    def onApplyButton(self):
        """Run processing when user clicks "Apply" button."""
        self.applyButton.enabled = False
        self.progressBar.show()
        self.progressBar.setValue(10)
        self.statusLabel.setText("Preparing for inference...")

        # Get parameters
        server_url = self.serverUrlLineEdit.text.strip()
        inputVolume = self.inputSelector.currentNode()
        force_cpu = self.forceCpuCheckBox.isChecked()

        wl_text = self.windowLevelLineEdit.text.strip()
        ww_text = self.windowWidthLineEdit.text.strip()
        window_level = float(wl_text) if wl_text else None
        window_width = float(ww_text) if ww_text else None

        if not inputVolume:
            LOGGER.warning("Inference aborted: no input volume selected")
            slicer.util.errorDisplay("Please select an input volume")
            self.applyButton.enabled = True
            self.progressBar.hide()
            return

        self.progressBar.setValue(30)
        self.statusLabel.setText("Running inference on server...")

        # Run inference asynchronously to avoid UI freeze
        def runInference():
            try:
                LOGGER.info("Running inference: force_cpu=%s", force_cpu)
                segmentationNode = self.logic.run_inference(
                    server_url=server_url,
                    input_volume=inputVolume,
                    output_segmentation=self.outputSelector.currentNode(),
                    force_cpu=force_cpu,
                    window_level=window_level,
                    window_width=window_width,
                )

                # Update UI in main thread
                def onComplete():
                    self.progressBar.setValue(100)
                    self.statusLabel.setText("Inference completed successfully!")
                    if self.outputSelector.currentNode() is None:
                        self.outputSelector.setCurrentNode(segmentationNode)
                    self.applyButton.enabled = True
                    self.progressBar.hide()

                qt.QTimer.singleShot(0, onComplete)

            except Exception:
                LOGGER.exception("Inference failed")

                def onError():
                    slicer.util.errorDisplay(f"Inference failed: {str(e)}")
                    self.statusLabel.setText(f"Error: {str(e)}")
                    self.applyButton.enabled = True
                    self.progressBar.hide()

                qt.QTimer.singleShot(0, onError)

        # Schedule async execution
        qt.QTimer.singleShot(100, runInference)


class ITKITInferenceLogic(ScriptedLoadableModuleLogic):
    """This class implements the client logic for communicating with ITKIT server."""

    def __init__(self) -> None:
        """Called when the logic class is instantiated."""
        ScriptedLoadableModuleLogic.__init__(self)

    def get_server_info(self, server_url: str) -> dict | None:
        """Get server information.

        Args:
            server_url: URL of the ITKIT server

        Returns:
            Dictionary with server info or None if failed
        """
        try:
            LOGGER.debug("GET %s/api/info", server_url)
            response = requests.get(f"{server_url}/api/info", timeout=5)
            response.raise_for_status()
            LOGGER.info("Server info retrieved")
            return response.json()
        except Exception as e:
            LOGGER.error("Failed to get server info: %s", e)
            return None

    def load_model(
        self,
        server_url: str,
        backend_type: str,
        config_path: Optional[str],
        model_path: str,
        inference_config: dict,
    ) -> bool:
        """Load a model on the server.

        Args:
            server_url: URL of the ITKIT server
            backend_type: 'mmengine' or 'onnx'
            config_path: Path to config file (optional)
            model_path: Path to model/checkpoint file
            inference_config: Inference configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "backend_type": backend_type,
                "model_path": model_path,
                "inference_config": inference_config,
            }

            if config_path:
                data["config_path"] = config_path
            LOGGER.debug(
                "POST %s/api/model payload keys: %s", server_url, list(data.keys())
            )
            response = requests.post(f"{server_url}/api/model", json=data, timeout=120)
            if not response.ok:
                try:
                    error_payload = response.json()
                    error_msg = error_payload.get("error", response.text)
                except Exception:
                    error_msg = response.text
                LOGGER.error(
                    "Model load failed (%s): %s", response.status_code, error_msg
                )
                raise RuntimeError(f"Server error {response.status_code}: {error_msg}")

            result = response.json()
            LOGGER.info("Model load response: %s", result.get("status"))

            return result.get("status") == "success"

        except Exception as e:
            LOGGER.error("Failed to load model: %s", e)
            raise

    def unload_model(self, server_url: str) -> bool:
        """Unload the current model from the server.

        Args:
            server_url: URL of the ITKIT server

        Returns:
            True if successful, False otherwise
        """
        try:
            LOGGER.debug("DELETE %s/api/model", server_url)
            response = requests.delete(f"{server_url}/api/model", timeout=10)
            response.raise_for_status()
            result = response.json()
            LOGGER.info("Model unload response: %s", result.get("status"))

            return result.get("status") == "success"

        except Exception as e:
            LOGGER.error("Failed to unload model: %s", e)
            return False

    def run_inference(
        self,
        server_url: str,
        input_volume,
        output_segmentation,
        force_cpu: bool = False,
        window_level: float | None = None,
        window_width: float | None = None,
    ) -> slicer.vtkMRMLSegmentationNode:
        """Run inference on the server.

        Args:
            server_url: URL of the ITKIT server
            input_volume: Input scalar volume node
            output_segmentation: Output segmentation node (can be None)
            force_cpu: Force CPU accumulation on server
            window_level: Override window level
            window_width: Override window width

        Returns:
            Output segmentation node
        """
        import time

        startTime = time.time()

        # Export input volume to temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_input_path = tmp.name
        LOGGER.debug("Temp input path: %s", tmp_input_path)

        slicer.util.saveNode(input_volume, tmp_input_path)
        LOGGER.info("Input volume saved to temp file")

        try:
            # Prepare request
            with open(tmp_input_path, "rb") as f:
                files = {"image": f}
                data = {"force_cpu": str(force_cpu).lower()}
                if window_level is not None:
                    data["window_level"] = str(window_level)
                if window_width is not None:
                    data["window_width"] = str(window_width)
                LOGGER.debug(
                    "POST %s/api/infer with force_cpu=%s", server_url, force_cpu
                )
                # Send inference request
                logging.info(f"Sending inference request to {server_url}")
                response = requests.post(
                    f"{server_url}/api/infer",
                    files=files,
                    data=data,
                    timeout=600,  # 10 minutes timeout for inference
                )
                response.raise_for_status()
            LOGGER.info("Inference response received: %d bytes", len(response.content))

            # Save segmentation result to temporary file
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp_output_path = tmp.name
                tmp.write(response.content)
            LOGGER.debug("Temp output path: %s", tmp_output_path)

            # Load segmentation
            segLabelMapNode = slicer.util.loadLabelVolume(tmp_output_path)
            LOGGER.info("Segmentation label volume loaded")

            # Create or update output segmentation node
            if output_segmentation is None:
                output_segmentation = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode"
                )
                output_segmentation.SetName(input_volume.GetName() + "_Segmentation")
                LOGGER.info(
                    "Created new segmentation node: %s", output_segmentation.GetName()
                )

            # Set reference to input volume
            output_segmentation.SetReferenceImageGeometryParameterFromVolumeNode(
                input_volume
            )

            # Import labelmap to segmentation
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                segLabelMapNode, output_segmentation
            )
            LOGGER.info("Imported labelmap to segmentation node")

            # Clean up temporary nodes and files
            slicer.mrmlScene.RemoveNode(segLabelMapNode)
            os.unlink(tmp_output_path)
            LOGGER.debug("Cleaned up temporary output resources")

            stopTime = time.time()
            LOGGER.info("Inference completed in %.2f seconds", stopTime - startTime)

            return output_segmentation

        finally:
            # Clean up input temporary file
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
                LOGGER.debug("Cleaned up temporary input file")


class ITKITInferenceTest(ScriptedLoadableModuleTest):
    """Test case for the scripted module."""

    def setUp(self):
        """Do whatever is needed to reset the state."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ITKITInference1()

    def test_ITKITInference1(self):
        """Test basic module loading."""
        self.delayDisplay("Starting the test")

        # Test that the logic can be instantiated
        logic = ITKITInferenceLogic()
        self.assertIsNotNone(logic)

        self.delayDisplay("Test passed")
