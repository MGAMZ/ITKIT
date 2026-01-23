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
import SimpleITK as sitk
import sitkUtils
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# Try to import requests (should be available in most Slicer installations)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


#
# ITKITInference
#


class ITKITInference(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ITKIT Inference"
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
            if not settings.contains('ITKITInference/serverUrl'):
                settings.setValue('ITKITInference/serverUrl', 'http://localhost:8000')


#
# ITKITInferenceWidget
#


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

        # Check if requests is available
        if not REQUESTS_AVAILABLE:
            errorLabel = qt.QLabel(
                "ERROR: 'requests' library is not installed.\n"
                "Please install it with: pip install requests\n"
                "(in Slicer's Python console or environment)"
            )
            errorLabel.setStyleSheet("color: red; font-weight: bold; padding: 10px;")
            self.layout.addWidget(errorLabel)
            return

        # Create logic
        self.logic = ITKITInferenceLogic()

        # Server Connection Section
        serverCollapsibleButton = ctk.ctkCollapsibleButton()
        serverCollapsibleButton.text = "Server Connection"
        self.layout.addWidget(serverCollapsibleButton)
        serverFormLayout = qt.QFormLayout(serverCollapsibleButton)

        # Server URL
        self.serverUrlLineEdit = qt.QLineEdit()
        settings = qt.QSettings()
        self.serverUrlLineEdit.setText(settings.value('ITKITInference/serverUrl', 'http://localhost:8000'))
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
        modelCollapsibleButton.text = "Model Management"
        self.layout.addWidget(modelCollapsibleButton)
        modelFormLayout = qt.QFormLayout(modelCollapsibleButton)

        # Model name
        self.modelNameLineEdit = qt.QLineEdit()
        self.modelNameLineEdit.setPlaceholderText("e.g., abdomen_seg")
        self.modelNameLineEdit.setToolTip("Name for the model (will be used to reference it)")
        modelFormLayout.addRow("Model Name:", self.modelNameLineEdit)

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
        self.modelPathLineEdit.setToolTip("Path to checkpoint (.pth) or ONNX model (.onnx)")
        self.modelBrowseButton = qt.QPushButton("Browse...")
        self.modelBrowseButton.clicked.connect(self.onModelBrowseClicked)
        modelLayout = qt.QHBoxLayout()
        modelLayout.addWidget(self.modelPathLineEdit)
        modelLayout.addWidget(self.modelBrowseButton)
        modelFormLayout.addRow("Model File:", modelLayout)

        # Inference parameters
        self.patchSizeLineEdit = qt.QLineEdit()
        self.patchSizeLineEdit.setPlaceholderText("e.g., 96,96,96 (optional)")
        self.patchSizeLineEdit.setToolTip("Sliding window patch size")
        modelFormLayout.addRow("Patch Size:", self.patchSizeLineEdit)

        self.patchStrideLineEdit = qt.QLineEdit()
        self.patchStrideLineEdit.setPlaceholderText("e.g., 48,48,48 (optional)")
        self.patchStrideLineEdit.setToolTip("Sliding window stride")
        modelFormLayout.addRow("Patch Stride:", self.patchStrideLineEdit)

        self.fp16CheckBox = qt.QCheckBox()
        self.fp16CheckBox.setToolTip("Use FP16 precision for inference")
        self.fp16CheckBox.setChecked(False)
        modelFormLayout.addRow("Use FP16:", self.fp16CheckBox)

        # Load/Unload buttons
        buttonLayout = qt.QHBoxLayout()
        self.loadModelButton = qt.QPushButton("Load Model on Server")
        self.loadModelButton.toolTip = "Load the model on the server"
        self.loadModelButton.clicked.connect(self.onLoadModelButton)
        self.loadModelButton.enabled = False
        buttonLayout.addWidget(self.loadModelButton)

        self.unloadModelButton = qt.QPushButton("Unload Model")
        self.unloadModelButton.toolTip = "Unload the model from server"
        self.unloadModelButton.clicked.connect(self.onUnloadModelButton)
        self.unloadModelButton.enabled = False
        buttonLayout.addWidget(self.unloadModelButton)
        modelFormLayout.addRow(buttonLayout)

        # Loaded models list
        self.modelsListWidget = qt.QListWidget()
        self.modelsListWidget.setToolTip("Models loaded on the server")
        modelFormLayout.addRow("Loaded Models:", self.modelsListWidget)

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

        # Model selector for inference
        self.inferenceModelSelector = qt.QComboBox()
        self.inferenceModelSelector.setToolTip("Select a loaded model for inference")
        inferenceFormLayout.addRow("Model:", self.inferenceModelSelector)

        # Force CPU checkbox
        self.forceCpuCheckBox = qt.QCheckBox()
        self.forceCpuCheckBox.setToolTip("Force CPU accumulation on server")
        self.forceCpuCheckBox.setChecked(False)
        inferenceFormLayout.addRow("Force CPU:", self.forceCpuCheckBox)

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

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        pass

    def onServerUrlChanged(self):
        """Called when server URL changes."""
        # Save to settings
        settings = qt.QSettings()
        settings.setValue('ITKITInference/serverUrl', self.serverUrlLineEdit.text)
        
        # Reset connection status
        self.serverStatusLabel.setText("Not connected")
        self.serverStatusLabel.setStyleSheet("color: gray;")
        self.loadModelButton.enabled = False
        self.applyButton.enabled = False

    def onConnectButton(self):
        """Called when connect button is clicked."""
        server_url = self.serverUrlLineEdit.text.strip()
        
        try:
            self.statusLabel.setText("Connecting to server...")
            slicer.app.processEvents()
            
            # Test connection
            info = self.logic.get_server_info(server_url)
            
            if info:
                self.serverStatusLabel.setText(f"Connected - {info.get('name', 'Unknown')}")
                self.serverStatusLabel.setStyleSheet("color: green;")
                self.loadModelButton.enabled = True
                self.statusLabel.setText(f"Server: {info.get('name')}, CUDA: {info.get('cuda_available')}")
                
                # Update loaded models list
                self.updateModelsList(info.get('models', {}))
            else:
                self.serverStatusLabel.setText("Connection failed")
                self.serverStatusLabel.setStyleSheet("color: red;")
                self.statusLabel.setText("Failed to connect to server")
        
        except Exception as e:
            self.serverStatusLabel.setText("Connection failed")
            self.serverStatusLabel.setStyleSheet("color: red;")
            self.statusLabel.setText(f"Error: {str(e)}")
            slicer.util.errorDisplay(f"Failed to connect to server: {str(e)}")

    def updateModelsList(self, models_dict):
        """Update the list of loaded models."""
        self.modelsListWidget.clear()
        self.inferenceModelSelector.clear()
        
        for model_name, model_info in models_dict.items():
            if model_info.get('loaded'):
                self.modelsListWidget.addItem(f"{model_name} ({model_info.get('backend_type')})")
                self.inferenceModelSelector.addItem(model_name)
        
        # Enable/disable inference button
        self.applyButton.enabled = self.inferenceModelSelector.count() > 0

    def onBackendChanged(self):
        """Called when backend type is changed."""
        backend = self.backendSelector.currentText
        # Config file is only needed for MMEngine
        if backend == "MMEngine":
            self.configPathLineEdit.setEnabled(True)
            self.configBrowseButton.setEnabled(True)
        else:
            self.configPathLineEdit.setEnabled(False)
            self.configBrowseButton.setEnabled(False)

    def onConfigBrowseClicked(self):
        """Called when config browse button is clicked."""
        filePath = qt.QFileDialog.getOpenFileName(
            self.parent,
            "Select Configuration File",
            "",
            "Python Files (*.py);;All Files (*)"
        )
        if filePath:
            self.configPathLineEdit.setText(filePath)

    def onModelBrowseClicked(self):
        """Called when model browse button is clicked."""
        backend = self.backendSelector.currentText
        if backend == "MMEngine":
            fileFilter = "PyTorch Checkpoint (*.pth *.pt);;All Files (*)"
        else:
            fileFilter = "ONNX Model (*.onnx);;All Files (*)"
        
        filePath = qt.QFileDialog.getOpenFileName(
            self.parent,
            "Select Model File",
            "",
            fileFilter
        )
        if filePath:
            self.modelPathLineEdit.setText(filePath)

    def onLoadModelButton(self):
        """Load a model on the server."""
        try:
            # Get parameters
            server_url = self.serverUrlLineEdit.text.strip()
            model_name = self.modelNameLineEdit.text.strip()
            backend = self.backendSelector.currentText
            config_path = self.configPathLineEdit.text.strip() if backend == "MMEngine" else None
            model_path = self.modelPathLineEdit.text.strip()
            
            # Validate
            if not model_name:
                slicer.util.errorDisplay("Please enter a model name")
                return
            
            if not model_path or not os.path.exists(model_path):
                slicer.util.errorDisplay("Please select a valid model file")
                return
            
            if backend == "MMEngine" and (not config_path or not os.path.exists(config_path)):
                slicer.util.errorDisplay("Please select a valid config file for MMEngine backend")
                return
            
            # Parse inference config
            inference_config = {}
            
            patch_size_text = self.patchSizeLineEdit.text.strip()
            if patch_size_text:
                try:
                    inference_config['patch_size'] = [int(x.strip()) for x in patch_size_text.split(',')]
                except:
                    slicer.util.errorDisplay("Invalid patch size format")
                    return
            
            patch_stride_text = self.patchStrideLineEdit.text.strip()
            if patch_stride_text:
                try:
                    inference_config['patch_stride'] = [int(x.strip()) for x in patch_stride_text.split(',')]
                except:
                    slicer.util.errorDisplay("Invalid patch stride format")
                    return
            
            inference_config['fp16'] = self.fp16CheckBox.isChecked()
            
            # Load model on server
            self.statusLabel.setText("Loading model on server...")
            self.progressBar.show()
            self.progressBar.setValue(50)
            slicer.app.processEvents()
            
            result = self.logic.load_model(
                server_url=server_url,
                model_name=model_name,
                backend_type=backend.lower(),
                config_path=config_path,
                model_path=model_path,
                inference_config=inference_config
            )
            
            self.progressBar.setValue(100)
            self.progressBar.hide()
            
            if result:
                self.statusLabel.setText(f"Model '{model_name}' loaded successfully")
                # Refresh models list
                self.onConnectButton()
            else:
                self.statusLabel.setText("Failed to load model")
        
        except Exception as e:
            self.progressBar.hide()
            self.statusLabel.setText(f"Error: {str(e)}")
            slicer.util.errorDisplay(f"Failed to load model: {str(e)}")

    def onUnloadModelButton(self):
        """Unload a model from the server."""
        try:
            server_url = self.serverUrlLineEdit.text.strip()
            model_name = self.inferenceModelSelector.currentText
            
            if not model_name:
                slicer.util.errorDisplay("Please select a model to unload")
                return
            
            result = self.logic.unload_model(server_url, model_name)
            
            if result:
                self.statusLabel.setText(f"Model '{model_name}' unloaded")
                # Refresh models list
                self.onConnectButton()
            else:
                self.statusLabel.setText("Failed to unload model")
        
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to unload model: {str(e)}")

    def onApplyButton(self):
        """Run processing when user clicks "Apply" button."""
        try:
            self.applyButton.enabled = False
            self.progressBar.show()
            self.progressBar.setValue(10)
            self.statusLabel.setText("Preparing for inference...")
            slicer.app.processEvents()

            # Get parameters
            server_url = self.serverUrlLineEdit.text.strip()
            model_name = self.inferenceModelSelector.currentText
            inputVolume = self.inputSelector.currentNode()
            force_cpu = self.forceCpuCheckBox.isChecked()

            if not inputVolume:
                slicer.util.errorDisplay("Please select an input volume")
                return

            if not model_name:
                slicer.util.errorDisplay("Please select a model")
                return

            self.progressBar.setValue(30)
            self.statusLabel.setText("Running inference on server...")
            slicer.app.processEvents()

            # Run inference
            segmentationNode = self.logic.run_inference(
                server_url=server_url,
                model_name=model_name,
                input_volume=inputVolume,
                output_segmentation=self.outputSelector.currentNode(),
                force_cpu=force_cpu
            )

            self.progressBar.setValue(100)
            self.statusLabel.setText("Inference completed successfully!")
            
            # Set output if it wasn't set before
            if self.outputSelector.currentNode() is None:
                self.outputSelector.setCurrentNode(segmentationNode)

        except Exception as e:
            slicer.util.errorDisplay(f"Inference failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.statusLabel.setText(f"Error: {str(e)}")
        finally:
            self.applyButton.enabled = True
            self.progressBar.hide()
            slicer.app.processEvents()


#
# ITKITInferenceLogic
#


class ITKITInferenceLogic(ScriptedLoadableModuleLogic):
    """This class implements the client logic for communicating with ITKIT server."""

    def __init__(self) -> None:
        """Called when the logic class is instantiated."""
        ScriptedLoadableModuleLogic.__init__(self)

    def get_server_info(self, server_url: str) -> dict:
        """Get server information.
        
        Args:
            server_url: URL of the ITKIT server
            
        Returns:
            Dictionary with server info or None if failed
        """
        try:
            response = requests.get(f"{server_url}/api/info", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Failed to get server info: {e}")
            return None

    def load_model(self, server_url: str, model_name: str, backend_type: str,
                   config_path: Optional[str], model_path: str,
                   inference_config: dict) -> bool:
        """Load a model on the server.
        
        Args:
            server_url: URL of the ITKIT server
            model_name: Name for the model
            backend_type: 'mmengine' or 'onnx'
            config_path: Path to config file (optional)
            model_path: Path to model/checkpoint file
            inference_config: Inference configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'name': model_name,
                'backend_type': backend_type,
                'model_path': model_path,
                'inference_config': inference_config
            }
            
            if config_path:
                data['config_path'] = config_path
            
            response = requests.post(f"{server_url}/api/models", json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            return result.get('status') == 'success'
        
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def unload_model(self, server_url: str, model_name: str) -> bool:
        """Unload a model from the server.
        
        Args:
            server_url: URL of the ITKIT server
            model_name: Name of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.delete(f"{server_url}/api/models/{model_name}", timeout=10)
            response.raise_for_status()
            result = response.json()
            
            return result.get('status') == 'success'
        
        except Exception as e:
            logging.error(f"Failed to unload model: {e}")
            return False

    def run_inference(self, server_url: str, model_name: str,
                     input_volume, output_segmentation,
                     force_cpu: bool = False) -> slicer.vtkMRMLSegmentationNode:
        """Run inference on the server.
        
        Args:
            server_url: URL of the ITKIT server
            model_name: Name of the model to use
            input_volume: Input scalar volume node
            output_segmentation: Output segmentation node (can be None)
            force_cpu: Force CPU accumulation on server
            
        Returns:
            Output segmentation node
        """
        import time
        startTime = time.time()
        
        # Export input volume to temporary file
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp_input_path = tmp.name
        
        slicer.util.saveNode(input_volume, tmp_input_path)
        
        try:
            # Prepare request
            with open(tmp_input_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'model_name': model_name,
                    'force_cpu': str(force_cpu).lower()
                }
                
                # Send inference request
                logging.info(f"Sending inference request to {server_url}")
                response = requests.post(
                    f"{server_url}/api/infer",
                    files=files,
                    data=data,
                    timeout=600  # 10 minutes timeout for inference
                )
                response.raise_for_status()
            
            # Save segmentation result to temporary file
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
                tmp_output_path = tmp.name
                tmp.write(response.content)
            
            # Load segmentation
            segLabelMapNode = slicer.util.loadLabelVolume(tmp_output_path)
            
            # Create or update output segmentation node
            if output_segmentation is None:
                output_segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                output_segmentation.SetName(input_volume.GetName() + "_Segmentation")
            
            # Set reference to input volume
            output_segmentation.SetReferenceImageGeometryParameterFromVolumeNode(input_volume)
            
            # Import labelmap to segmentation
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                segLabelMapNode,
                output_segmentation
            )
            
            # Clean up temporary nodes and files
            slicer.mrmlScene.RemoveNode(segLabelMapNode)
            os.unlink(tmp_output_path)
            
            stopTime = time.time()
            logging.info(f"Inference completed in {stopTime-startTime:.2f} seconds")
            
            return output_segmentation
        
        finally:
            # Clean up input temporary file
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)


#
# ITKITInferenceTest
#


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
