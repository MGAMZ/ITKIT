"""
3D Slicer Extension for ITKIT Inference

This module provides a 3D Slicer interface for running ITKIT inference on medical images.
Users can specify configuration files, checkpoint paths, and inference parameters to
initialize the inferencer and perform segmentation on loaded volumes.
"""

import logging
import os
from typing import Optional

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

try:
    import numpy as np
    import torch
    from itkit.mm.inference import (
        MMEngineInferBackend,
        ONNXInferBackend,
        Inferencer_Seg3D,
        InferenceConfig,
    )
    ITKIT_AVAILABLE = True
except ImportError:
    ITKIT_AVAILABLE = False


#
# ITKITInference
#


class ITKITInference(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ITKIT Inference"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["ITKIT Team"]
        self.parent.helpText = """
This module provides an interface for running ITKIT inference on 3D medical images.
It supports both MMEngine and ONNX backends for deep learning model inference.

To use:
1. Load a 3D volume in Slicer
2. Select the backend type (MMEngine or ONNX)
3. Specify the configuration file and checkpoint/model path
4. (Optional) Configure inference parameters
5. Click 'Run Inference' to generate segmentation
"""
        self.parent.acknowledgementText = """
This module was developed using the ITKIT framework.
For more information, visit: https://github.com/MGAMZ/ITKIT
"""


#
# ITKITInferenceWidget
#


class ITKITInferenceWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Check if ITKIT is available
        if not ITKIT_AVAILABLE:
            errorLabel = qt.QLabel(
                "ERROR: ITKIT is not installed or cannot be imported.\n"
                "Please install ITKIT with: pip install itkit[advanced]\n"
                "And ensure PyTorch is installed."
            )
            errorLabel.setStyleSheet("color: red; font-weight: bold; padding: 10px;")
            self.layout.addWidget(errorLabel)
            return

        # Load widget from .ui file (create widgets manually)
        
        # Create logic
        self.logic = ITKITInferenceLogic()

        # Connections section
        connectionsCollapsibleButton = ctk.ctkCollapsibleButton()
        connectionsCollapsibleButton.text = "Backend Configuration"
        self.layout.addWidget(connectionsCollapsibleButton)
        connectionsFormLayout = qt.QFormLayout(connectionsCollapsibleButton)

        # Backend type selector
        self.backendSelector = qt.QComboBox()
        self.backendSelector.addItem("MMEngine")
        self.backendSelector.addItem("ONNX")
        self.backendSelector.setToolTip("Select the inference backend type")
        connectionsFormLayout.addRow("Backend Type:", self.backendSelector)

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
        connectionsFormLayout.addRow("Input Volume:", self.inputSelector)

        # Config file path (for MMEngine)
        self.configPathLineEdit = qt.QLineEdit()
        self.configPathLineEdit.setToolTip("Path to the model configuration file (.py)")
        self.configBrowseButton = qt.QPushButton("Browse...")
        self.configBrowseButton.clicked.connect(self.onConfigBrowseClicked)
        configLayout = qt.QHBoxLayout()
        configLayout.addWidget(self.configPathLineEdit)
        configLayout.addWidget(self.configBrowseButton)
        connectionsFormLayout.addRow("Config File:", configLayout)

        # Checkpoint/Model path
        self.checkpointPathLineEdit = qt.QLineEdit()
        self.checkpointPathLineEdit.setToolTip("Path to checkpoint (.pth) or ONNX model (.onnx)")
        self.checkpointBrowseButton = qt.QPushButton("Browse...")
        self.checkpointBrowseButton.clicked.connect(self.onCheckpointBrowseClicked)
        checkpointLayout = qt.QHBoxLayout()
        checkpointLayout.addWidget(self.checkpointPathLineEdit)
        checkpointLayout.addWidget(self.checkpointBrowseButton)
        connectionsFormLayout.addRow("Checkpoint/Model:", checkpointLayout)

        # Parameters section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Inference Parameters"
        parametersCollapsibleButton.collapsed = True
        self.layout.addWidget(parametersCollapsibleButton)
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # Patch size
        self.patchSizeLineEdit = qt.QLineEdit()
        self.patchSizeLineEdit.setToolTip(
            "Sliding window patch size as comma-separated values (e.g., '96,96,96'). "
            "Leave empty for full volume inference."
        )
        self.patchSizeLineEdit.setPlaceholderText("e.g., 96,96,96 (optional)")
        parametersFormLayout.addRow("Patch Size:", self.patchSizeLineEdit)

        # Patch stride
        self.patchStrideLineEdit = qt.QLineEdit()
        self.patchStrideLineEdit.setToolTip(
            "Sliding window stride as comma-separated values (e.g., '48,48,48'). "
            "Leave empty to use half of patch size."
        )
        self.patchStrideLineEdit.setPlaceholderText("e.g., 48,48,48 (optional)")
        parametersFormLayout.addRow("Patch Stride:", self.patchStrideLineEdit)

        # FP16 checkbox
        self.fp16CheckBox = qt.QCheckBox()
        self.fp16CheckBox.setToolTip("Use FP16 precision for inference (faster, less memory)")
        self.fp16CheckBox.setChecked(False)
        parametersFormLayout.addRow("Use FP16:", self.fp16CheckBox)

        # Force CPU checkbox
        self.forceCpuCheckBox = qt.QCheckBox()
        self.forceCpuCheckBox.setToolTip("Force accumulation on CPU to avoid GPU OOM")
        self.forceCpuCheckBox.setChecked(False)
        parametersFormLayout.addRow("Force CPU Accumulation:", self.forceCpuCheckBox)

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
        parametersFormLayout.addRow("Output Segmentation:", self.outputSelector)

        # Apply Button
        self.applyButton = qt.QPushButton("Run Inference")
        self.applyButton.toolTip = "Run the inference algorithm"
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # Progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.hide()
        parametersFormLayout.addRow(self.progressBar)

        # Status label
        self.statusLabel = qt.QLabel()
        parametersFormLayout.addRow(self.statusLabel)

        # Connections
        self.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.backendSelector.connect("currentIndexChanged(int)", self.onBackendChanged)

        # Add vertical spacer
        self.layout.addStretch(1)

        # Initial GUI update
        self.updateParameterNodeFromGUI()
        self.onBackendChanged()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        pass

    def onBackendChanged(self) -> None:
        """Called when backend type is changed."""
        backend = self.backendSelector.currentText
        # Config file is only needed for MMEngine
        if backend == "MMEngine":
            self.configPathLineEdit.setEnabled(True)
            self.configBrowseButton.setEnabled(True)
        else:
            self.configPathLineEdit.setEnabled(False)
            self.configBrowseButton.setEnabled(False)
        self.updateParameterNodeFromGUI()

    def onConfigBrowseClicked(self) -> None:
        """Called when config browse button is clicked."""
        filePath = qt.QFileDialog.getOpenFileName(
            self.parent,
            "Select Configuration File",
            "",
            "Python Files (*.py);;All Files (*)"
        )
        if filePath:
            self.configPathLineEdit.setText(filePath)
            self.updateParameterNodeFromGUI()

    def onCheckpointBrowseClicked(self) -> None:
        """Called when checkpoint browse button is clicked."""
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
            self.checkpointPathLineEdit.setText(filePath)
            self.updateParameterNodeFromGUI()

    def updateParameterNodeFromGUI(self, caller=None, event=None) -> None:
        """Update apply button state based on input validity."""
        backend = self.backendSelector.currentText
        inputVolume = self.inputSelector.currentNode()
        checkpointPath = self.checkpointPathLineEdit.text
        
        if backend == "MMEngine":
            configPath = self.configPathLineEdit.text
            isValid = (inputVolume is not None and 
                      configPath and os.path.exists(configPath) and
                      checkpointPath and os.path.exists(checkpointPath))
        else:  # ONNX
            isValid = (inputVolume is not None and 
                      checkpointPath and os.path.exists(checkpointPath))
        
        self.applyButton.enabled = isValid

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        try:
            self.applyButton.enabled = False
            self.progressBar.show()
            self.progressBar.setValue(10)
            self.statusLabel.text = "Initializing..."
            slicer.app.processEvents()

            # Get parameters from GUI
            backend = self.backendSelector.currentText
            inputVolume = self.inputSelector.currentNode()
            configPath = self.configPathLineEdit.text if backend == "MMEngine" else None
            checkpointPath = self.checkpointPathLineEdit.text
            fp16 = self.fp16CheckBox.isChecked()
            forceCpu = self.forceCpuCheckBox.isChecked()
            
            # Parse patch size and stride
            patchSizeText = self.patchSizeLineEdit.text.strip()
            patchStrideText = self.patchStrideLineEdit.text.strip()
            
            patchSize = None
            patchStride = None
            if patchSizeText:
                try:
                    patchSize = tuple(int(x.strip()) for x in patchSizeText.split(','))
                except ValueError:
                    slicer.util.errorDisplay("Invalid patch size format. Use comma-separated integers.")
                    return
            
            if patchStrideText:
                try:
                    patchStride = tuple(int(x.strip()) for x in patchStrideText.split(','))
                except ValueError:
                    slicer.util.errorDisplay("Invalid patch stride format. Use comma-separated integers.")
                    return

            self.progressBar.setValue(30)
            self.statusLabel.text = "Running inference..."
            slicer.app.processEvents()

            # Run inference
            segmentationNode = self.logic.process(
                inputVolume=inputVolume,
                outputSegmentation=self.outputSelector.currentNode(),
                backend=backend,
                configPath=configPath,
                checkpointPath=checkpointPath,
                patchSize=patchSize,
                patchStride=patchStride,
                fp16=fp16,
                forceCpu=forceCpu,
                progressCallback=self.updateProgress
            )

            self.progressBar.setValue(100)
            self.statusLabel.text = "Inference completed successfully!"
            
            # Set output if it wasn't set before
            if self.outputSelector.currentNode() is None:
                self.outputSelector.setCurrentNode(segmentationNode)

        except Exception as e:
            slicer.util.errorDisplay(f"Inference failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.statusLabel.text = f"Error: {str(e)}"
        finally:
            self.applyButton.enabled = True
            slicer.app.processEvents()

    def updateProgress(self, progress: int, message: str = "") -> None:
        """Update progress bar and status label."""
        self.progressBar.setValue(progress)
        if message:
            self.statusLabel.text = message
        slicer.app.processEvents()


#
# ITKITInferenceLogic
#


class ITKITInferenceLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual computation done by your module.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def process(self,
                inputVolume,
                outputSegmentation,
                backend: str,
                configPath: Optional[str],
                checkpointPath: str,
                patchSize: Optional[tuple],
                patchStride: Optional[tuple],
                fp16: bool = False,
                forceCpu: bool = False,
                progressCallback=None) -> slicer.vtkMRMLSegmentationNode:
        """
        Run the inference algorithm.
        
        Args:
            inputVolume: Input scalar volume node
            outputSegmentation: Output segmentation node (can be None)
            backend: Backend type ("MMEngine" or "ONNX")
            configPath: Path to config file (for MMEngine)
            checkpointPath: Path to checkpoint/model file
            patchSize: Optional patch size for sliding window
            patchStride: Optional patch stride for sliding window
            fp16: Whether to use FP16 precision
            forceCpu: Whether to force CPU accumulation
            progressCallback: Optional callback for progress updates
            
        Returns:
            Output segmentation node
        """

        if not ITKIT_AVAILABLE:
            raise ImportError("ITKIT is not available. Please install with: pip install itkit[advanced]")

        import time
        startTime = time.time()
        
        if progressCallback:
            progressCallback(40, "Extracting volume data...")

        # Get numpy array from input volume
        inputArray = slicer.util.arrayFromVolume(inputVolume)
        
        # ITKIT expects (Z, Y, X) order
        # Slicer arrays are (Z, Y, X) by default when using arrayFromVolume
        
        if progressCallback:
            progressCallback(50, "Initializing backend...")

        # Create inference config
        # If forceCpu is True, use CPU for accumulation to avoid GPU OOM
        # Otherwise, check if CUDA is available and use it, else fall back to CPU
        if forceCpu:
            accumulate_device = 'cpu'
        else:
            import torch
            accumulate_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        inferenceConfig = InferenceConfig(
            patch_size=patchSize,
            patch_stride=patchStride,
            accumulate_device=accumulate_device
        )

        # Initialize backend
        if backend == "MMEngine":
            backendInstance = MMEngineInferBackend(
                cfg_path=configPath,
                ckpt_path=checkpointPath,
                inference_config=inferenceConfig,
                allow_tqdm=False
            )
        else:  # ONNX
            backendInstance = ONNXInferBackend(
                onnx_path=checkpointPath,
                inference_config=inferenceConfig,
                allow_tqdm=False
            )

        if progressCallback:
            progressCallback(60, "Running inference...")

        # Create inferencer
        inferencer = Inferencer_Seg3D(
            backend=backendInstance,
            fp16=fp16,
            allow_tqdm=False
        )

        # Run inference
        seg_logits, sem_seg_map = inferencer.Inference_FromNDArray(inputArray)
        
        if progressCallback:
            progressCallback(80, "Converting results to segmentation...")

        # Convert to numpy and get segmentation map
        sem_seg_map_np = sem_seg_map.cpu().numpy()[0]  # Remove batch dimension

        # Create or update output segmentation node
        if outputSegmentation is None:
            outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            outputSegmentation.SetName(inputVolume.GetName() + "_Segmentation")
        
        # Set reference to input volume
        outputSegmentation.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        # Create a label map from the segmentation map
        # Get unique labels (excluding background which is 0)
        uniqueLabels = np.unique(sem_seg_map_np)
        uniqueLabels = uniqueLabels[uniqueLabels > 0]  # Exclude background

        # Clear existing segments
        outputSegmentation.GetSegmentation().RemoveAllSegments()

        # Add segments for each unique label
        for label in uniqueLabels:
            segmentId = f"Segment_{int(label)}"
            # Create binary mask for this label
            labelMask = (sem_seg_map_np == label).astype(np.uint8)
            
            # Create a temporary volume node for the label map
            labelVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"temp_label_{int(label)}")
            slicer.util.updateVolumeFromArray(labelVolumeNode, labelMask)
            labelVolumeNode.CopyOrientation(inputVolume)
            
            # Import the label map into the segmentation
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelVolumeNode,
                outputSegmentation
            )
            
            # Remove temporary volume
            slicer.mrmlScene.RemoveNode(labelVolumeNode)

        if progressCallback:
            progressCallback(95, "Finalizing...")

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

        return outputSegmentation


#
# ITKITInferenceTest
#


class ITKITInferenceTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
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
