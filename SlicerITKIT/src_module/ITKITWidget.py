import logging

import ctk
import qt
import slicer
from ITKITLogic import ITKITLogic
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin

LOGGER = logging.getLogger(__name__)


class ITKITWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """UI layer for ITKIT inference module."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.matchedSeriesPairs = []

    def setup(self) -> None:
        super().setup()

        self.logic = ITKITLogic()
        LOGGER.info("ITKIT UI setup started")

        serverCollapsibleButton = ctk.ctkCollapsibleButton()
        serverCollapsibleButton.text = "Server Connection"
        self.layout.addWidget(serverCollapsibleButton)
        serverFormLayout = qt.QFormLayout(serverCollapsibleButton)

        self.serverUrlLineEdit = qt.QLineEdit()
        settings = qt.QSettings()
        self.serverUrlLineEdit.setText(
            settings.value("ITKIT/serverUrl", "http://localhost:8000")
        )
        self.serverUrlLineEdit.setToolTip("URL of the ITKIT Inference Server")
        serverFormLayout.addRow("Server URL:", self.serverUrlLineEdit)

        self.connectButton = qt.QPushButton("Connect to Server")
        self.connectButton.toolTip = "Test connection to ITKIT server"
        self.connectButton.clicked.connect(self.onConnectButton)
        serverFormLayout.addRow(self.connectButton)

        self.serverStatusLabel = qt.QLabel("Not connected")
        self.serverStatusLabel.setStyleSheet("color: gray;")
        serverFormLayout.addRow("Status:", self.serverStatusLabel)

        modelCollapsibleButton = ctk.ctkCollapsibleButton()
        modelCollapsibleButton.text = "Model Configuration"
        self.layout.addWidget(modelCollapsibleButton)
        modelFormLayout = qt.QFormLayout(modelCollapsibleButton)

        self.backendSelector = qt.QComboBox()
        self.backendSelector.addItem("MMEngine")
        self.backendSelector.addItem("ONNX")
        self.backendSelector.setToolTip("Select the inference backend type")
        self.backendSelector.currentIndexChanged.connect(self.onBackendChanged)
        modelFormLayout.addRow("Backend Type:", self.backendSelector)

        self.configPathLineEdit = qt.QLineEdit()
        self.configPathLineEdit.setToolTip("Path to the model configuration file (.py)")
        self.configBrowseButton = qt.QPushButton("Browse...")
        self.configBrowseButton.clicked.connect(self.onConfigBrowseClicked)
        configLayout = qt.QHBoxLayout()
        configLayout.addWidget(self.configPathLineEdit)
        configLayout.addWidget(self.configBrowseButton)
        modelFormLayout.addRow("Config File:", configLayout)

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

        self.fp16CheckBox = qt.QCheckBox()
        self.fp16CheckBox.setToolTip("Use FP16 precision for inference")
        self.fp16CheckBox.setChecked(False)
        modelFormLayout.addRow("Use FP16:", self.fp16CheckBox)

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

        self.modelStatusLabel = qt.QLabel("No model loaded")
        self.modelStatusLabel.setStyleSheet("color: gray;")
        modelFormLayout.addRow("Current Model:", self.modelStatusLabel)

        pairLoadingCollapsibleButton = ctk.ctkCollapsibleButton()
        pairLoadingCollapsibleButton.text = "Load Image-Label Pairs"
        self.layout.addWidget(pairLoadingCollapsibleButton)
        pairLoadingFormLayout = qt.QFormLayout(pairLoadingCollapsibleButton)

        self.imageFolderLineEdit = qt.QLineEdit()
        self.imageFolderLineEdit.setText(settings.value("ITKIT/imageFolder", ""))
        self.imageFolderLineEdit.setToolTip(
            "Folder that contains image files following the ITKIT dataset structure"
        )
        self.imageFolderBrowseButton = qt.QPushButton("Browse...")
        self.imageFolderBrowseButton.clicked.connect(self.onImageFolderBrowseClicked)
        imageFolderLayout = qt.QHBoxLayout()
        imageFolderLayout.addWidget(self.imageFolderLineEdit)
        imageFolderLayout.addWidget(self.imageFolderBrowseButton)
        pairLoadingFormLayout.addRow("Image Folder:", imageFolderLayout)

        self.labelFolderLineEdit = qt.QLineEdit()
        self.labelFolderLineEdit.setText(settings.value("ITKIT/labelFolder", ""))
        self.labelFolderLineEdit.setToolTip(
            "Folder that contains label files with filenames matching the image folder"
        )
        self.labelFolderBrowseButton = qt.QPushButton("Browse...")
        self.labelFolderBrowseButton.clicked.connect(self.onLabelFolderBrowseClicked)
        labelFolderLayout = qt.QHBoxLayout()
        labelFolderLayout.addWidget(self.labelFolderLineEdit)
        labelFolderLayout.addWidget(self.labelFolderBrowseButton)
        pairLoadingFormLayout.addRow("Label Folder:", labelFolderLayout)

        self.scanPairsButton = qt.QPushButton("Scan Pairs")
        self.scanPairsButton.toolTip = (
            "Find image-label pairs whose filenames match exactly in both folders"
        )
        self.scanPairsButton.clicked.connect(self.onScanPairsButton)
        pairLoadingFormLayout.addRow(self.scanPairsButton)

        self.pairSummaryLabel = qt.QLabel("No series scanned")
        self.pairSummaryLabel.setStyleSheet("color: gray;")
        pairLoadingFormLayout.addRow("Matched Series:", self.pairSummaryLabel)

        self.seriesListWidget = qt.QListWidget()
        self.seriesListWidget.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.seriesListWidget.setToolTip(
            "Select one matched image-label pair to load into Slicer"
        )
        self.seriesListWidget.itemSelectionChanged.connect(
            self.updatePairSelectionState
        )
        pairLoadingFormLayout.addRow("Series List:", self.seriesListWidget)

        self.loadPairButton = qt.QPushButton("Load Selected Pair")
        self.loadPairButton.toolTip = (
            "Load the selected image volume and convert its label to a segmentation"
        )
        self.loadPairButton.clicked.connect(self.onLoadPairButton)
        self.loadPairButton.enabled = False
        pairLoadingFormLayout.addRow(self.loadPairButton)

        self.pairStatusLabel = qt.QLabel()
        self.pairStatusLabel.setWordWrap(True)
        pairLoadingFormLayout.addRow(self.pairStatusLabel)

        # Reorder sections so Load-related controls appear before Model Configuration.
        self.layout.insertWidget(1, pairLoadingCollapsibleButton)

        inferenceCollapsibleButton = ctk.ctkCollapsibleButton()
        inferenceCollapsibleButton.text = "Run Inference"
        self.layout.addWidget(inferenceCollapsibleButton)
        inferenceFormLayout = qt.QFormLayout(inferenceCollapsibleButton)

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

        self.patchSizeLineEdit = qt.QLineEdit()
        self.patchSizeLineEdit.setPlaceholderText("e.g., 96,96,96 (optional)")
        self.patchSizeLineEdit.setToolTip("Sliding window patch size")
        inferenceFormLayout.addRow("Patch Size:", self.patchSizeLineEdit)

        self.patchStrideLineEdit = qt.QLineEdit()
        self.patchStrideLineEdit.setPlaceholderText("e.g., 48,48,48 (optional)")
        self.patchStrideLineEdit.setToolTip("Sliding window stride")
        inferenceFormLayout.addRow("Patch Stride:", self.patchStrideLineEdit)

        self.forceCpuCheckBox = qt.QCheckBox()
        self.forceCpuCheckBox.setToolTip("Force CPU accumulation on server")
        self.forceCpuCheckBox.setChecked(False)
        inferenceFormLayout.addRow("Force CPU:", self.forceCpuCheckBox)

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

        self.applyButton = qt.QPushButton("Run Inference")
        self.applyButton.toolTip = "Run the inference on the server"
        self.applyButton.enabled = False
        inferenceFormLayout.addRow(self.applyButton)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.hide()
        inferenceFormLayout.addRow(self.progressBar)

        self.statusLabel = qt.QLabel()
        inferenceFormLayout.addRow(self.statusLabel)

        self.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.serverUrlLineEdit.textChanged.connect(self.onServerUrlChanged)

        self.layout.addStretch(1)

        self.onBackendChanged()
        self.updatePairSelectionState()
        LOGGER.info("ITKIT UI setup completed")

    def onImageFolderBrowseClicked(self):
        folderPath = qt.QFileDialog.getExistingDirectory(
            self.parent,
            "Select Image Folder",
            self.imageFolderLineEdit.text.strip(),
        )
        if folderPath:
            settings = qt.QSettings()
            settings.setValue("ITKIT/imageFolder", folderPath)
            self.imageFolderLineEdit.setText(folderPath)

    def onLabelFolderBrowseClicked(self):
        folderPath = qt.QFileDialog.getExistingDirectory(
            self.parent,
            "Select Label Folder",
            self.labelFolderLineEdit.text.strip(),
        )
        if folderPath:
            settings = qt.QSettings()
            settings.setValue("ITKIT/labelFolder", folderPath)
            self.labelFolderLineEdit.setText(folderPath)

    def currentPairRow(self) -> int:
        currentItem = self.seriesListWidget.currentItem()
        if currentItem is None:
            return -1
        return self.seriesListWidget.row(currentItem)

    def onScanPairsButton(self):
        image_folder = self.imageFolderLineEdit.text.strip()
        label_folder = self.labelFolderLineEdit.text.strip()

        settings = qt.QSettings()
        settings.setValue("ITKIT/imageFolder", image_folder)
        settings.setValue("ITKIT/labelFolder", label_folder)

        try:
            self.pairStatusLabel.setText("Scanning image-label pairs...")
            slicer.app.processEvents()

            matched_pairs = self.logic.scan_image_label_pairs(
                image_folder=image_folder,
                label_folder=label_folder,
            )

            self.matchedSeriesPairs = matched_pairs
            self.seriesListWidget.clear()
            for pair in matched_pairs:
                item = qt.QListWidgetItem(pair["series_name"])
                item.setToolTip(
                    f"Image: {pair['image_path']}\nLabel: {pair['label_path']}"
                )
                self.seriesListWidget.addItem(item)

            match_count = len(matched_pairs)
            if match_count == 0:
                self.pairSummaryLabel.setText("0 matched series")
                self.pairSummaryLabel.setStyleSheet("color: orange;")
                self.pairStatusLabel.setText(
                    "No matched filenames were found between the two folders"
                )
            else:
                self.pairSummaryLabel.setText(f"{match_count} matched series")
                self.pairSummaryLabel.setStyleSheet("color: green;")
                self.seriesListWidget.setCurrentRow(0)
                self.pairStatusLabel.setText(
                    "Select one matched series and click Load Selected Pair"
                )

            self.updatePairSelectionState()

        except Exception as e:
            LOGGER.exception("Pair scanning failed")
            self.matchedSeriesPairs = []
            self.seriesListWidget.clear()
            self.pairSummaryLabel.setText("Scan failed")
            self.pairSummaryLabel.setStyleSheet("color: red;")
            self.pairStatusLabel.setText(f"Error: {str(e)}")
            self.updatePairSelectionState()
            slicer.util.errorDisplay(f"Failed to scan folders: {str(e)}")

    def updatePairSelectionState(self):
        self.loadPairButton.enabled = self.currentPairRow() >= 0

    def onLoadPairButton(self):
        selected_row = self.currentPairRow()
        if selected_row < 0 or selected_row >= len(self.matchedSeriesPairs):
            slicer.util.errorDisplay("Please select one matched series to load")
            return

        pair = self.matchedSeriesPairs[selected_row]
        self.loadPairButton.enabled = False
        self.pairStatusLabel.setText(f"Loading series: {pair['series_name']}")
        slicer.app.processEvents()

        try:
            image_node, segmentation_node = self.logic.load_image_label_pair(
                image_path=pair["image_path"],
                label_path=pair["label_path"],
                series_name=pair["series_name"],
            )
            self.inputSelector.setCurrentNode(image_node)
            self.outputSelector.setCurrentNode(segmentation_node)
            self.pairStatusLabel.setText(
                f"Loaded {pair['series_name']} into volume and segmentation nodes"
            )
        except Exception as e:
            LOGGER.exception("Pair loading failed")
            self.pairStatusLabel.setText(f"Error: {str(e)}")
            slicer.util.errorDisplay(f"Failed to load selected pair: {str(e)}")
        finally:
            self.updatePairSelectionState()

    def cleanup(self) -> None:
        pass

    def onServerUrlChanged(self):
        settings = qt.QSettings()
        settings.setValue("ITKIT/serverUrl", self.serverUrlLineEdit.text)
        LOGGER.info("Server URL updated")

        self.serverStatusLabel.setText("Not connected")
        self.serverStatusLabel.setStyleSheet("color: gray;")
        self.loadModelButton.enabled = False
        self.applyButton.enabled = False

    def onConnectButton(self):
        server_url = self.serverUrlLineEdit.text.strip()
        LOGGER.info("Connecting to server: %s", server_url)

        try:
            self.statusLabel.setText("Connecting to server...")
            slicer.app.processEvents()

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
        backend = self.backendSelector.currentText
        LOGGER.info("Backend changed to: %s", backend)
        if backend == "MMEngine":
            self.configPathLineEdit.setEnabled(True)
            self.configBrowseButton.setEnabled(True)
        else:
            self.configPathLineEdit.setEnabled(False)
            self.configBrowseButton.setEnabled(False)

    def onConfigBrowseClicked(self):
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
        try:
            server_url = self.serverUrlLineEdit.text.strip()
            backend = self.backendSelector.currentText
            config_path = (
                self.configPathLineEdit.text.strip() if backend == "MMEngine" else None
            )
            model_path = self.modelPathLineEdit.text.strip()

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

            inference_config = {"fp16": self.fp16CheckBox.isChecked()}

            LOGGER.info(
                "Loading model (backend=%s, fp16=%s)", backend, inference_config["fp16"]
            )
            LOGGER.debug("Model paths: config=%s model=%s", config_path, model_path)
            LOGGER.debug("Inference config: %s", inference_config)

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
        try:
            server_url = self.serverUrlLineEdit.text.strip()

            result = self.logic.unload_model(server_url)

            if result:
                LOGGER.info("Model unloaded")
                self.statusLabel.setText("Model unloaded")
                self.onConnectButton()
            else:
                LOGGER.warning("Model unload failed")
                self.statusLabel.setText("Failed to unload model")

        except Exception as e:
            LOGGER.exception("Unload model failed")
            slicer.util.errorDisplay(f"Failed to unload model: {str(e)}")

    def onApplyButton(self):
        self.applyButton.enabled = False
        self.progressBar.show()
        self.progressBar.setValue(10)
        self.statusLabel.setText("Preparing for inference...")

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

                def onComplete():
                    self.progressBar.setValue(100)
                    self.statusLabel.setText("Inference completed successfully!")
                    if self.outputSelector.currentNode() is None:
                        self.outputSelector.setCurrentNode(segmentationNode)
                    self.applyButton.enabled = True
                    self.progressBar.hide()

                qt.QTimer.singleShot(0, onComplete)

            except Exception as e:
                LOGGER.exception("Inference failed")

                def onError():
                    slicer.util.errorDisplay(f"Inference failed: {str(e)}")
                    self.statusLabel.setText(f"Error: {str(e)}")
                    self.applyButton.enabled = True
                    self.progressBar.hide()

                qt.QTimer.singleShot(0, onError)

        qt.QTimer.singleShot(100, runInference)
