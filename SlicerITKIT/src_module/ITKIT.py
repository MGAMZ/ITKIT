# pyright: reportUndefinedVariable=false
"""
3D Slicer Extension for ITKIT Inference (Client)

This module provides a lightweight 3D Slicer interface for running ITKIT inference.
It communicates with an ITKIT Inference Server via REST API, eliminating the need
to install ITKIT and PyTorch in Slicer's Python environment.
"""

import logging

import qt
import slicer
from ITKITLogic import ITKITLogic
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleTest

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)


class ITKIT(ScriptedLoadableModule):
    """ScriptedLoadableModule entry class for ITKIT."""

    def __init__(self, parent):
        super().__init__(parent)
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
        slicer.app.connect("startupCompleted()", self.initializeAfterStartup)

    def initializeAfterStartup(self):
        """Initialize module settings after application startup."""
        if not slicer.app.commandOptions().noMainWindow:
            settings = qt.QSettings()
            if not settings.contains("ITKIT/serverUrl"):
                settings.setValue("ITKIT/serverUrl", "http://localhost:8000")
            LOGGER.info("ITKIT settings initialized")


class ITKITTest(ScriptedLoadableModuleTest):
    """Test case for the scripted module."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ITKIT1()

    def test_ITKIT1(self):
        self.delayDisplay("Starting the test")
        logic = ITKITLogic()
        self.assertIsNotNone(logic)
        self.delayDisplay("Test passed")
