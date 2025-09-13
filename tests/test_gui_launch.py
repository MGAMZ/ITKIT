import os
import sys
import pytest


def _ensure_offscreen():
    # Ensure headless-friendly platform for CI
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Avoid high DPI surprises in CI
    os.environ.setdefault("QT_SCALE_FACTOR", "1.0")


@pytest.mark.gui
def test_start_and_close_gui(monkeypatch):
    """Try to create QApplication and MainWindow and close it without entering exec loop.

    The test ensures the GUI can be instantiated in a headless (offscreen) environment.
    It does not show any interactive UI or require a display server.
    """
    _ensure_offscreen()

    # Ensure PyQt6 is available (skip test if not installed)
    pytest.importorskip("PyQt6", reason="PyQt6 not installed")

    # Import the app module under test
    from itkit.gui import app as itkit_app
    from PyQt6 import QtWidgets

    # Create application instance if none exists
    app = QtWidgets.QApplication.instance()
    created = False
    if app is None:
        app = QtWidgets.QApplication([])
        created = True

    # Instantiate main window and ensure show/close work
    win = itkit_app.MainWindow()
    # Show and immediately process events to let Qt initialize objects
    win.show()
    app.processEvents()

    # Perform some basic assertions about the widget
    assert win.windowTitle() == "ITKIT Preprocessing"
    assert win.isVisible()

    # Close and process events again
    win.close()
    app.processEvents()
    assert not win.isVisible()

    # If we created the QApplication instance, quit it to clean up
    if created:
        app.quit()
