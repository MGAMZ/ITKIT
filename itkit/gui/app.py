import os, sys, signal
from PyQt6 import QtCore, QtGui, QtWidgets


def _setup_qt_platform():
    # Decide a sensible QT_QPA_PLATFORM when not provided, but do not return early
    # so we can always ensure QT_SCALE_FACTOR is set.
    if "QT_QPA_PLATFORM" not in os.environ:
        display = os.environ.get("DISPLAY")
        wayland = os.environ.get("WAYLAND_DISPLAY")
        chosen = None
        if wayland:
            runtime = os.environ.get("XDG_RUNTIME_DIR", "")
            sock = os.path.join(runtime, wayland) if runtime else ""
            if sock and os.path.exists(sock):
                chosen = "wayland"
        if chosen is None:
            if display:
                chosen = "xcb"
            else:
                chosen = "offscreen"
        os.environ["QT_QPA_PLATFORM"] = chosen

    # Ensure a sensible default scale factor if the user/environment didn't set one.
    # Do not rely on reading it before ensuring existence.
    if "QT_SCALE_FACTOR" not in os.environ:
        os.environ["QT_SCALE_FACTOR"] = "1.75"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ITKIT Preprocessing")
        self.resize(720, 540)
        self._build_ui()

    def _build_ui(self):
        # 推迟导入，确保 QApplication 已存在再构建子控件
        from .ItkTabs import (
            ItkCheckTab,
            ItkResampleTab,
            ItkOrientTab,
            ItkPatchTab,
            ItkAugTab,
            ItkExtractTab,
        )

        tabs = QtWidgets.QTabWidget()
        tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        tabs.addTab(ItkCheckTab(), "itk_check")
        tabs.addTab(ItkResampleTab(), "itk_resample")
        tabs.addTab(ItkOrientTab(), "itk_orient")
        tabs.addTab(ItkPatchTab(), "itk_patch")
        tabs.addTab(ItkAugTab(), "itk_aug")
        tabs.addTab(ItkExtractTab(), "itk_extract")
        self.setCentralWidget(tabs)


def main():
    # Enable HiDPI and better scaling before app creation
    _setup_qt_platform()
    # Prefer Fusion style for consistent look; fallback to platform default
    app = QtWidgets.QApplication(sys.argv)
    # Restore default SIGINT handler so Ctrl-C immediately terminates the process.
    # Without this, the Qt event loop may swallow SIGINT and prevent KeyboardInterrupt.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    scr = app.primaryScreen()
    if scr is not None:
        plug = app.platformName() if hasattr(app, "platformName") else "?"
        qsf = os.environ.get("QT_SCALE_FACTOR")
        qfd = os.environ.get("QT_FONT_DPI")
        print(
            f"[Qt] platform={plug} "
            f"[DPI] logicalDPI={scr.logicalDotsPerInch():.1f} "
            f"devicePixelRatio={scr.devicePixelRatio():.2f} "
            f"geometry={scr.geometry().width()}x{scr.geometry().height()} "
            f"QT_SCALE_FACTOR={qsf if qsf else 'not-set'} "
            f"QT_FONT_DPI={qfd if qfd else 'not-set'}"
        )

    if "Fusion" in QtWidgets.QStyleFactory.keys():
        app.setStyle("Fusion")
        # Optional dark palette that is easy on eyes
        dark = QtGui.QPalette()
        dark.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
        dark.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
        dark.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(35, 35, 35))
        dark.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
        dark.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
        dark.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
        dark.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
        dark.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
        dark.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
        dark.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
        dark.setColor(
            QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(142, 45, 197).lighter()
        )
        dark.setColor(
            QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black
        )
        app.setPalette(dark)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
