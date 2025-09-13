import os
import sys
import subprocess
import pytest


@pytest.mark.gui
def test_app_main_subprocess_smoke():
    """Spawn a subprocess that calls itkit.gui.app.main() and exits quickly.

    We patch QApplication.exec to schedule a quit soon, so the process won't block.
    This verifies that the entry function can initialize the GUI in a headless setup.
    """

    code = r"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtCore, QtWidgets
import itkit.gui.app as m

# Patch QApplication.exec to auto-quit shortly after entering the loop
_orig_exec = QtWidgets.QApplication.exec
def _patched_exec(self):
    QtCore.QTimer.singleShot(50, self.quit)
    # PyQt6 QApplication.exec is static and usually expects no args; be tolerant.
    try:
        return _orig_exec()
    except TypeError:
        return _orig_exec(self)
QtWidgets.QApplication.exec = _patched_exec

m.main()
"""

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Subprocess timed out while running GUI main()")

    stderr_lower = (proc.stderr or "").lower()
    if proc.returncode != 0 and "could not load the qt platform plugin" in stderr_lower:
        pytest.xfail("Qt platform plugin missing in environment; offscreen/minimal not available")

    assert proc.returncode == 0, (
        f"GUI main() subprocess failed with rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
