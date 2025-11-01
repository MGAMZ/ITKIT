import os
import shlex
from dataclasses import dataclass
from PyQt6 import QtCore


def _default_shell() -> str:
    # Linux default per environment info; allow override via env
    return os.environ.get("SHELL", "/bin/bash")


@dataclass
class CmdChunk:
    program: str
    args: list[str]

    def to_string(self) -> str:
        return " ".join([shlex.quote(self.program), *map(shlex.quote, self.args)])


class ProcessRunner(QtCore.QObject):
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal(int)
    stdout = QtCore.pyqtSignal(str)
    stderr = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(float)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(
            QtCore.QProcess.ProcessChannelMode.SeparateChannels
        )
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.started.connect(self.started)
        self.proc.finished.connect(self._on_finished)
        # Basic progress hint from tqdm-like lines
        self._buffer = bytearray()

    def run(self, chunks: list[CmdChunk], workdir: str | None = None):
        # Join chunks with '&&' so later commands run only if previous succeed
        commands = [c.to_string() for c in chunks]
        joined = " && ".join(commands)
        shell = _default_shell()
        env = QtCore.QProcessEnvironment.systemEnvironment()
        self.proc.setProcessEnvironment(env)
        if workdir:
            self.proc.setWorkingDirectory(workdir)
        self.stdout.emit(f"$ {joined}\n")
        self.proc.start(shell, ["-lc", joined])

    def _on_stdout(self):
        data = bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")
        self.stdout.emit(data)
        self._emit_progress_if_tqdm(data)

    def _on_stderr(self):
        data = bytes(self.proc.readAllStandardError()).decode(errors="ignore")
        self.stderr.emit(data)
        self._emit_progress_if_tqdm(data)

    def _on_finished(self, code: int, _status: QtCore.QProcess.ExitStatus):
        self.finished.emit(code)

    def kill(self):
        if self.proc.state() != QtCore.QProcess.ProcessState.NotRunning:
            self.proc.kill()

    def _emit_progress_if_tqdm(self, text: str):
        # Heuristic: find `xx%` in line and emit max percent
        import re

        percents = [float(p[:-1]) for p in re.findall(r"(\d{1,3}%)", text)]
        if percents:
            self.progress.emit(min(100.0, max(percents)))
