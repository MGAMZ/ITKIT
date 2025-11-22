from __future__ import annotations

import os

from PyQt6 import QtCore, QtGui, QtWidgets

from .runner import CmdChunk, ProcessRunner


def _style_lineedit_placeholder(line: QtWidgets.QLineEdit|QtWidgets.QPlainTextEdit,
                                placeholder_color: str = "#9d9d9d",
                                bg_color: str | None = None):
    if bg_color:
        # 可选：同时设置浅色背景以提高对比
        line.setStyleSheet(f"background-color: {bg_color};")

    pal = line.palette()
    pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor(placeholder_color))
    line.setPalette(pal)


class CommandFormBase(QtWidgets.QWidget):
    def __init__(self, title: str, parent: QtWidgets.QWidget|None = None):
        super().__init__(parent)
        self.title = title
        self.runner = ProcessRunner(self)
        self._build_ui()
        self._wire()

    # UI skeleton and shared widgets
    def _build_ui(self):
        self.setObjectName(self.title)
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        self.form = QtWidgets.QFormLayout()
        self.form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        main.addLayout(self.form)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Process")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        main.addLayout(btn_row)

        # Progress and log
        prog_row = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFormat("%p%")
        prog_row.addWidget(self.progress)
        main.addLayout(prog_row)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        main.addWidget(self.log)

    def _wire(self):
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_stop.clicked.connect(self.runner.kill)
        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)
        self.runner.stdout.connect(lambda s: self._append_text(s))
        self.runner.stderr.connect(lambda s: self._append_text(s))
        self.runner.progress.connect(self._on_progress)

    def _append_text(self, t: str):
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.log.insertPlainText(t)
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _on_progress(self, v: float):
        self.progress.setValue(int(v))

    def _on_started(self):
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

    def _on_finished(self, code: int):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0 and self.progress.value() < 100:
            self.progress.setValue(100)

    # Methods for subclasses to implement
    def build_chunks(self) -> list[CmdChunk]:
        raise NotImplementedError

    def _on_run_clicked(self):
        try:
            chunks = self.build_chunks()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Param Error", str(e))
            return
        self.log.clear()
        self.runner.run(chunks, workdir=os.getcwd())

    # Helper builders
    def _file_picker(self, button_text: str, mode: str = "file"):
        line = QtWidgets.QLineEdit()
        btn = QtWidgets.QPushButton(button_text)
        wrapper = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(line)
        lay.addWidget(btn)

        def pick():
            if mode == "dir":
                path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File")
            if path:
                line.setText(path)

        btn.clicked.connect(pick)
        return wrapper, line

    def _check(self, text: str, checked: bool = False) -> QtWidgets.QCheckBox:
        c = QtWidgets.QCheckBox(text)
        c.setChecked(checked)
        return c

    def _spin_int3(self, minimum: int, maximum: int, step: int = 1, placeholder: str = "Z Y X"):
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText(placeholder)
        w.setToolTip("Three integers seperated by space, -1 to ignore on its dimension")
        _style_lineedit_placeholder(w)
        return w

    def _spin_float3(self, placeholder: str = "Z Y X"):
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText(placeholder)
        w.setToolTip("Three floats seperated by space, -1 to ignore on its dimension")
        _style_lineedit_placeholder(w)
        return w

    @staticmethod
    def parse_triplet_int(text: str) -> list[int]:
        if not text.strip():
            return []
        parts = text.replace(",", " ").split()
        if len(parts) not in (0, 1, 3):
            raise ValueError("Triplet must be 1 or 3 integers, or empty")
        vals = list(map(int, parts))
        if len(vals) == 1:
            return [vals[0], vals[0], vals[0]]
        return vals

    @staticmethod
    def parse_triplet_float(text: str) -> list[float]:
        if not text.strip():
            return []
        parts = text.replace(",", " ").split()
        if len(parts) not in (0, 1, 3):
            raise ValueError("Triplet must be 1 or 3 floats, or empty")
        vals = list(map(float, parts))
        if len(vals) == 1:
            return [vals[0], vals[0], vals[0]]
        return vals


class ItkCheckTab(CommandFormBase):
    def __init__(self, parent=None):
        super().__init__("itk_check", parent)
        # Fill the form
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["check", "delete", "copy", "symlink"])
        self.form.addRow("mode", self.mode)

        srcw, self.sample_folder = self._file_picker("Browse", mode="dir")
        self.form.addRow("sample_folder", srcw)

        outw, self.out_dir = self._file_picker("Browse", mode="dir")
        self.form.addRow("output (-o)", outw)

        self.min_size = self._spin_int3(-1, 10**6)
        self.max_size = self._spin_int3(-1, 10**6)
        self.min_spacing = self._spin_float3()
        self.max_spacing = self._spin_float3()
        self.same_spacing = QtWidgets.QLineEdit()
        self.same_spacing.setPlaceholderText("A B, e.g. X Y")
        _style_lineedit_placeholder(self.same_spacing)
        self.same_size = QtWidgets.QLineEdit()
        self.same_size.setPlaceholderText("A B, e.g. X Y")
        _style_lineedit_placeholder(self.same_size)
        self.mp = self._check("Multi Processing")

        self.form.addRow("min-size (Z Y X)", self.min_size)
        self.form.addRow("max-size (Z Y X)", self.max_size)
        self.form.addRow("min-spacing (Z Y X)", self.min_spacing)
        self.form.addRow("max-spacing (Z Y X)", self.max_spacing)
        self.form.addRow("same-spacing", self.same_spacing)
        self.form.addRow("same-size", self.same_size)
        self.form.addRow(self.mp)

    def build_chunks(self) -> list[CmdChunk]:
        if not self.sample_folder.text().strip():
            raise ValueError("Please fill in sample_folder")
        args: list[str] = ["itk_check", self.mode.currentText(), self.sample_folder.text()]
        if self.out_dir.text():
            args += ["-o", self.out_dir.text()]
        for key, val in (
            ("--min-size", self.min_size.text()),
            ("--max-size", self.max_size.text()),
        ):
            trip_int = self.parse_triplet_int(val)
            if trip_int:
                args += [key, *map(str, trip_int)]
        for key, val in (
            ("--min-spacing", self.min_spacing.text()),
            ("--max-spacing", self.max_spacing.text()),
        ):
            trip_float = self.parse_triplet_float(val)
            if trip_float:
                args += [key, *map(str, trip_float)]
        if self.same_spacing.text().strip():
            parts = self.same_spacing.text().split()
            if len(parts) != 2:
                raise ValueError("same-spacing requires two dimensions, e.g. 'X Y'")
            args += ["--same-spacing", *parts]
        if self.same_size.text().strip():
            parts = self.same_size.text().split()
            if len(parts) != 2:
                raise ValueError("same-size requires two dimensions, e.g. 'X Y'")
            args += ["--same-size", *parts]
        if self.mp.isChecked():
            args.append("--mp")
        return [CmdChunk(program=args[0], args=args[1:])]


class ItkResampleTab(CommandFormBase):
    def __init__(self, parent=None):
        super().__init__("itk_resample", parent)
        self.field = QtWidgets.QComboBox()
        self.field.addItems(["image", "label"])
        self.form.addRow("field", self.field)
        srcw, self.src = self._file_picker("Browse", mode="dir")
        dstw, self.dst = self._file_picker("Browse", mode="dir")
        self.form.addRow("source_folder", srcw)
        self.form.addRow("dest_folder", dstw)

        self.spacing_input = self._spin_float3()
        self.size_input = self._spin_int3(-1, 10**6)
        tfw, target_folder = self._file_picker("Browse", mode="dir")
        self.target_folder: QtWidgets.QLineEdit = target_folder
        self.recursive = self._check("recursively process all samples")
        self.mp = self._check("Multi Processing")
        self.workers = QtWidgets.QSpinBox()
        self.workers.setRange(1, 512)
        self.workers.setValue(8)

        self.form.addRow("spacing (Z Y X)", self.spacing_input)
        self.form.addRow("size (Z Y X)", self.size_input)
        self.form.addRow("target-folder", tfw)
        self.form.addRow(self.recursive)
        self.form.addRow(self.mp)
        self.form.addRow("workers", self.workers)

    def build_chunks(self) -> list[CmdChunk]:
        if not self.src.text().strip():
            raise ValueError("Please fill in source_folder")
        if not self.dst.text().strip():
            raise ValueError("Please fill in dest_folder")
        args = [
            "itk_resample",
            self.field.currentText(),
            self.src.text(),
            self.dst.text(),
        ]
        spacing = self.parse_triplet_float(self.spacing_input.text())
        size = self.parse_triplet_int(self.size_input.text())
        if spacing and size:
            pass  # allow both; script may validate
        if spacing:
            args += ["--spacing", *map(str, spacing)]
        if size:
            args += ["--size", *map(str, size)]
        if self.target_folder.text():  # type: ignore[attr-defined]
            args += ["--target-folder", self.target_folder.text()]
        if self.recursive.isChecked():
            args.append("--recursive")
        if self.mp.isChecked():
            args.append("--mp")
        if self.workers.value() > 0:
            args += ["--workers", str(self.workers.value())]
        return [CmdChunk(program=args[0], args=args[1:])]


class ItkOrientTab(CommandFormBase):
    def __init__(self, parent=None):
        super().__init__("itk_orient", parent)
        srcw, self.src = self._file_picker("Browse", mode="dir")
        dstw, self.dst = self._file_picker("Browse", mode="dir")
        self.orient = QtWidgets.QLineEdit()
        self.orient.setPlaceholderText("e.g.: LPI")
        _style_lineedit_placeholder(self.orient)
        self.mp = self._check("Multi Processing")
        self.form.addRow("src_dir", srcw)
        self.form.addRow("dst_dir", dstw)
        self.form.addRow("orient", self.orient)
        self.form.addRow(self.mp)

    def build_chunks(self) -> list[CmdChunk]:
        if not self.src.text().strip():
            raise ValueError("Please fill in src_dir")
        if not self.dst.text().strip():
            raise ValueError("Please fill in dst_dir")
        if not self.orient.text().strip():
            raise ValueError("Please fill in orient, e.g. LPI")
        args = [
            "itk_orient",
            self.src.text(),
            self.dst.text(),
            self.orient.text(),
        ]
        if self.mp.isChecked():
            args.append("--mp")
        return [CmdChunk(program=args[0], args=args[1:])]


class ItkPatchTab(CommandFormBase):
    def __init__(self, parent=None):
        super().__init__("itk_patch", parent)
        srcw, self.src = self._file_picker("Browse", mode="dir")
        dstw, self.dst = self._file_picker("Browse", mode="dir")
        self.patch_size = self._spin_int3(1, 10**6, placeholder="PZ [PY PX]")
        self.patch_stride = self._spin_int3(1, 10**6, placeholder="SZ [SY SX]")
        self.min_fg = QtWidgets.QDoubleSpinBox()
        self.min_fg.setRange(0.0, 1.0)
        self.min_fg.setSingleStep(0.01)
        self.min_fg.setValue(0.0)
        self.still_save = self._check("Save samples without labels")
        self.mp = self._check("Multi Processing")

        self.form.addRow("src_folder", srcw)
        self.form.addRow("dst_folder", dstw)
        self.form.addRow("patch-size", self.patch_size)
        self.form.addRow("patch-stride", self.patch_stride)
        self.form.addRow("minimum-foreground-ratio", self.min_fg)
        self.form.addRow(self.still_save)
        self.form.addRow(self.mp)

    def build_chunks(self) -> list[CmdChunk]:
        if not self.src.text().strip():
            raise ValueError("Please fill in src_folder")
        if not self.dst.text().strip():
            raise ValueError("Please fill in dst_folder")
        ps = self.parse_triplet_int(self.patch_size.text())
        if not ps:
            raise ValueError("Please fill in patch-size, e.g. '32 32 32' or '64'")
        st = self.parse_triplet_int(self.patch_stride.text())
        if not st:
            raise ValueError("Please fill in patch-stride, e.g. '32 32 32' or '64'")
        args = [
            "itk_patch",
            self.src.text(),
            self.dst.text(),
            "--patch-size",
            *map(str, ps),
            "--patch-stride",
            *map(str, st),
        ]
        if self.min_fg.value() > 0:
            args += ["--minimum-foreground-ratio", str(self.min_fg.value())]
        if self.still_save.isChecked():
            args.append("--still-save-when-no-label")
        if self.mp.isChecked():
            args.append("--mp")
        return [CmdChunk(program=args[0], args=args[1:])]


class ItkAugTab(CommandFormBase):
    def __init__(self, parent=None):
        super().__init__("itk_aug", parent)
        imgw, self.img = self._file_picker("Browse", mode="dir")
        lblw, self.lbl = self._file_picker("Browse", mode="dir")
        oimgw, self.oimg = self._file_picker("Browse", mode="dir")
        olblw, self.olbl = self._file_picker("Browse", mode="dir")
        self.num = QtWidgets.QSpinBox()
        self.num.setRange(1, 1000)
        self.num.setValue(5)
        self.mp = self._check("Multi Processing")
        self.random_rot = self._spin_int3(-1, 360, placeholder="Z Y X")
        self.form.addRow("img_folder", imgw)
        self.form.addRow("lbl_folder", lblw)
        self.form.addRow("out-img-folder (-oimg)", oimgw)
        self.form.addRow("out-lbl-folder (-olbl)", olblw)
        self.form.addRow("num", self.num)
        self.form.addRow("random-rot (Z Y X)", self.random_rot)
        self.form.addRow(self.mp)

    def build_chunks(self) -> list[CmdChunk]:
        if not self.img.text().strip():
            raise ValueError("Please fill in img_folder")
        if not self.lbl.text().strip():
            raise ValueError("Please fill in lbl_folder")
        args = [
            "itk_aug",
            self.img.text(),
            self.lbl.text(),
        ]
        if self.oimg.text():
            args += ["-oimg", self.oimg.text()]
        if self.olbl.text():
            args += ["-olbl", self.olbl.text()]
        if self.num.value() > 0:
            args += ["-n", str(self.num.value())]
        if self.mp.isChecked():
            args.append("--mp")
        rot = self.parse_triplet_int(self.random_rot.text())
        if rot:
            args += ["--random-rot", *map(str, rot)]
        return [CmdChunk(program=args[0], args=args[1:])]


class ItkExtractTab(CommandFormBase):
    def __init__(self, parent=None):
        super().__init__("itk_extract", parent)
        srcw, self.src = self._file_picker("Browse", mode="dir")
        dstw, self.dst = self._file_picker("Browse", mode="dir")
        self.mappings = QtWidgets.QPlainTextEdit()
        self.mappings.setPlaceholderText("One mapping per line: source:target e.g.\n1:0\n5:1")
        _style_lineedit_placeholder(self.mappings)
        self.recursive = self._check("recursively process all samples")
        self.mp = self._check("Multi Processing")
        self.workers = QtWidgets.QSpinBox()
        self.workers.setRange(1, 512)
        self.workers.setValue(8)
        self.form.addRow("source_folder", srcw)
        self.form.addRow("dest_folder", dstw)
        self.form.addRow("mappings", self.mappings)
        self.form.addRow(self.recursive)
        self.form.addRow(self.mp)
        self.form.addRow("workers", self.workers)

    def build_chunks(self) -> list[CmdChunk]:
        mapping_lines = [ln.strip() for ln in self.mappings.toPlainText().splitlines() if ln.strip()]
        if not self.src.text().strip():
            raise ValueError("Please fill in source_folder")
        if not self.dst.text().strip():
            raise ValueError("Please fill in dest_folder")
        if not mapping_lines:
            raise ValueError("Please provide at least one mapping, e.g. '1:0' or '5:1'")
        args = [
            "itk_extract",
            self.src.text(),
            self.dst.text(),
            *mapping_lines,
        ]
        if self.recursive.isChecked():
            args.append("--recursive")
        if self.mp.isChecked():
            args.append("--mp")
        if self.workers.value() > 0:
            args += ["--workers", str(self.workers.value())]
        return [CmdChunk(program=args[0], args=args[1:])]
