import sys
from pathlib import Path

import numpy as np
import pytest


def _write_mha(path: Path, array: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img = sitk.DICOMOrient(img, "LPI")
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)


def _make_toy_image(shape=(8, 8, 8), value=10):
    return np.full(shape, value, dtype=np.int16)


def _make_toy_label(shape=(8, 8, 8), fg=False):
    arr = np.zeros(shape, dtype=np.uint8)
    if fg:
        arr[1:3, 1:3, 1:3] = 1
    return arr


@pytest.mark.itk_process
def test_itk_check_main(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    from itkit.process import itk_check

    sample = tmp_path / "sample"
    img_dir = sample / "image"
    lbl_dir = sample / "label"
    _write_mha(img_dir / "case.mha", _make_toy_image())
    _write_mha(lbl_dir / "case.mha", _make_toy_label())

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    monkeypatch.setattr(sys, "argv", ["itk_check", "check", str(sample)])
    itk_check.main()


@pytest.mark.itk_process
def test_itk_resample_main(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    from itkit.process import itk_resample

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_mha(src / "img.mha", _make_toy_image())

    monkeypatch.setattr(sys, "argv", [
        "itk_resample", "image", str(src), str(dst),
        "--spacing", "-1", "-1", "-1",
        "--size", "4", "4", "4",
    ])
    itk_resample.main()
    assert (dst / "img.mha").exists()


@pytest.mark.itk_process
def test_itk_orient_main(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    from itkit.process import itk_orient

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_mha(src / "a.mha", _make_toy_image())

    monkeypatch.setattr(sys, "argv", ["itk_orient", str(src), str(dst), "LPI"])
    itk_orient.main()
    assert (dst / "a.mha").exists()


@pytest.mark.itk_process
def test_itk_patch_main(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    from itkit.process import itk_patch

    src = tmp_path / "pairs"
    dst = tmp_path / "patches"
    _write_mha(src / "image" / "k.mha", _make_toy_image())
    _write_mha(src / "label" / "k.mha", _make_toy_label(fg=True))

    monkeypatch.setattr(sys, "argv", [
        "itk_patch", str(src), str(dst),
        "--patch-size", "4", "4", "4",
        "--patch-stride", "4", "4", "4",
        "--minimum-foreground-ratio", "0.0",
        "--still-save-when-no-label",
    ])
    itk_patch.main()
    # Should produce global image/ and label/ folders and crop_meta.json
    assert (dst / "image").exists()
    assert (dst / "label").exists()
    assert (dst / "crop_meta.json").exists()
    # At least one patch for the series 'k' should be generated in both dirs
    imgs = list((dst / "image").glob("k_p*.mha"))
    lbls = list((dst / "label").glob("k_p*.mha"))
    assert len(imgs) >= 1
    assert len(lbls) >= 1


@pytest.mark.itk_process
def test_itk_aug_main(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    from itkit.process import itk_aug

    img = tmp_path / "img"
    lbl = tmp_path / "lbl"
    oimg = tmp_path / "oimg"
    olbl = tmp_path / "olbl"
    _write_mha(img / "p.mha", _make_toy_image())
    _write_mha(lbl / "p.mha", _make_toy_label(fg=True))

    monkeypatch.setattr(sys, "argv", [
        "itk_aug", str(img), str(lbl),
        "-oimg", str(oimg), "-olbl", str(olbl),
        "-n", "1", "--random-rot", "0", "0", "0"
    ])
    itk_aug.main()
    # Augmented outputs should exist when out folders are provided
    outs = list(oimg.glob("*.mha"))
    assert len(outs) >= 1


@pytest.mark.itk_process
def test_itk_extract_main(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    from itkit.process import itk_extract

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    # Build a label map with values 0 and 2
    arr = _make_toy_label()
    arr[2:4, 2:4, 2:4] = 2
    _write_mha(src / "lab.mha", arr)

    monkeypatch.setattr(sys, "argv", [
        "itk_extract", str(src), str(dst), "2:1"
    ])
    itk_extract.main()
    assert (dst / "lab.mha").exists()
    assert (dst / "meta.json").exists()
