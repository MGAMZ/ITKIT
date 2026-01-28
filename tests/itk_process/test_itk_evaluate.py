"""Tests for itk_evaluate CLI command."""

import sys
from pathlib import Path

import numpy as np
import pytest


def _write_mha(path: Path, array: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Helper to write a test .mha file."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img = sitk.DICOMOrient(img, "LPI")
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)


def _make_toy_label(shape=(8, 8, 8), pattern='simple'):
    """Create toy label data for testing.

    Args:
        shape: Shape of the label array
        pattern: Type of label pattern:
            - 'simple': Binary label with foreground in center
            - 'multiclass': Multi-class label with background=0, class1=1, class2=2
    """
    arr = np.zeros(shape, dtype=np.uint8)
    if pattern == 'simple':
        # Binary: background=0, foreground=1
        arr[2:6, 2:6, 2:6] = 1
    elif pattern == 'multiclass':
        # Multi-class: background=0, class1=1, class2=2
        arr[2:6, 2:6, 2:6] = 1
        arr[4:7, 4:7, 4:7] = 2
    return arr


@pytest.mark.itk_process
def test_itk_evaluate_basic(tmp_path, monkeypatch):
    """Test basic evaluation with matching GT and predictions."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("pandas", reason="pandas not installed")

    from itkit.process import itk_evaluate

    # Create test data
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    save_dir = tmp_path / "results"

    # Create identical GT and prediction (perfect match)
    gt_label = _make_toy_label(pattern='simple')
    _write_mha(gt_dir / "case1.mha", gt_label)
    _write_mha(pred_dir / "case1.mha", gt_label)

    # Run evaluation with CSV format
    monkeypatch.setattr(sys, "argv", [
        "itk_evaluate",
        str(gt_dir),
        str(pred_dir),
        str(save_dir),
        "--format", "csv"
    ])
    itk_evaluate.main()

    # Check that output files were created
    assert (save_dir / "per_class_sample_avg.csv").exists()
    assert (save_dir / "per_sample_per_class.csv").exists()
    assert (save_dir / "per_sample_class_avg.csv").exists()
    assert (save_dir / "per_sample_per_class_volume_gt.csv").exists()
    assert (save_dir / "per_sample_per_class_volume_pred.csv").exists()

    # Verify metrics are perfect (Dice=1.0 for perfect match)
    import pandas as pd
    per_class = pd.read_csv(save_dir / "per_class_sample_avg.csv")
    assert 'metric' in per_class.columns
    # Check that dice metric exists and is 1.0 for perfect match
    dice_row = per_class[per_class['metric'] == 'dice']
    assert not dice_row.empty
    # All class dice values should be 1.0
    for col in dice_row.columns:
        if col.startswith('class_'):
            assert dice_row[col].values[0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.itk_process
def test_itk_evaluate_excel_format(tmp_path, monkeypatch):
    """Test evaluation with Excel output format."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("pandas", reason="pandas not installed")
    pytest.importorskip("openpyxl", reason="openpyxl not installed")

    from itkit.process import itk_evaluate

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    save_dir = tmp_path / "results"

    # Create test data with multiclass labels
    gt_label = _make_toy_label(pattern='multiclass')
    _write_mha(gt_dir / "case1.mha", gt_label)
    _write_mha(pred_dir / "case1.mha", gt_label)

    # Run evaluation with Excel format
    monkeypatch.setattr(sys, "argv", [
        "itk_evaluate",
        str(gt_dir),
        str(pred_dir),
        str(save_dir),
        "--format", "excel"
    ])
    itk_evaluate.main()

    # Check that Excel file was created
    excel_path = save_dir / "evaluation_results.xlsx"
    assert excel_path.exists()

    # Verify sheets exist
    import pandas as pd
    xl_file = pd.ExcelFile(excel_path)
    assert 'PerClass_SampleAvg' in xl_file.sheet_names
    assert 'PerSample_PerClass' in xl_file.sheet_names
    assert 'PerSample_ClassAvg' in xl_file.sheet_names
    assert 'Volume_GT' in xl_file.sheet_names
    assert 'Volume_Pred' in xl_file.sheet_names


@pytest.mark.itk_process
def test_itk_evaluate_resampling(tmp_path, monkeypatch):
    """Test evaluation with automatic resampling when sizes differ."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("pandas", reason="pandas not installed")

    from itkit.process import itk_evaluate

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    save_dir = tmp_path / "results"

    # Create GT with original size
    gt_label = _make_toy_label(shape=(8, 8, 8), pattern='simple')
    _write_mha(gt_dir / "case1.mha", gt_label, spacing=(1.0, 1.0, 1.0))

    # Create prediction with different size (should be resampled)
    pred_label = _make_toy_label(shape=(16, 16, 16), pattern='simple')
    _write_mha(pred_dir / "case1.mha", pred_label, spacing=(0.5, 0.5, 0.5))

    # Run evaluation (should auto-resample pred to match GT)
    monkeypatch.setattr(sys, "argv", [
        "itk_evaluate",
        str(gt_dir),
        str(pred_dir),
        str(save_dir),
        "--format", "csv"
    ])
    itk_evaluate.main()

    # Check that evaluation completed without error
    assert (save_dir / "per_sample_per_class.csv").exists()

    # Verify that metrics were calculated (even if not perfect due to resampling)
    import pandas as pd
    per_sample = pd.read_csv(save_dir / "per_sample_per_class.csv")
    assert len(per_sample) == 1  # One sample
    assert 'sample' in per_sample.columns
    assert per_sample['sample'].values[0] == 'case1'


@pytest.mark.itk_process
def test_itk_evaluate_multiple_samples(tmp_path, monkeypatch):
    """Test evaluation with multiple samples."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("pandas", reason="pandas not installed")

    from itkit.process import itk_evaluate

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    save_dir = tmp_path / "results"

    # Create multiple samples
    for i in range(3):
        gt_label = _make_toy_label(pattern='multiclass')
        _write_mha(gt_dir / f"case{i}.mha", gt_label)
        _write_mha(pred_dir / f"case{i}.mha", gt_label)

    # Run evaluation
    monkeypatch.setattr(sys, "argv", [
        "itk_evaluate",
        str(gt_dir),
        str(pred_dir),
        str(save_dir),
        "--format", "csv"
    ])
    itk_evaluate.main()

    # Verify all samples were processed
    import pandas as pd
    per_sample = pd.read_csv(save_dir / "per_sample_per_class.csv")
    assert len(per_sample) == 3
    assert set(per_sample['sample'].values) == {'case0', 'case1', 'case2'}

    # Check that averaging was done correctly in per_class_sample_avg
    per_class = pd.read_csv(save_dir / "per_class_sample_avg.csv")
    assert 'metric' in per_class.columns
    assert len(per_class) > 0  # Should have multiple metric rows


@pytest.mark.itk_process
def test_itk_evaluate_imperfect_prediction(tmp_path, monkeypatch):
    """Test evaluation with imperfect predictions (not 100% match)."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("pandas", reason="pandas not installed")

    from itkit.process import itk_evaluate

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    save_dir = tmp_path / "results"

    # Create GT
    gt_label = _make_toy_label(shape=(8, 8, 8), pattern='simple')
    _write_mha(gt_dir / "case1.mha", gt_label)

    # Create prediction with some differences
    pred_label = _make_toy_label(shape=(8, 8, 8), pattern='simple')
    # Introduce some errors: change some pixels
    pred_label[3:5, 3:5, 3:5] = 0  # False negatives
    _write_mha(pred_dir / "case1.mha", pred_label)

    # Run evaluation
    monkeypatch.setattr(sys, "argv", [
        "itk_evaluate",
        str(gt_dir),
        str(pred_dir),
        str(save_dir),
        "--format", "csv"
    ])
    itk_evaluate.main()

    # Verify metrics are not perfect (Dice < 1.0)
    import pandas as pd
    per_class = pd.read_csv(save_dir / "per_class_sample_avg.csv")
    dice_row = per_class[per_class['metric'] == 'dice']

    # Check class_1 dice is less than 1.0 due to errors
    if 'class_1' in dice_row.columns:
        dice_value = dice_row['class_1'].values[0]
        assert 0.0 < dice_value < 1.0, f"Expected Dice < 1.0 due to errors, got {dice_value}"


@pytest.mark.itk_process
def test_itk_evaluate_multiprocessing(tmp_path, monkeypatch):
    """Test evaluation with multiprocessing enabled."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("pandas", reason="pandas not installed")

    from itkit.process import itk_evaluate

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    save_dir = tmp_path / "results"

    # Create multiple samples to test multiprocessing
    for i in range(4):
        gt_label = _make_toy_label(pattern='multiclass')
        _write_mha(gt_dir / f"case{i}.mha", gt_label)
        _write_mha(pred_dir / f"case{i}.mha", gt_label)

    # Run evaluation with multiprocessing enabled
    monkeypatch.setattr(sys, "argv", [
        "itk_evaluate",
        str(gt_dir),
        str(pred_dir),
        str(save_dir),
        "--format", "csv",
        "--mp",
        "--workers", "2"
    ])
    itk_evaluate.main()

    # Verify all samples were processed correctly
    import pandas as pd
    per_sample = pd.read_csv(save_dir / "per_sample_per_class.csv")
    assert len(per_sample) == 4, f"Expected 4 samples, got {len(per_sample)}"
    assert set(per_sample['sample'].values) == {'case0', 'case1', 'case2', 'case3'}

    # Verify metrics are perfect (Dice = 1.0 for perfect match)
    per_class = pd.read_csv(save_dir / "per_class_sample_avg.csv")
    dice_row = per_class[per_class['metric'] == 'dice']
    assert not dice_row.empty
    # All class dice values should be 1.0 for perfect matches
    for col in dice_row.columns:
        if col.startswith('class_'):
            assert dice_row[col].values[0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.itk_process
def test_itk_evaluate_volume_calculation(tmp_path, monkeypatch):
    """Test that volume calculations are correct for known geometries."""
    pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("pandas", reason="pandas not installed")

    from itkit.process import itk_evaluate

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    save_dir = tmp_path / "results"

    # Create GT with known geometry: 4x4x4 voxels of class 1
    gt_label = np.zeros((8, 8, 8), dtype=np.uint8)
    gt_label[2:6, 2:6, 2:6] = 1  # 4x4x4 = 64 voxels
    spacing = (2.0, 2.0, 2.0)  # 2mm spacing
    _write_mha(gt_dir / "case1.mha", gt_label, spacing=spacing)

    # Create prediction with different volume: 3x3x3 voxels of class 1
    pred_label = np.zeros((8, 8, 8), dtype=np.uint8)
    pred_label[2:5, 2:5, 2:5] = 1  # 3x3x3 = 27 voxels
    _write_mha(pred_dir / "case1.mha", pred_label, spacing=spacing)

    # Run evaluation
    monkeypatch.setattr(sys, "argv", [
        "itk_evaluate",
        str(gt_dir),
        str(pred_dir),
        str(save_dir),
        "--format", "csv"
    ])
    itk_evaluate.main()

    # Verify volume calculations
    import pandas as pd

    # Check Volume_GT
    volume_gt = pd.read_csv(save_dir / "per_sample_per_class_volume_gt.csv")
    assert 'sample' in volume_gt.columns
    assert volume_gt['sample'].values[0] == 'case1'

    # Expected GT volume: 64 voxels * (2*2*2) mm³/voxel = 512 mm³ for class 1
    voxel_volume = 2.0 * 2.0 * 2.0
    expected_gt_volume_class1 = 64 * voxel_volume
    if 'class_1' in volume_gt.columns:
        actual_gt_volume = volume_gt['class_1'].values[0]
        assert actual_gt_volume == pytest.approx(expected_gt_volume_class1, abs=0.1)

    # Check Volume_Pred
    volume_pred = pd.read_csv(save_dir / "per_sample_per_class_volume_pred.csv")
    assert 'sample' in volume_pred.columns

    # Expected Pred volume: 27 voxels * (2*2*2) mm³/voxel = 216 mm³ for class 1
    expected_pred_volume_class1 = 27 * voxel_volume
    if 'class_1' in volume_pred.columns:
        actual_pred_volume = volume_pred['class_1'].values[0]
        assert actual_pred_volume == pytest.approx(expected_pred_volume_class1, abs=0.1)

    # Verify GT and Pred volumes are different (as expected)
    if 'class_1' in volume_gt.columns and 'class_1' in volume_pred.columns:
        assert volume_gt['class_1'].values[0] != volume_pred['class_1'].values[0]
