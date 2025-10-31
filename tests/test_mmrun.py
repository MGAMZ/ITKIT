import sys
import pytest

@pytest.mark.torch
@pytest.mark.mm
def test_mmrun_smoke_empty_configdir(tmp_path, monkeypatch):
    # Skip if heavy deps are missing
    pytest.importorskip("torch", reason="PyTorch not installed")
    pytest.importorskip("mmengine", reason="MMEngine not installed")

    config_root = tmp_path / "cfg"
    work_root = tmp_path / "work"
    test_root = tmp_path / "test"
    config_root.mkdir()
    work_root.mkdir()
    test_root.mkdir()

    monkeypatch.setenv("mm_configdir", str(config_root))
    monkeypatch.setenv("mm_workdir", str(work_root))
    monkeypatch.setenv("mm_testdir", str(test_root))
    # Ensure SUPPORTED_MODELS is empty; parser default may read it
    monkeypatch.setenv("supported_models", "")

    from itkit.mm import run as mm_run

    # Provide a non-existing experiment; runner should not crash
    monkeypatch.setattr(sys, "argv", ["mmrun", "0.1.DummyExp"])
    mm_run.main()
