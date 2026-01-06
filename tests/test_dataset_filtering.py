import os
import json
import pytest
import tempfile
import SimpleITK as sitk
from itkit.dataset.base import SeriesVolumeDataset
from mmengine.logging import MMLogger

def create_test_image(path: str, size: tuple, spacing: tuple):
    """Helper to create test MHA images (Size and Spacing in Z, Y, X)"""
    # SimpleITK uses XYZ ordering, so we reverse ZYX to XYZ
    img = sitk.Image(size[2], size[1], size[0], sitk.sitkUInt8)
    img.SetSpacing((spacing[2], spacing[1], spacing[0]))
    sitk.WriteImage(img, path)

class SimpleSeriesDataset(SeriesVolumeDataset):
    """A minimal concrete implementation of SeriesVolumeDataset for testing"""
    def sample_iterator(self):
        # This is not used for _split but required by ITKITBaseSegDataset
        yield from []

@pytest.fixture
def mock_dataset_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = os.path.join(tmp_dir, "image")
        lbl_dir = os.path.join(tmp_dir, "label")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        # S0: Small size, good spacing
        # Size: (10, 10, 10), Spacing: (1.0, 1.0, 1.0)
        create_test_image(os.path.join(img_dir, "S0.mha"), (10, 10, 10), (1.0, 1.0, 1.0))
        create_test_image(os.path.join(lbl_dir, "S0.mha"), (10, 10, 10), (1.0, 1.0, 1.0))

        # S1: Good size, small spacing
        # Size: (100, 100, 100), Spacing: (0.1, 0.1, 0.1)
        create_test_image(os.path.join(img_dir, "S1.mha"), (100, 100, 100), (0.1, 0.1, 0.1))
        create_test_image(os.path.join(lbl_dir, "S1.mha"), (100, 100, 100), (0.1, 0.1, 0.1))

        # S2: Good size, good spacing
        # Size: (100, 100, 100), Spacing: (1.0, 1.0, 1.0)
        create_test_image(os.path.join(img_dir, "S2.mha"), (100, 100, 100), (1.0, 1.0, 1.0))
        create_test_image(os.path.join(lbl_dir, "S2.mha"), (100, 100, 100), (1.0, 1.0, 1.0))

        # Ensure no accidental meta.json
        meta_json = os.path.join(tmp_dir, "meta.json")
        if os.path.exists(meta_json):
            os.remove(meta_json)

        yield tmp_dir

def test_dataset_filtering_and_auto_indexing(mock_dataset_dir):
    # Initialize MMLogger for tests if not exists
    MMLogger.get_instance('mmengine')

    # Test Case 1: Filter by size and spacing
    # min_size=(20, 20, 20) -> should filter S0 (10 < 20)
    # min_spacing=(0.5, 0.5, 0.5) -> should filter S1 (0.1 < 0.5)
    # S2 should be kept

    ds = SimpleSeriesDataset(
        data_root=mock_dataset_dir,
        split='all',
        min_size=(20, 20, 20),
        min_spacing=(0.5, 0.5, 0.5),
        pipeline=[],
        lazy_init=True  # Avoid calling full_init which might require more setup
    )

    # Trigger _split() which calls _filter_by_meta()
    series = ds._split()

    assert "S2" in series
    assert "S0" not in series
    assert "S1" not in series
    assert len(series) == 1, f"Expected 1 series, but got {len(series)}: {series}"

    # Verify meta.json was auto-generated
    meta_path = os.path.join(mock_dataset_dir, "meta.json")
    assert os.path.exists(meta_path), "meta.json was not auto-generated"

    with open(meta_path, 'r') as f:
        meta_content = json.load(f)
        assert "S0.mha" in meta_content
        assert "S1.mha" in meta_content
        assert "S2.mha" in meta_content
        assert meta_content["S0.mha"]["size"] == [10, 10, 10]
        assert meta_content["S1.mha"]["spacing"] == [0.1, 0.1, 0.1]

def test_no_filtering(mock_dataset_dir):
    MMLogger.get_instance('mmengine')

    # Test Case 2: No constraints
    ds = SimpleSeriesDataset(
        data_root=mock_dataset_dir,
        split='all',
        min_size=(-1, -1, -1),
        min_spacing=(-1, -1, -1),
        pipeline=[],
        lazy_init=True
    )

    series = ds._split()
    assert len(series) == 3
    assert set(series) == {"S0", "S1", "S2"}

if __name__ == "__main__":
    # If run directly, used for manual verification
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # We need a proper dir structure for the fixture-like setup if run manually
        pass
