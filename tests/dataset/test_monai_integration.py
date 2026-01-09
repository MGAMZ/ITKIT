import os
import shutil
import tempfile
import numpy as np
import pytest
import SimpleITK as sitk
from itkit.dataset.monai import MONAI_PatchedDataset


@pytest.fixture
def mock_dataset_dir():
    # Setup: Create temporary directory structure
    test_dir = tempfile.mkdtemp()
    image_dir = os.path.join(test_dir, 'image')
    label_dir = os.path.join(test_dir, 'label')
    os.makedirs(image_dir)
    os.makedirs(label_dir)

    # Create dummy data
    # 5 subjects: 5*0.8 = 4 train
    for i in range(1, 6):
        filename = f'subject{i}.mha'
        size = (64, 64, 64)

        # Image
        arr = np.random.rand(*size).astype(np.float32)
        image = sitk.GetImageFromArray(arr)
        sitk.WriteImage(image, os.path.join(image_dir, filename))

        # Label
        label_arr = np.random.randint(0, 2, size=size).astype(np.uint8)
        label = sitk.GetImageFromArray(label_arr)
        sitk.WriteImage(label, os.path.join(label_dir, filename))

    yield test_dir

    # Teardown: Clean up temporary directory
    shutil.rmtree(test_dir)


def test_dataset_initialization_and_fetching(mock_dataset_dir):
    # Define parameters
    patch_size = (32, 32, 32)
    samples_per_volume = 2

    # Initialize dataset with empty pipeline to inspect raw output
    dataset = MONAI_PatchedDataset(
        data_root=mock_dataset_dir,
        pipeline=[],
        split='train',
        samples_per_volume=samples_per_volume,
        patch_size=patch_size,
        mode='sup'
    )

    # 1. Verify PatchDataset Initialization
    assert dataset.monai_patch_ds is not None, "MONAI PatchDataset should be initialized"
    assert dataset.samples_per_volume == samples_per_volume
    assert dataset.patch_size == patch_size

    # 2. Verify Dataset Length
    # 5 subjects total. 0.8 split -> 4 subjects in train.
    # 4 subjects * 2 samples/vol = 8 samples total.
    expected_len = 4 * samples_per_volume
    assert len(dataset) == expected_len, f"Expected length {expected_len}, got {len(dataset)}"

    # 3. Fetch a sample (this triggers prepare_data -> patch iterator)
    data = dataset[0]

    # 4. Verify Data Content
    assert 'img' in data
    assert 'gt_seg_map' in data
    assert 'img_shape' in data
    assert 'ori_shape' in data

    # 5. Verify Shapes
    # img should be (D, H, W) because of squeeze(0) in prepare_data for single channel
    assert data['img'].shape == patch_size, f"Expected shape {patch_size}, got {data['img'].shape}"
    assert data['gt_seg_map'].shape == patch_size

    # 6. Verify Data Types
    assert isinstance(data['img'], np.ndarray)
    assert isinstance(data['gt_seg_map'], np.ndarray)

    # 7. Verify Metadata Preservation
    # mm_meta fields like 'img_path' should be preserved through the pipeline
    assert 'img_path' in data, "Metadata 'img_path' lost during processing"
    assert 'seg_map_path' in data, "Metadata 'seg_map_path' lost during processing"


def test_dataloader_multiprocessing(mock_dataset_dir):
    """Test that the dataset works with multiple workers (sharding)"""
    import torch

    patch_size = (32, 32, 32)
    dataset = MONAI_PatchedDataset(
        data_root=mock_dataset_dir,
        pipeline=[],
        split='train',
        samples_per_volume=2,
        patch_size=patch_size,
        mode='sup'
    )

    # Use a DataLoader with multiprocessing
    # num_workers=2 requires at least 2 volumes in the split to ensure each worker gets data
    # The mock dataset has 4 train subjects, so 2 workers is safe.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        shuffle=True
    )

    # Iterate through one batch to trigger worker initialization and data fetching
    for batch in dataloader:
        assert 'img' in batch
        # batch['img'] should be (B, D, H, W) -> (2, 32, 32, 32)
        # Note: DataLoader collates numpy arrays into torch tensors
        assert batch['img'].shape == (2, *patch_size)
        assert 'img_path' in batch # Check metadata collation
        break


def test_infinite_iterator_behavior(mock_dataset_dir):
    """Test that the iterator restarts automatically when exhausted"""
    patch_size = (32, 32, 32)
    samples_per_volume = 2

    dataset = MONAI_PatchedDataset(
        data_root=mock_dataset_dir,
        pipeline=[],
        split='train',
        samples_per_volume=samples_per_volume,
        patch_size=patch_size,
        mode='sup'
    )

    # Fetch more samples than available (should restart iterator automatically)
    expected_len = len(dataset)
    samples_to_fetch = expected_len + 5  # Exceed dataset length

    for i in range(samples_to_fetch):
        data = dataset[i % expected_len]  # Use modulo to stay within bounds
        assert 'img' in data
        assert data['img'].shape == patch_size


def test_volume_padding(mock_dataset_dir):
    """Test that volumes smaller than patch_size are padded correctly"""
    # Create a small volume
    test_dir = tempfile.mkdtemp()
    image_dir = os.path.join(test_dir, 'image')
    label_dir = os.path.join(test_dir, 'label')
    os.makedirs(image_dir)
    os.makedirs(label_dir)

    # Create 2 small volumes to ensure train split has at least 1 sample (2 * 0.8 = 1)
    small_size = (20, 20, 20)
    for i in range(1, 3):
        arr = np.random.rand(*small_size).astype(np.float32)
        image = sitk.GetImageFromArray(arr)
        sitk.WriteImage(image, os.path.join(image_dir, f'subject{i}.mha'))

        label_arr = np.random.randint(0, 2, size=small_size).astype(np.uint8)
        label = sitk.GetImageFromArray(label_arr)
        sitk.WriteImage(label, os.path.join(label_dir, f'subject{i}.mha'))

    try:
        patch_size = (32, 32, 32)
        dataset = MONAI_PatchedDataset(
            data_root=test_dir,
            pipeline=[],
            split='train',
            samples_per_volume=1,
            patch_size=patch_size,
            mode='sup'
        )

        # Should not raise error due to padding
        data = dataset[0]
        assert data['img'].shape == patch_size

    finally:
        shutil.rmtree(test_dir)


def test_data_list_conversion(mock_dataset_dir):
    """Test that mmseg format is correctly converted to MONAI format"""
    dataset = MONAI_PatchedDataset(
        data_root=mock_dataset_dir,
        pipeline=[],
        split='train',
        samples_per_volume=2,
        patch_size=(32, 32, 32),
        mode='sup'
    )

    # Verify MONAI data list format
    assert len(dataset.monai_data_list) == 4  # 4 train subjects

    for item in dataset.monai_data_list:
        assert 'image' in item
        assert 'mm_meta' in item
        assert 'label' in item
        assert os.path.exists(item['image'])
        assert os.path.exists(item['label'])
