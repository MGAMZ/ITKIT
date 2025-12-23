import os
import shutil
import tempfile
import numpy as np
import pytest
import SimpleITK as sitk
import torchio as tio
from itkit.dataset.torchio import mgam_TorchIO_Patched_Structure


@pytest.fixture
def mock_dataset_dir():
    # Setup: 创建临时目录结构
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

    # Teardown: 清理临时目录
    shutil.rmtree(test_dir)

def test_dataset_initialization_and_fetching(mock_dataset_dir):
    # Define parameters
    patch_size = (32, 32, 32)
    queue_length = 20
    samples_per_volume = 2

    # Initialize dataset
    # We use an empty pipeline so we can inspect the raw output of prepare_data
    dataset = mgam_TorchIO_Patched_Structure(
        data_root=mock_dataset_dir,
        pipeline=[],
        split='train',
        queue_max_length=queue_length,
        samples_per_volume=samples_per_volume,
        patch_size=patch_size,
        queue_num_workers=0, # Use 0 to avoid multiprocessing in tests
        mode='sup'
    )

    # 1. Verify Queue Initialization
    # assert dataset.tio_queue is not None, "TorchIO Queue should be initialized"
    assert dataset.queue_max_length == queue_length
    assert isinstance(dataset.subjects_dataset, tio.SubjectsDataset)

    # 2. Verify Dataset Length
    # 5 subjects total. 0.8 split -> 4 subjects in train.
    # 4 subjects * 2 samples/vol = 8 samples total.
    expected_len = 4 * samples_per_volume
    assert len(dataset) == expected_len, f"Expected length {expected_len}, got {len(dataset)}"

    # 3. Fetch a sample (this triggers prepare_data -> queue pop)
    data = dataset[0]

    # 4. Verify Data Content
    assert 'img' in data
    assert 'gt_seg_map' in data
    assert 'img_shape' in data
    assert 'patch_location' in data

    # 5. Verify Shapes
    # img should be (D, H, W) because of squeeze(0) in prepare_data for single channel
    assert data['img'].shape == patch_size
    assert data['gt_seg_map'].shape == patch_size

    # 6. Verify Data Types
    assert isinstance(data['img'], np.ndarray)
    assert isinstance(data['gt_seg_map'], np.ndarray)


def test_dataloader_multiprocessing(mock_dataset_dir):
    """Test that the dataset works with multiple workers (sharding)"""
    import torch

    patch_size = (32, 32, 32)
    # Use 0 queue workers to avoid nested multiprocessing issues in test environment
    dataset = mgam_TorchIO_Patched_Structure(
        data_root=mock_dataset_dir,
        pipeline=[],
        split='train',
        queue_max_length=20,
        samples_per_volume=2,
        patch_size=patch_size,
        queue_num_workers=0,
        mode='sup'
    )

    # Use a DataLoader with multiprocessing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        shuffle=True
    )

    # Iterate through one batch
    for batch in dataloader:
        assert 'img' in batch
        assert batch['img'].shape == (2, *patch_size)
        break


def test_volume_padding(mock_dataset_dir):
    """Test that volumes smaller than patch_size are handled (expecting error currently)"""
    test_dir = tempfile.mkdtemp()
    image_dir = os.path.join(test_dir, 'image')
    label_dir = os.path.join(test_dir, 'label')
    os.makedirs(image_dir)
    os.makedirs(label_dir)

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
        dataset = mgam_TorchIO_Patched_Structure(
            data_root=test_dir,
            pipeline=[],
            split='train',
            samples_per_volume=1,
            patch_size=patch_size,
            queue_num_workers=0,
            mode='sup'
        )

        # TorchIO UniformSampler requires image >= patch_size.
        # Without explicit padding, this should raise RuntimeError.
        with pytest.raises(RuntimeError):
             _ = dataset[0]

    finally:
        shutil.rmtree(test_dir)


def test_data_list_conversion(mock_dataset_dir):
    """Test that mmseg format is correctly converted to TorchIO format"""
    dataset = mgam_TorchIO_Patched_Structure(
        data_root=mock_dataset_dir,
        pipeline=[],
        split='train',
        samples_per_volume=2,
        patch_size=(32, 32, 32),
        queue_num_workers=0,
        mode='sup'
    )

    # Verify TorchIO subjects
    assert dataset.subjects_dataset is not None
    assert len(dataset.subjects_dataset) == 4

    for subject in dataset.subjects_dataset:
        assert 'image' in subject
        assert 'label' in subject
        assert 'mm_meta' in subject
        assert isinstance(subject['image'], tio.ScalarImage)
        assert isinstance(subject['label'], tio.LabelMap)
