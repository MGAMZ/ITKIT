import os
import shutil
import tempfile
import numpy as np
import pytest
import SimpleITK as sitk
import torchio as tio
from itkit.dataset.base import mgam_TorchIO_Patched_Structure

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
    assert dataset.tio_queue is not None, "TorchIO Queue should be initialized"
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
