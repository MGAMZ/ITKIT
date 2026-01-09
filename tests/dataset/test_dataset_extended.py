from unittest.mock import MagicMock, patch

from itkit.dataset.base import SeriesVolumeDataset
from itkit.dataset.monai import MONAI_PatchedDataset
from itkit.dataset.torchio import TorchIO_PatchedDataset
from mmengine.logging import MMLogger

class SimpleSeriesDataset(SeriesVolumeDataset):
    """A minimal concrete implementation for testing split logic"""
    def sample_iterator(self):
        for series in self._split():
            yield (series, series)

def test_dataset_split_ratios(itkit_dummy_dataset):
    """测试数据集划分比例 [0.8, 0.05, 0.15]"""
    MMLogger.get_instance('mmengine')

    # 20 samples total in itkit_dummy_dataset
    # Train: 20 * 0.8 = 16
    # Val: 20 * 0.05 = 1 + 1 (hack in code) = 2
    # Test: 20 - 16 - 2 = 2

    train_ds = SimpleSeriesDataset(data_root=itkit_dummy_dataset, split='train', pipeline=[], lazy_init=True)
    val_ds = SimpleSeriesDataset(data_root=itkit_dummy_dataset, split='val', pipeline=[], lazy_init=True)
    test_ds = SimpleSeriesDataset(data_root=itkit_dummy_dataset, split='test', pipeline=[], lazy_init=True)
    all_ds = SimpleSeriesDataset(data_root=itkit_dummy_dataset, split='all', pipeline=[], lazy_init=True)

    train_series = train_ds._split()
    val_series = val_ds._split()
    test_series = test_ds._split()
    all_series = all_ds._split()

    assert len(all_series) == 20
    assert len(train_series) == 16
    assert len(val_series) == 2
    assert len(test_series) == 2

    # Check intersection
    assert set(train_series).isdisjoint(set(val_series))
    assert set(val_series).isdisjoint(set(test_series))
    assert set(train_series).union(set(val_series)).union(set(test_series)) == set(all_series)

def test_monai_worker_partitioning(itkit_dummy_dataset):
    """测试 MONAI 数据集在多进程下的数据分片逻辑"""
    MMLogger.get_instance('mmengine')

    ds = MONAI_PatchedDataset(
        data_root=itkit_dummy_dataset,
        split='all',
        samples_per_volume=2,
        patch_size=(32, 32, 32),
        pipeline=[],
        lazy_init=True
    )
    # Initialize it manually
    ds.load_data_list()

    # Mock worker info for worker 0 of 2
    mock_worker_info = MagicMock()
    mock_worker_info.id = 0
    mock_worker_info.num_workers = 2

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        iterator = ds._get_subset_iterator()
        # 20 个 volume 分给 2 个 worker，每个 worker 10 个 volume
        # 每个 volume 采 2 个 patch，总共应该是 20 个 patch
        samples = []
        for _ in range(20):
            samples.append(next(iterator))
        assert len(samples) == 20
        # 验证再取一个会触发循环或报错（取决于实现，MONAI 这里底层是 PatchDataset 的 iterator）
        # 实际实现中 next(iterator) 在 StopIteration 后会被 MONAI_PatchedDataset.prepare_data 重新获取

def test_torchio_worker_partitioning(itkit_dummy_dataset):
    """测试 TorchIO 数据集在多进程下的数据分片逻辑"""
    MMLogger.get_instance('mmengine')

    ds = TorchIO_PatchedDataset(
        data_root=itkit_dummy_dataset,
        split='all',
        samples_per_volume=3,
        patch_size=(32, 32, 32),
        pipeline=[],
        lazy_init=True
    )
    ds.load_data_list()

    # Mock worker info for worker 1 of 2
    mock_worker_info = MagicMock()
    mock_worker_info.id = 1
    mock_worker_info.num_workers = 2

    with patch('torch.utils.data.get_worker_info', return_value=mock_worker_info):
        ds._initialize_queue()
        # 20 / 2 = 10 subjects
        assert len(ds.tio_queue.subjects_dataset) == 10


def test_palette_consistency():
    """测试 Palette 的 background 归零逻辑"""
    from itkit.dataset.AbdomenCT_1K.mm_dataset import AbdomenCT_1K_Mha

    # Mock data_root to avoid file errors
    with patch('os.path.exists', return_value=True), \
         patch('os.listdir', return_value=[]), \
         patch('itkit.dataset.base.SeriesVolumeDataset._load_series_meta', return_value={}):

        ds = AbdomenCT_1K_Mha(
            data_root='/tmp/fake',
            split='train',
            pipeline=[],
            lazy_init=True
        )

        palette = ds._update_palette()
        assert palette[0] == [0, 0, 0], "Background palette must be [0, 0, 0]"
