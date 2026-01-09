import pytest
from unittest.mock import patch
from mmengine.logging import MMLogger

# Import dataset classes
from itkit.dataset.AbdomenCT_1K.mm_dataset import AbdomenCT_1K_Mha
from itkit.dataset.CT_ORG.mm_dataset import CT_ORG_Mha
from itkit.dataset.KiTS23.mm_dataset import KiTS23_Mha
from itkit.dataset.FLARE_2022.mm_dataset import FLARE_2022_Mha
from itkit.dataset.FLARE_2023.mm_dataset import FLARE_2023_Mha
from itkit.dataset.CTSpine1K.mm_dataset import CTSpine1K_Mha
from itkit.dataset.ImageTBAD.mm_dataset import TBAD_Mha
from itkit.dataset.LUNA16.mm_dataset import LUNA16_Mha
from itkit.dataset.LiTS.mm_dataset import LiTS_Mha

@pytest.mark.parametrize("dataset_class, extra_kwargs", [
    (AbdomenCT_1K_Mha, {}),
    (CT_ORG_Mha, {}),
    (KiTS23_Mha, {}),
    (FLARE_2022_Mha, {}),
    (FLARE_2023_Mha, {}),
    (CTSpine1K_Mha, {}),
    (TBAD_Mha, {}),
    (LUNA16_Mha, {}),
    (LiTS_Mha, {}),
])
def test_dataset_common_metainfo(dataset_class, extra_kwargs):
    """验证不同数据集类的 METAINFO 和 Palette 初始化一致性"""
    MMLogger.get_instance('mmengine')

    with patch('os.path.exists', return_value=True), \
         patch('os.listdir', return_value=[]), \
         patch('itkit.dataset.base.SeriesVolumeDataset._load_series_meta', return_value={}):

        # 对于 BraTs2024_Dataset，它直接继承 ITKITBaseSegDataset，逻辑略有不同但 _update_palette 是一致的
        ds = dataset_class(
            data_root='/tmp/fake',
            split='train',
            pipeline=[],
            lazy_init=True,
            **extra_kwargs
        )

        # 1. 验证 classes 存在且包含 background
        assert 'classes' in ds.METAINFO
        assert ds.METAINFO['classes'][0] == 'background'

        # 2. 验证 palette 修正逻辑 (背景归零)
        palette = ds._update_palette()
        assert palette[0] == [0, 0, 0], f"{dataset_class.__name__} background palette should be black"
        assert len(palette) == len(ds.METAINFO['classes'])

def test_dataset_split_compatibility(itkit_dummy_dataset):
    """验证各数据集类在真实（模拟）目录下的初始化"""
    MMLogger.get_instance('mmengine')

    # 测试几个典型的数据集
    for ds_cls in [AbdomenCT_1K_Mha, CT_ORG_Mha]:
        ds = ds_cls(
            data_root=itkit_dummy_dataset,
            split='train',
            pipeline=[],
            lazy_init=True
        )
        # 触发 split
        series = ds._split()
        assert len(series) > 0
        assert isinstance(series[0], str)
