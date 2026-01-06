from mmengine.config import read_base
with read_base():
    from .mgam import *

from itkit.criterions.segment import DiceCELoss_3D
from itkit.mm.task_models import SemSeg3D
from itkit.models.MedNeXt import MedNeXt

model = dict(
    type=SemSeg3D,
    binary_segment_threshold=None,
    num_classes=num_classes,
    backbone=dict(
        type=MedNeXt,
        in_channels=in_channels, # pyright: ignore
        n_channels=16,
        n_classes=num_classes,
        exp_r=2,
    ),
    criterion=dict(
        type=DiceCELoss_3D,
        split_Z=False,
        to_onehot_y=True,
        sigmoid=False,
        softmax=True,
        squared_pred=False,
        batch=True,
    ),
    inference_config=dict(
        patch_size=size,
        patch_stride=[s//4*3 for s in size],
        accumulate_device='cpu',
        forward_device='cuda',
        forward_batch_windows=8,
        argmax_batchsize=8,
    ),
)
