from mmengine.config import read_base
with read_base():
    from .mgam import *

from itkit.criterions.segment import DiceCELoss_3D
from itkit.mm.task_models import SemSeg3D
from itkit.models.SegFormer3D import SegFormer3D

model = dict(
    type=SemSeg3D,
    binary_segment_threshold=None,
    num_classes=num_classes,
    backbone=dict(
        type=SegFormer3D,
        in_channels=in_channels, # pyright: ignore
        num_classes=num_classes,
        sr_ratios=         [(4,8,8), (2,4,4), (2,2,2), (1,1,1)],
        patch_kernel_size= [7,       3,       3,       3      ],
        patch_stride=      [4,       2,       2,       2      ],
        patch_padding=     [3,       1,       1,       1      ],
        embed_dims=        [128,     256,     512,     1024   ],
        num_heads=         [4,       8,       16,      32     ],
        depths=            [2,       2,       2,       2      ],
        mlp_ratios=        [2,       2,       2,       2      ],
        decoder_head_embedding_dim=384,
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
