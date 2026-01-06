from torch.optim.adamw import AdamW
from torch.distributed.fsdp.api import ShardingStrategy

from mmengine.runner import ValLoop
from mmengine.runner import TestLoop
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks import DistSamplerSeedHook
from mmengine.runner import IterBasedTrainLoop
from mmengine.optim.scheduler import LinearLR, PolyLR
from mmengine.optim import OptimWrapper, AmpOptimWrapper
from mmengine.model.wrappers import MMFullyShardedDataParallel
from mmengine._strategy.deepspeed import DeepSpeedOptimWrapper, DeepSpeedStrategy
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler
from mmengine.dataset.utils import default_collate
from mmengine.visualization import TensorboardVisBackend

# customize
from itkit.mm.mmeng_PlugIn import (
    RemasteredDDP, LoggerJSON, RuntimeInfoHook, multi_sample_collate,
    RatioSampler, RemasteredFSDP_Strategy)
from itkit.process.GeneralPreProcess import WindowSet, TypeConvert
from itkit.process.LoadBiomedicalData import LoadImageFromMHA, LoadMaskFromMHA
from itkit.mm.mmseg_Dev3D import PackSeg3DInputs, Seg3DDataPreProcessor
from itkit.mm.mmseg_PlugIn import IoUMetric_PerClass
from itkit.dataset import ITKITConcatDataset, MONAI_PatchedDataset
from itkit.dataset.Totalsegmentator.mm_dataset import TsdMRI_Mha
from itkit.mm.visualization import SegViser, BaseVisHook, LocalVisBackend



# --------------------PARAMETERS-------------------- #

# PyTorch
debug = False
use_AMP = True
dist = False if not debug else False  # distribution
MP_mode = "ddp"  # Literal[`"ddp", "fsdp", "deepspeed"]
Compile = False if not debug else False  # torch.dynamo
workers = 4 if not debug else 0  # DataLoader Worker

# Starting
resume = True
load_from = None
resume_optimizer = True
resume_param_scheduler = True

# NN Hyper Params
lr = 1e-4
batch_size_loader = 4
grad_accumulation = 1
weight_decay = 1e-2
in_channels = 1

# Train Process
iters = 100000 if not debug else 3
logger_interval = 200 if not debug else 1
val_on_train = True
val_sample_ratio = 0.1 if not debug else 0.01
val_interval = 10000 if not debug else 2
vis_interval = 1 if not debug else 1
save_interval = val_interval
dynamic_intervals = None

# Dataset
data_root = "/mnt/wsl/Fwsldatavhdx/mgam_datasets/TotalSegmentatorMRI/spacing2"
num_classes = 57
wl = 100     # window level
ww = 300    # window width
size = (32,32,32) # [Z, Y, X]
pad_val = 0
seg_pad_val = 0

# --------------------PARAMETERS-------------------- #
# ////////////////////////////////////////////////// #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# --------------------COMPONENTS-------------------- #

# Data Load and Preprocess
meta_keys = ('img_path', 'seg_map_path', 'ori_shape',
             'img_shape', 'reduce_zero_label', 'seg_fields')
train_pipeline = [
    dict(type=WindowSet, level=wl, width=ww),
    dict(type=TypeConvert, key=['img'], dtype='float16'),
    dict(type=PackSeg3DInputs, meta_keys=meta_keys),
]
val_pipeline = test_pipeline = [
    dict(type=LoadImageFromMHA),
    dict(type=LoadMaskFromMHA),
    dict(type=WindowSet, level=wl, width=ww),
    dict(type=TypeConvert, key=['img'], dtype='float16'),
    dict(type=PackSeg3DInputs, meta_keys=meta_keys),
]

train_dataloader = dict(
    batch_size=batch_size_loader,
    num_workers=workers,
    drop_last=False if debug else True,
    pin_memory=True,
    persistent_workers=True if workers > 0 else False,
    collate_fn=dict(type=multi_sample_collate),
    sampler=dict(
        type=InfiniteSampler,
        shuffle=False),
    dataset=dict(
        type=ITKITConcatDataset,
        datasets=[
            dict(type=MONAI_PatchedDataset,
                 data_root=data_root,
                 pipeline=train_pipeline,
                 split='train',
                 debug=debug,
                 patch_size=size,
                 samples_per_volume=100,
                 min_size=size,),
        ]
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=workers,
    drop_last=False,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type=RatioSampler,
                 use_sample_ratio=val_sample_ratio,
                 shuffle=False if debug else True),
    collate_fn=dict(type=default_collate),
    dataset=dict(
        type=ITKITConcatDataset,
        datasets=[
            dict(type=TsdMRI_Mha,
                 data_root=data_root,
                 min_size=size,
                 pipeline=val_pipeline,
                 split='val',
                 debug=debug),
        ]
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=workers,
    drop_last=False,
    pin_memory=False,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type=DefaultSampler),
    collate_fn=dict(type=default_collate),
    dataset=dict(
        type=ITKITConcatDataset,
        datasets=[
            dict(type=TsdMRI_Mha,
                 data_root=data_root,
                 min_size=size,
                 pipeline=test_pipeline,
                 split='test',
                 debug=debug),
        ]
    )
)

val_evaluator = test_evaluator = dict(
    type=IoUMetric_PerClass,
    prefix='Perf',
    num_classes=num_classes,
)

data_preprocessor = dict(
    type=Seg3DDataPreProcessor,
    size=size,
    pad_val=pad_val,
    seg_pad_val=seg_pad_val,
    test_cfg=dict(size=size),
    non_blocking=True,
)

train_cfg = dict(
    type=IterBasedTrainLoop,
    max_iters=iters,
    val_interval=val_interval,
    dynamic_intervals=dynamic_intervals
)
val_cfg  = dict(type=ValLoop, fp16=True)
test_cfg = dict(type=TestLoop)

if not val_on_train:
    val_dataloader = None
    val_evaluator = None
    val_cfg = None

# Optimizer
if MP_mode == "deepspeed" and dist:
    optim_wrapper = dict(
        type=DeepSpeedOptimWrapper,
        optimizer=dict(type=AdamW, lr=lr, weight_decay=weight_decay),
        accumulative_counts=grad_accumulation,
    )
else:
    optim_wrapper = dict(
        type=AmpOptimWrapper if use_AMP else OptimWrapper,
        accumulative_counts=grad_accumulation,
        optimizer=dict(type=AdamW, lr=lr, weight_decay=weight_decay),
        clip_grad=dict(max_norm=5, norm_type=2, error_if_nonfinite=False),
    )
if use_AMP and dist and MP_mode=='fsdp':
    optim_wrapper["use_fsdp"] = True

# LR Strategy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-2,
        end=iters * 0.1,
        by_epoch=False,
    ),
    dict(
        type=PolyLR,
        eta_min=lr * 1e-2,
        power=0.6,
        begin=0.5 * iters,
        end=0.95 * iters,
        by_epoch=False,
    ),
] if not debug else []

default_hooks = dict(
    runtime_info=dict(type=RuntimeInfoHook),
    timer=dict(type=IterTimerHook),
    logger=dict(
        type=LoggerJSON,
        interval=logger_interval,
        log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        max_keep_ckpts=1 if not debug else 0,
        interval=save_interval if not debug else 0,
        save_best='Perf/mDice' if not debug else None,
        rule='greater' if not debug else None,
        save_last=False if not debug else True),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(
        type=BaseVisHook,
        val_vis_interval=vis_interval if not debug else 1,
        test_vis_interval=vis_interval if not debug else 1),
)

visualizer = dict(
    type=SegViser,
    dim=3,
    plt_figsize=(20,5),
    vis_backends=[dict(type=LocalVisBackend),
                  dict(type=TensorboardVisBackend)])

# torch.dynamo
compile = dict(
    fullgraph=False,
    dynamic=False,
    disable=not Compile,
)

# Distributed Training
runner_type = "ITKITRunner"
if dist:
    launcher = "pytorch"
    if MP_mode == "deepspeed":
        strategy = dict(
            type=DeepSpeedStrategy,
            fp16=dict(
                enabled=True,
                auto_cast=True,
                fp16_master_weights_and_grads=False,
                loss_scale=0,
                loss_scale_window=500,
                hysteresis=2,
                min_loss_scale=1,
                initial_scale_power=15,
            ),
            inputs_to_half=None,
            zero_optimization=dict(
                stage=3,
                allgather_partitions=True,
                reduce_scatter=True,
                allgather_bucket_size=5e7,
                reduce_bucket_size=5e7, # 1e6 available
                overlap_comm=True,
                contiguous_gradients=True,
                cpu_offload=False,
                ignore_unused_parameters=True,
                stage3_gather_16bit_weights_on_model_save=True),
        )
    elif MP_mode == "ddp":
        model_wrapper_cfg = dict(type=RemasteredDDP)
    elif MP_mode == "fsdp":
        strategy = dict(
            type=RemasteredFSDP_Strategy,
            model_wrapper=dict(
                type=MMFullyShardedDataParallel,
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
            ),
        )

else:
    launcher = "none"

# Runtime Environment
env_cfg = dict(
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=4),
    dist_cfg=dict(backend="nccl"),
    allow_tf32=True,
    benchmark=True,
    allow_fp16_reduced_precision_reduction=True,
    allow_bf16_reduced_precision_reduction=True,
    dynamo_cache_size=3,
    dynamo_supress_errors=False,
    dynamo_logging_level="ERROR",
    torch_logging_level="ERROR",
)
log_processor = dict(by_epoch=False)
log_level = 'DEBUG'
tta_model = None
