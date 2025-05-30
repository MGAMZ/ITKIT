"""
Pipeline configuration system for Lightning data processing.

This module provides a flexible way to configure data preprocessing pipelines
by accepting lists of preprocessing steps as parameters. It bridges the gap
between the existing BaseTransform infrastructure and Lightning components.
"""

from typing import Any
from dataclasses import dataclass, field
from .compose import LightningCompose


@dataclass
class PipelineConfig:
    """
    Configuration class for data preprocessing pipelines.
    
    This class encapsulates the configuration for a complete data processing
    pipeline, including preprocessing steps, validation, and runtime options.
    """
    
    # Core pipeline configuration
    steps: list[dict[str, Any]] = field(default_factory=list)
    
    # Pipeline metadata
    name: str = "default_pipeline"
    description: str = ""
    
    # Runtime options
    enable_caching: bool = False
    cache_size: int = 1000
    num_workers: int = 0
    
    # Validation options
    validate_keys: list[str] = field(default_factory=lambda: ['img', 'gt_seg_map'])
    strict_validation: bool = False
    
    # Error handling
    skip_on_error: bool = False
    error_log_file: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.steps:
            raise ValueError("Pipeline must have at least one processing step")
        
        if self.enable_caching and self.cache_size <= 0:
            raise ValueError("Cache size must be positive when caching is enabled")


def create_pipeline_from_config(config: PipelineConfig) -> LightningCompose:
    """
    Create a pipeline from a PipelineConfig object.
    
    Args:
        config: Pipeline configuration object
        
    Returns:
        Configured pipeline ready for use
    """
    from .compose import LightningCompose
    
    return LightningCompose(
        transforms=config.steps,
        validate_keys=config.validate_keys,
        strict_validation=config.strict_validation,
        skip_on_error=config.skip_on_error,
        enable_caching=config.enable_caching,
        cache_size=config.cache_size
    )


def create_pipeline_from_steps(
    steps: list[dict[str, Any]],
    name: str = "custom_pipeline",
    **kwargs
) -> 'LightningCompose':
    """
    Create a pipeline directly from a list of processing steps.
    
    Args:
        steps: List of transform configurations or instances
        name: Name for the pipeline
        **kwargs: Additional configuration options
        
    Returns:
        Configured pipeline ready for use
    """
    config = PipelineConfig(
        steps=steps,
        name=name,
        **kwargs
    )
    return create_pipeline_from_config(config)


# Common preprocessing pipeline configurations
COMMON_PREPROCESS_CONFIGS = {
    "basic_3d_segmentation": PipelineConfig(
        name="basic_3d_segmentation",
        description="Basic 3D medical image segmentation preprocessing",
        steps=[
            {"type": "LoadCTPreCroppedSampleFromNpz", "load_type": ["img", "anno"]},
            {"type": "WindowSet", "level": 40, "width": 400},
            {"type": "InstanceNorm", "eps": 1e-6},
            {"type": "AutoPad", "size": (128, 128, 128), "dim": "3d"},
            {"type": "TypeConvert", "key": ["img", "gt_seg_map"], "dtype": "float32"},
        ],
        validate_keys=["img", "gt_seg_map"]
    ),
    
    "augmented_3d_segmentation": PipelineConfig(
        name="augmented_3d_segmentation", 
        description="Augmented 3D segmentation with geometric and intensity transforms",
        steps=[
            {"type": "LoadCTPreCroppedSampleFromNpz", "load_type": ["img", "anno"]},
            {"type": "WindowSet", "level": 40, "width": 400},
            {"type": "InstanceNorm", "eps": 1e-6},
            {"type": "RandomCrop3D", "crop_size": (96, 96, 96), "cat_max_ratio": 1.0},
            {"type": "RandomFlip3D", "axis": [0, 1, 2], "prob": 0.5},
            {"type": "RandomRotate3D", "degree": 15, "prob": 0.3},
            {"type": "RandomGaussianBlur3D", "max_sigma": 1.0, "prob": 0.2},
            {"type": "AutoPad", "size": (128, 128, 128), "dim": "3d"},
            {"type": "TypeConvert", "key": ["img", "gt_seg_map"], "dtype": "float32"},
        ],
        validate_keys=["img", "gt_seg_map"]
    ),
    
    "minimal_2d_segmentation": PipelineConfig(
        name="minimal_2d_segmentation",
        description="Minimal 2D medical image segmentation preprocessing",
        steps=[
            {"type": "LoadImgFromOpenCV"},
            {"type": "LoadAnnoFromOpenCV"},
            {"type": "ForceResize", "image_size": (256, 256), "label_size": (256, 256)},
            {"type": "Normalize", "mode": "instance", "size": (256, 256)},
            {"type": "TypeConvert", "key": ["img", "gt_seg_map"], "dtype": "float32"},
        ],
        validate_keys=["img", "gt_seg_map"]
    ),
    
    "brain_mri_preprocessing": PipelineConfig(
        name="brain_mri_preprocessing",
        description="Brain MRI-specific preprocessing pipeline",
        steps=[
            {"type": "LoadBraTs2024PreCroppedSample", "load_type": ["t1c", "t1n", "t2f", "t2w", "anno"]},
            {"type": "InstanceNorm", "eps": 1e-6},
            {"type": "CenterCrop3D", "size": [128, 128, 128]},
            {"type": "ExpandOneHot", "num_classes": 4},
            {"type": "TypeConvert", "key": ["img", "gt_seg_map"], "dtype": "float32"},
        ],
        validate_keys=["img", "gt_seg_map"]
    )
}


class PipelineConfigRegistry:
    """
    Registry for managing pipeline configurations.
    
    This class provides a centralized way to register, retrieve, and manage
    preprocessing pipeline configurations across the framework.
    """
    
    _configs: dict[str, PipelineConfig] = {}
    
    @classmethod
    def register(cls, name: str, config: PipelineConfig) -> None:
        """Register a pipeline configuration."""
        cls._configs[name] = config
    
    @classmethod
    def get(cls, name: str) -> PipelineConfig:
        """Get a registered pipeline configuration."""
        if name not in cls._configs:
            raise KeyError(f"Pipeline configuration '{name}' not found")
        return cls._configs[name]
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available pipeline configurations."""
        return list(cls._configs.keys())
    
    @classmethod
    def create_pipeline(cls, name: str) -> 'LightningCompose':
        """Create a pipeline from a registered configuration."""
        config = cls.get(name)
        return create_pipeline_from_config(config)


# Register common configurations
for name, config in COMMON_PREPROCESS_CONFIGS.items():
    PipelineConfigRegistry.register(name, config)
