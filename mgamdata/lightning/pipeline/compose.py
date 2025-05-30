"""
Simple Lightning data transform composition.

This module provides a simplified transform composition class that takes a list
of transform configurations and executes them sequentially.
"""

import importlib
from typing import Any, Callable


class LightningCompose:
    """
    Simple transform composition for Lightning datasets.
    
    Takes a list of transform configurations and builds a callable pipeline.
    Each transform config should be a dictionary with 'type' key specifying
    the transform class name, and other keys as parameters.
    """
    
    def __init__(self, transforms: list[dict[str, Any]]):
        """
        Initialize the transform composition.
        
        Args:
            transforms: List of transform configurations
                Format: [{"type": "TransformName", "param1": value1, ...}, ...]
        """
        self.transform_configs = transforms
        self.transforms = self._build_transforms(transforms)
    
    def _build_transforms(self, transform_configs: list[dict[str, Any]]) -> list[Callable]:
        """
        Build transform instances from configurations.
        
        Args:
            transform_configs: List of transform configurations
            
        Returns:
            List of callable transform instances
        """
        transforms = []
        for config in transform_configs:
            transform = self._build_single_transform(config)
            transforms.append(transform)
        return transforms
    
    def _build_single_transform(self, config: dict[str, Any]) -> Callable:
        """
        Build a single transform from configuration.
        
        Args:
            config: Transform configuration dictionary
            
        Returns:
            Callable transform instance
        """
        if not isinstance(config, dict) or 'type' not in config:
            raise ValueError(f"Transform config must be a dict with 'type' key, got: {config}")
        
        transform_type = config['type']
        transform_params = {k: v for k, v in config.items() if k != 'type'}
        
        # Try to import and instantiate the transform
        transform_class = self._get_transform_class(transform_type)
        return transform_class(**transform_params)
    
    def _get_transform_class(self, transform_type: str):
        """
        Get transform class by name.
        
        This method looks for transform classes in common locations:
        1. mgamdata.process module
        2. torchvision.transforms module
        3. Current module globals
        
        Args:
            transform_type: Name of the transform class
            
        Returns:
            Transform class
        """
        # Try mgamdata.process first (our custom transforms)
        try:
            module = importlib.import_module('mgamdata.process')
            if hasattr(module, transform_type):
                return getattr(module, transform_type)
        except ImportError:
            pass
        
        # Try torchvision transforms
        try:
            module = importlib.import_module('torchvision.transforms')
            if hasattr(module, transform_type):
                return getattr(module, transform_type)
        except ImportError:
            pass
        
        # Try torch transforms
        try:
            module = importlib.import_module('torch.nn')
            if hasattr(module, transform_type):
                return getattr(module, transform_type)
        except ImportError:
            pass
        
        raise ValueError(f"Transform '{transform_type}' not found in any known module")
    
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Apply all transforms to the sample.
        
        Args:
            sample: Input sample dictionary
            
        Returns:
            Transformed sample dictionary
        """
        for transform in self.transforms:
            try:
                sample = transform(sample)
            except Exception as e:
                print(f"Error applying transform {transform.__class__.__name__}: {e}")
                raise
        
        return sample
    
    def __repr__(self) -> str:
        """String representation of the compose object."""
        transform_names = [config.get('type', 'Unknown') for config in self.transform_configs]
        return f"LightningCompose({transform_names})"


# Common preprocessing pipelines as simple lists
COMMON_PIPELINES = {
    "basic_3d_segmentation": [
        {"type": "LoadCTPreCroppedSampleFromNpz", "load_type": ["img", "anno"]},
        {"type": "WindowSet", "level": 40, "width": 400},
        {"type": "InstanceNorm", "eps": 1e-6},
        {"type": "AutoPad", "size": (128, 128, 128), "dim": "3d"},
        {"type": "TypeConvert", "key": ["img", "gt_seg_map"], "dtype": "float32"},
    ],
    
    "augmented_3d_segmentation": [
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
    
    "minimal_2d_segmentation": [
        {"type": "LoadImgFromOpenCV"},
        {"type": "LoadAnnoFromOpenCV"},
        {"type": "ForceResize", "image_size": (256, 256), "label_size": (256, 256)},
        {"type": "Normalize", "mode": "instance", "size": (256, 256)},
        {"type": "TypeConvert", "key": ["img", "gt_seg_map"], "dtype": "float32"},
    ],
    
    "brain_mri_preprocessing": [
        {"type": "LoadBraTs2024PreCroppedSample", "load_type": ["t1c", "t1n", "t2f", "t2w", "anno"]},
        {"type": "InstanceNorm", "eps": 1e-6},
        {"type": "CenterCrop3D", "size": [128, 128, 128]},
        {"type": "ExpandOneHot", "num_classes": 4},
        {"type": "TypeConvert", "key": ["img", "gt_seg_map"], "dtype": "float32"},
    ]
}


def get_pipeline(name: str) -> list[dict[str, Any]]:
    """
    Get a predefined pipeline by name.
    
    Args:
        name: Name of the pipeline
        
    Returns:
        List of transform configurations
    """
    if name not in COMMON_PIPELINES:
        raise ValueError(f"Pipeline '{name}' not found. Available: {list(COMMON_PIPELINES.keys())}")
    return COMMON_PIPELINES[name].copy()
