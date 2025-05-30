"""
Simplified Lightning data processing pipeline module.

This module provides a simple way to configure data preprocessing pipelines
using plain Python lists of transform configurations.
"""

from .compose import LightningCompose, COMMON_PIPELINES, get_pipeline

__all__ = [
    'LightningCompose',
    'COMMON_PIPELINES', 
    'get_pipeline',
]
