"""
Neural network modules for PyTorch Lightning framework.

This package contains standalone neural network implementations that can be used
with Lightning task modules. Models are completely decoupled from task logic,
allowing flexible composition and reuse.
"""

from .segformer3d import SegFormer3D

__all__ = [
    'SegFormer3D',
]
