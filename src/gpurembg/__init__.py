"""
GPU-accelerated background removal toolkit.

This package exposes high-performance matting backends that run exclusively
on NVIDIA GPUs and leverage Tensor Core friendly inference paths (FP16/AMP).
"""

from .pipeline import BackgroundRemover, MattingConfig

__all__ = ["BackgroundRemover", "MattingConfig"]
