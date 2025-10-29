from __future__ import annotations

from .onnx_base import ONNXMattingModel

__all__ = ["RMBG14Matting"]


class RMBG14Matting(ONNXMattingModel):
    MODEL_NAME = "rmbg-1.4"
    WEIGHTS_URL = (
        "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.onnx?download=1"
    )
    SUPPORTS_FP16 = False
    DEFAULT_SIZE = 1024
