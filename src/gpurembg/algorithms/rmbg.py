from __future__ import annotations

from .onnx_base import ONNXMattingModel

__all__ = ["RMBG14Matting"]


class RMBG14Matting(ONNXMattingModel):
    MODEL_NAME = "rmbg-1.4"
    WEIGHTS_URL = (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "rmbg-1.4.onnx"
    )
    SUPPORTS_FP16 = False
    DEFAULT_SIZE = 2048
