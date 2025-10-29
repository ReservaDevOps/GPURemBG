from __future__ import annotations

from .onnx_base import ONNXMattingModel

__all__ = ["ISNetMatting"]


class ISNetMatting(ONNXMattingModel):
    MODEL_NAME = "isnet-general-use"
    WEIGHTS_URL = (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "isnet-general-use.onnx"
    )
    SUPPORTS_FP16 = False
    DEFAULT_SIZE = 1536
