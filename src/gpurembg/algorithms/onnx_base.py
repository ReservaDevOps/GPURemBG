from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import MattingModel, PreprocessResult


class ONNXMattingModel(MattingModel):
    """
    Shared ONNXRuntime-backed matting implementation.
    """

    DEFAULT_SIZE: int = 1024
    NORMALIZE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    NORMALIZE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    OUTPUT_SIGMOID: bool = True
    WEIGHTS_EXTENSION: str = ".onnx"

    def __init__(self, weights_root: Path, device: torch.device, use_fp16: bool = False) -> None:
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        super().__init__(weights_root, device, use_fp16=False)

    def build_model(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def _load(self, root: Path) -> torch.nn.Module:
        onnx_path = self.ensure_weights(root)
        device_id = self.device.index if self.device.index is not None else 0
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_use_max_workspace": "1",
                    "do_copy_in_default_stream": "1",
                },
            )
        ]
        session = ort.InferenceSession(onnx_path.as_posix(), providers=providers)
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "onnxruntime did not initialize the CUDAExecutionProvider. "
                "Install `onnxruntime-gpu` and ensure the NVIDIA driver/CUDA stack is available."
            )
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        return torch.nn.Identity()

    def preprocess(self, image: Image.Image) -> PreprocessResult:
        image = image.convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((self.DEFAULT_SIZE, self.DEFAULT_SIZE), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD),
            ]
        )
        tensor = transform(image).unsqueeze(0)
        return PreprocessResult(
            tensor=tensor.to(self.device),
            meta={
                "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64, device=self.device),
            },
        )

    def extract_alpha(self, raw_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ONNXMattingModel overrides forward; extract_alpha unused.")

    def forward(self, image: Image.Image) -> Image.Image:
        if self.session is None or self.input_name is None or self.output_name is None:
            raise RuntimeError("ONNX session not initialized.")

        tensors = self.preprocess(image)
        ort_inputs = {
            self.input_name: tensors.tensor.detach()
            .to("cpu")
            .numpy()
            .astype(np.float32)
        }
        outputs = self.session.run([self.output_name], ort_inputs)[0]
        alpha = torch.from_numpy(outputs).to(self.device)
        if self.OUTPUT_SIGMOID:
            alpha = torch.sigmoid(alpha)
        return self.postprocess(alpha, tensors.meta)

    def postprocess(self, alpha_pred: torch.Tensor, meta: Dict[str, torch.Tensor]) -> Image.Image:
        alpha = alpha_pred[0, 0]
        orig_h, orig_w = meta["orig_size"].cpu().tolist()
        alpha = F.interpolate(
            alpha.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        alpha_img = (alpha.cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(alpha_img, mode="L")
