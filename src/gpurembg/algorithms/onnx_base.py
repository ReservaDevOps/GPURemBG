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
        orig_h, orig_w = image.height, image.width
        input_size = self.DEFAULT_SIZE

        scale = min(1.0, input_size / float(max(orig_h, orig_w)))
        resized_h = max(int(round(orig_h * scale)), 1)
        resized_w = max(int(round(orig_w * scale)), 1)
        resized = image.resize((resized_w, resized_h), Image.BILINEAR)

        canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        pad_top = (input_size - resized_h) // 2
        pad_left = (input_size - resized_w) // 2
        canvas.paste(resized, (pad_left, pad_top))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD),
            ]
        )
        tensor = transform(canvas).unsqueeze(0)
        return PreprocessResult(
            tensor=tensor.to(self.device),
            meta={
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64, device=self.device),
                "resized_size": torch.tensor([resized_h, resized_w], dtype=torch.int64, device=self.device),
                "pad": torch.tensor([pad_top, pad_left], dtype=torch.int64, device=self.device),
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
        resized_h, resized_w = meta["resized_size"].cpu().tolist()
        pad_top, pad_left = meta["pad"].cpu().tolist()

        alpha = alpha[
            pad_top : pad_top + resized_h,
            pad_left : pad_left + resized_w,
        ]

        alpha = F.interpolate(
            alpha.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        alpha_img = (alpha.cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(alpha_img, mode="L")
