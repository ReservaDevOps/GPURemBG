from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict, List, Sequence, Tuple

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

    def __init__(
        self,
        weights_root: Path,
        device: torch.device,
        use_fp16: bool = False,
        use_tensorrt: bool = False,
    ) -> None:
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.use_tensorrt = use_tensorrt
        super().__init__(weights_root, device, use_fp16=False)

    def build_model(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def _load(self, root: Path) -> torch.nn.Module:
        onnx_path = self.ensure_weights(root)
        device_id = self.device.index if self.device.index is not None else 0

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self._build_providers(device_id, tensorrt=self.use_tensorrt)
        provider_names = [name for name, _ in providers]
        provider_options = [options for _, options in providers]

        try:
            session = ort.InferenceSession(
                onnx_path.as_posix(),
                sess_options=session_options,
                providers=provider_names,
                provider_options=provider_options,
            )
        except Exception as exc:
            if self.use_tensorrt:
                logging.warning(
                    "TensorRT provider failed to initialize; retrying without TensorRT."
                )
                providers = self._build_providers(device_id, tensorrt=False)
                provider_names = [name for name, _ in providers]
                provider_options = [options for _, options in providers]
                session = ort.InferenceSession(
                    onnx_path.as_posix(),
                    sess_options=session_options,
                    providers=provider_names,
                    provider_options=provider_options,
                )
            else:
                raise
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "onnxruntime did not initialize the CUDAExecutionProvider. "
                "Install `onnxruntime-gpu` and ensure the NVIDIA driver/CUDA stack is available."
            )
        if self.use_tensorrt and "TensorrtExecutionProvider" not in session.get_providers():
            logging.warning(
                "TensorRT Execution Provider requested but not available; falling back to CUDA."
            )
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        return torch.nn.Identity()

    def _build_providers(
        self, device_id: int, *, tensorrt: bool
    ) -> List[Tuple[str, Dict[str, str]]]:
        cuda_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
        }
        providers: List[Tuple[str, Dict[str, str]]] = [("CUDAExecutionProvider", cuda_options)]

        if tensorrt:
            trt_options = {
                "device_id": device_id,
                "trt_fp16_enable": "True",
                "trt_max_workspace_size": str(1 << 30),
            }
            providers.insert(0, ("TensorrtExecutionProvider", trt_options))

        return providers

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

    def forward_batch(self, images: Sequence[Image.Image]) -> List[Image.Image]:
        if self.session is None or self.input_name is None or self.output_name is None:
            raise RuntimeError("ONNX session not initialized.")

        input_shape = self.session.get_inputs()[0].shape
        static_batch = None
        if input_shape:
            first_dim = input_shape[0]
            if isinstance(first_dim, int):
                static_batch = first_dim

        if static_batch is not None and static_batch not in (0, len(images)):
            return [self.forward(image) for image in images]

        tensors = [self.preprocess(image) for image in images]
        batch = torch.cat([item.tensor for item in tensors], dim=0)
        ort_inputs = {
            self.input_name: batch.detach().to("cpu").numpy().astype(np.float32)
        }
        outputs = self.session.run([self.output_name], ort_inputs)[0]

        alphas = torch.from_numpy(outputs).to(self.device)
        if self.OUTPUT_SIGMOID:
            alphas = torch.sigmoid(alphas)

        results: List[Image.Image] = []
        for idx, meta in enumerate([item.meta for item in tensors]):
            alpha = alphas[idx : idx + 1]
            results.append(self.postprocess(alpha, meta))
        return results

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
