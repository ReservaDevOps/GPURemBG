from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .algorithms.base import MattingModel
from .algorithms.onnx_base import ONNXMattingModel
from .algorithms.isnet import ISNetMatting
from .algorithms.rmbg import RMBG14Matting
from .algorithms.u2net import (
    U2NETMatting,
    U2NETPMatting,
    U2NETPortraitMatting,
    U2NETHumanMatting,
)


MODEL_REGISTRY: Dict[str, type[MattingModel]] = {
    "u2net": U2NETMatting,
    "u2netp": U2NETPMatting,
    "u2net-portrait": U2NETPortraitMatting,
    "u2net-human": U2NETHumanMatting,
    "isnet": ISNetMatting,
    "rmbg14": RMBG14Matting,
}


@dataclass
class MattingConfig:
    model_name: str = "u2net"
    weights_dir: Path = field(default_factory=lambda: Path("~/.cache/gpurembg").expanduser())
    use_fp16: bool = True
    device: str = "cuda:0"
    alpha_threshold: Optional[float] = None
    refine_foreground: bool = True
    refine_dilate: int = 0
    refine_feather: int = 0
    batch_size: int = 1
    use_tensorrt: bool = False

    def torch_device(self) -> torch.device:
        return torch.device(self.device)


class BackgroundRemover:
    def __init__(self, config: MattingConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Install PyTorch with CUDA support.")

        self.config = config
        device = config.torch_device()

        torch.backends.cudnn.benchmark = True

        if config.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{config.model_name}'. Choices: {list(MODEL_REGISTRY)}")

        model_cls = MODEL_REGISTRY[config.model_name]
        if issubclass(model_cls, ONNXMattingModel):
            self.model = model_cls(
                config.weights_dir,
                device,
                config.use_fp16,
                use_tensorrt=config.use_tensorrt,
            )
        else:
            self.model = model_cls(
                config.weights_dir,
                device,
                config.use_fp16,
            )
        self.device = device

    def predict_alpha(self, image: Image.Image) -> Image.Image:
        return self.model.forward(image)

    def remove_background(self, image: Image.Image) -> Image.Image:
        image_rgb = image.convert("RGB")
        alpha_pred = self.predict_alpha(image_rgb)
        return self._compose_image(image_rgb, alpha_pred)

    def predict_alpha_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        return self.model.forward_batch(images)

    def _process_alpha_np(self, alpha_np: np.ndarray) -> np.ndarray:
        alpha_np = np.clip(alpha_np, 0.0, 1.0)

        threshold = self.config.alpha_threshold
        if threshold is not None:
            threshold = max(0.0, min(1.0, threshold))
            alpha_np = (alpha_np >= threshold).astype(np.float32)

        if self.config.refine_dilate > 0 or self.config.refine_feather > 0:
            alpha_tensor = (
                torch.from_numpy(alpha_np)
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            if self.config.refine_dilate > 0:
                for _ in range(self.config.refine_dilate):
                    alpha_tensor = F.max_pool2d(alpha_tensor, kernel_size=3, stride=1, padding=1)
            if self.config.refine_feather > 0:
                kernel = 2 * self.config.refine_feather + 1
                alpha_tensor = TF.gaussian_blur(
                    alpha_tensor, kernel_size=kernel, sigma=self.config.refine_feather
                )
            alpha_np = alpha_tensor.squeeze().clamp(0, 1).detach().cpu().numpy()

        return alpha_np

    def _compose_image(self, image_rgb: Image.Image, alpha_image: Image.Image) -> Image.Image:
        alpha_np = np.array(alpha_image, dtype=np.float32) / 255.0
        alpha_np = self._process_alpha_np(alpha_np)
        alpha = Image.fromarray((alpha_np * 255).astype("uint8"), mode="L")

        if self.config.refine_foreground:
            composed = self._blend_foreground(image_rgb, alpha)
        else:
            composed = image_rgb.convert("RGBA")
            composed.putalpha(alpha)
        return composed

    @staticmethod
    def _blend_foreground(image: Image.Image, alpha: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
        return rgba

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        overwrite: bool = False,
    ) -> Dict[str, float]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timings: Dict[str, float] = {}

        batch_size = max(1, self.config.batch_size)
        batch: List[Path] = []

        def flush(current_batch: List[Path]) -> None:
            if not current_batch:
                return
            batch_start = time.perf_counter()
            images_rgb: List[Image.Image] = []
            for path in current_batch:
                with Image.open(path) as img:
                    images_rgb.append(img.convert("RGB"))

            alphas = self.predict_alpha_batch(images_rgb)
            results = [self._compose_image(img_rgb, alpha) for img_rgb, alpha in zip(images_rgb, alphas)]
            batch_end = time.perf_counter()
            per_image = (batch_end - batch_start) / len(current_batch)

            for path, result in zip(current_batch, results):
                destination = output_dir / (path.stem + ".png")
                result.save(destination)
                timings[str(path)] = per_image

        for image_path in sorted(self._iter_images(input_dir)):
            destination = output_dir / (image_path.stem + ".png")
            if destination.exists() and not overwrite:
                continue
            batch.append(image_path)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []

        if batch:
            flush(batch)

        return timings

    @staticmethod
    def _iter_images(path: Path) -> Iterable[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for file in path.rglob("*"):
            if file.suffix.lower() in exts:
                yield file
