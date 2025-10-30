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

        alpha_np = np.array(alpha_pred, dtype=np.float32) / 255.0

        if self.config.alpha_threshold is not None:
            threshold = max(0.0, min(1.0, self.config.alpha_threshold))
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
                alpha_tensor = TF.gaussian_blur(alpha_tensor, kernel_size=kernel, sigma=self.config.refine_feather)
            alpha_np = alpha_tensor.squeeze().clamp(0, 1).detach().cpu().numpy()

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

        for image_path in sorted(self._iter_images(input_dir)):
            destination = output_dir / (image_path.stem + ".png")
            if destination.exists() and not overwrite:
                continue
            start = time.perf_counter()
            image = Image.open(image_path)
            result = self.remove_background(image)
            result.save(destination)
            end = time.perf_counter()
            timings[str(image_path)] = end - start

        return timings

    @staticmethod
    def _iter_images(path: Path) -> Iterable[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for file in path.rglob("*"):
            if file.suffix.lower() in exts:
                yield file
