from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Optional

import torch
from PIL import Image

from ..utils.downloads import download_file


@dataclass
class PreprocessResult:
    tensor: torch.Tensor
    meta: Dict[str, torch.Tensor]


class MattingModel(abc.ABC):
    """
    Abstract base class for GPU background matting models.
    """

    MODEL_NAME: ClassVar[str]
    WEIGHTS_URL: ClassVar[Optional[str]]
    WEIGHTS_SHA256: ClassVar[Optional[str]] = None
    DEFAULT_SIZE: ClassVar[int] = 512
    SUPPORTS_FP16: ClassVar[bool] = True
    WEIGHTS_EXTENSION: ClassVar[str] = ".pth"
    WEIGHTS_GDRIVE_ID: ClassVar[Optional[str]] = None

    def __init__(
        self,
        weights_root: Path,
        device: torch.device,
        use_fp16: bool = True,
    ) -> None:
        self.device = device
        self.use_fp16 = use_fp16 and self.SUPPORTS_FP16
        self.model = self._load(weights_root)
        self.model.eval()

    @classmethod
    def weights_path(cls, root: Path) -> Path:
        return root / f"{cls.MODEL_NAME}{cls.WEIGHTS_EXTENSION}"

    @classmethod
    def ensure_weights(cls, root: Path) -> Path:
        path = cls.weights_path(root)
        if path.exists() and (not cls.WEIGHTS_SHA256 or path.stat().st_size > 0):
            if not cls.WEIGHTS_SHA256:
                return path

            from ..utils.downloads import sha256_file

            if sha256_file(path) == cls.WEIGHTS_SHA256.lower():
                return path

        last_exc: Optional[Exception] = None

        if getattr(cls, "WEIGHTS_URL", None):
            try:
                return download_file(cls.WEIGHTS_URL, path, cls.WEIGHTS_SHA256)
            except Exception as exc:  # pragma: no cover - network failure is runtime only
                last_exc = exc

        if cls.WEIGHTS_GDRIVE_ID:
            from ..utils.downloads import download_google_drive_file

            return download_google_drive_file(
                cls.WEIGHTS_GDRIVE_ID,
                path,
                cls.WEIGHTS_SHA256,
            )

        if last_exc is not None:
            raise last_exc

        raise RuntimeError(f"No download source available for model '{cls.MODEL_NAME}'.")

    def _load(self, root: Path) -> torch.nn.Module:
        weights = self.ensure_weights(root)
        state = torch.load(weights, map_location="cpu")
        model = self.build_model()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict mismatch for {self.MODEL_NAME}. "
                f"Missing keys: {missing}, unexpected keys: {unexpected}"
            )
        model.to(self.device)
        return model

    @abc.abstractmethod
    def build_model(self) -> torch.nn.Module:
        ...

    @abc.abstractmethod
    def preprocess(self, image: Image.Image) -> PreprocessResult:
        ...

    @abc.abstractmethod
    def extract_alpha(self, raw_output: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def postprocess(
        self, alpha_pred: torch.Tensor, meta: Dict[str, torch.Tensor]
    ) -> Image.Image:
        ...

    def forward(self, image: Image.Image) -> Image.Image:
        tensors = self.preprocess(image)

        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=self.use_fp16, device_type="cuda"
        ):
            raw = self.model(tensors.tensor)

        alpha = self.extract_alpha(raw)
        return self.postprocess(alpha, tensors.meta)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.model.parameters()
