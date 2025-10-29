from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import MattingModel, PreprocessResult

__all__ = ["U2NETMatting", "U2NETPMatting"]


def _conv_block(ch_in: int, ch_out: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(ch_out),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(ch_out),
        torch.nn.ReLU(inplace=True),
    )


class RSUBlock(torch.nn.Module):
    """
    Re-implementation of the RSU blocks used by U^2-Net.
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.in_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()

        for i in range(depth):
            inch = out_ch if i == 0 else mid_ch
            self.encoders.append(_conv_block(inch, mid_ch))

        for i in range(depth - 1):
            self.decoders.append(_conv_block(mid_ch * 2, mid_ch))

        self.bottom = _conv_block(mid_ch, mid_ch)
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(mid_ch * 2, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

        self.pool = torch.nn.MaxPool2d(2, ceil_mode=True)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        hx1 = self.in_conv(hx)

        downs = []
        h = hx1

        for enc in self.encoders:
            h = enc(h)
            downs.append(h)
            h = self.pool(h)

        h = self.bottom(h)

        for i, dec in enumerate(self.decoders):
            skip = downs[-(i + 2)]
            h = self.upsample(h)
            if h.shape[-1] != skip.shape[-1] or h.shape[-2] != skip.shape[-2]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=True)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = self.upsample(h)
        if h.shape[-1] != hx1.shape[-1] or h.shape[-2] != hx1.shape[-2]:
            h = F.interpolate(h, size=hx1.shape[2:], mode="bilinear", align_corners=True)
        h = torch.cat([h, hx1], dim=1)
        return self.out(h) + hx1


class U2NETImpl(torch.nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 64) -> None:
        super().__init__()

        self.stage1 = RSUBlock(in_ch, base, base, depth=7)
        self.pool1 = torch.nn.MaxPool2d(2, ceil_mode=True)

        self.stage2 = RSUBlock(base, base, base, depth=6)
        self.pool2 = torch.nn.MaxPool2d(2, ceil_mode=True)

        self.stage3 = RSUBlock(base, 2 * base, 2 * base, depth=5)
        self.pool3 = torch.nn.MaxPool2d(2, ceil_mode=True)

        self.stage4 = RSUBlock(2 * base, 4 * base, 4 * base, depth=4)
        self.pool4 = torch.nn.MaxPool2d(2, ceil_mode=True)

        self.stage5 = RSUBlock(4 * base, 8 * base, 8 * base, depth=4)
        self.pool5 = torch.nn.MaxPool2d(2, ceil_mode=True)

        self.stage6 = RSUBlock(8 * base, 8 * base, 8 * base, depth=4)

        self.stage5d = RSUBlock(16 * base, 8 * base, 8 * base, depth=4)
        self.stage4d = RSUBlock(16 * base, 4 * base, 4 * base, depth=4)
        self.stage3d = RSUBlock(8 * base, 2 * base, 2 * base, depth=5)
        self.stage2d = RSUBlock(4 * base, base, base, depth=6)
        self.stage1d = RSUBlock(2 * base, base, base, depth=7)

        self.side1 = torch.nn.Conv2d(base, out_ch, 3, padding=1)
        self.side2 = torch.nn.Conv2d(base, out_ch, 3, padding=1)
        self.side3 = torch.nn.Conv2d(2 * base, out_ch, 3, padding=1)
        self.side4 = torch.nn.Conv2d(4 * base, out_ch, 3, padding=1)
        self.side5 = torch.nn.Conv2d(8 * base, out_ch, 3, padding=1)
        self.side6 = torch.nn.Conv2d(8 * base, out_ch, 3, padding=1)

        self.outconv = torch.nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        hx1 = self.stage1(x)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool3(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool4(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool5(hx5)

        hx6 = self.stage6(hx)

        hx5d = self.stage5d(torch.cat([hx6, hx5], dim=1))
        hx5d = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=True)

        hx4d = self.stage4d(torch.cat([hx5d, hx4], dim=1))
        hx4d = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=True)

        hx3d = self.stage3d(torch.cat([hx4d, hx3], dim=1))
        hx3d = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.stage2d(torch.cat([hx3d, hx2], dim=1))
        hx2d = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.stage1d(torch.cat([hx2d, hx1], dim=1))

        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        d2 = F.interpolate(d2, size=d1.shape[2:], mode="bilinear", align_corners=True)
        d3 = F.interpolate(d3, size=d1.shape[2:], mode="bilinear", align_corners=True)
        d4 = F.interpolate(d4, size=d1.shape[2:], mode="bilinear", align_corners=True)
        d5 = F.interpolate(d5, size=d1.shape[2:], mode="bilinear", align_corners=True)
        d6 = F.interpolate(d6, size=d1.shape[2:], mode="bilinear", align_corners=True)

        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))
        return d0, d1, d2, d3, d4, d5, d6


class U2NETMatting(MattingModel):
    MODEL_NAME = "u2net"
    WEIGHTS_URL = (
        "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
    )
    SUPPORTS_FP16 = True
    DEFAULT_SIZE = 1024

    def build_model(self) -> torch.nn.Module:
        return U2NETImpl(3, 1, base=64)

    def preprocess(self, image: Image.Image) -> PreprocessResult:
        image = image.convert("RGB")
        orig_size = torch.tensor([image.height, image.width], dtype=torch.int64)

        max_side = float(orig_size.max())
        scale = min(self.DEFAULT_SIZE / max_side, 1.0)
        new_size = (orig_size.float() * scale).to(torch.int64)
        if scale != 1.0:
            image = image.resize((new_size[1].item(), new_size[0].item()), Image.BILINEAR)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        tensor = transform(image).unsqueeze(0).to(self.device)
        return PreprocessResult(
            tensor=tensor,
            meta={
                "orig_size": orig_size.to(self.device),
                "scale": torch.tensor(scale, device=self.device),
            },
        )

    def extract_alpha(self, raw_output: torch.Tensor) -> torch.Tensor:
        if isinstance(raw_output, (list, tuple)):
            pred = raw_output[0]
        else:
            pred = raw_output
        return torch.sigmoid(pred)

    def postprocess(self, alpha_pred: torch.Tensor, meta: Dict[str, torch.Tensor]) -> Image.Image:
        alpha = alpha_pred[0, 0]
        orig_h, orig_w = meta["orig_size"].cpu().tolist()
        alpha = F.interpolate(
            alpha.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        alpha = alpha.clamp(0, 1)
        alpha_img = (alpha.cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(alpha_img, mode="L")


class U2NETPMatting(U2NETMatting):
    MODEL_NAME = "u2netp"
    WEIGHTS_URL = (
        "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2netp.pth"
    )

    def build_model(self) -> torch.nn.Module:
        return U2NETImpl(3, 1, base=16)
