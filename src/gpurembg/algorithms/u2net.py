from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import MattingModel, PreprocessResult

__all__ = [
    "U2NETMatting",
    "U2NETPMatting",
    "U2NETPortraitMatting",
    "U2NETHumanMatting",
]


class REBNCONV(torch.nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1) -> None:
        super().__init__()
        self.conv_s1 = torch.nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=dirate, dilation=dirate
        )
        self.bn_s1 = torch.nn.BatchNorm2d(out_ch)
        self.relu_s1 = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU7(torch.nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode="bilinear", align_corners=True)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=True)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=True)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(torch.nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=True)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=True)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(torch.nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=True)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(torch.nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(torch.nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class U2NET(torch.nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        self.side2 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        self.side3 = torch.nn.Conv2d(128, out_ch, kernel_size=3, padding=1)
        self.side4 = torch.nn.Conv2d(256, out_ch, kernel_size=3, padding=1)
        self.side5 = torch.nn.Conv2d(512, out_ch, kernel_size=3, padding=1)
        self.side6 = torch.nn.Conv2d(512, out_ch, kernel_size=3, padding=1)

        self.outconv = torch.nn.Conv2d(out_ch * 6, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode="bilinear", align_corners=True)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=True)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=True)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

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

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return d0, d1, d2, d3, d4, d5, d6


class U2NETP(torch.nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__()
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        self.side2 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        self.side3 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        self.side4 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        self.side5 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        self.side6 = torch.nn.Conv2d(64, out_ch, kernel_size=3, padding=1)

        self.outconv = torch.nn.Conv2d(out_ch * 6, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode="bilinear", align_corners=True)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode="bilinear", align_corners=True)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode="bilinear", align_corners=True)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

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

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return d0, d1, d2, d3, d4, d5, d6


class U2NETMatting(MattingModel):
    MODEL_NAME = "u2net"
    WEIGHTS_URL = None
    WEIGHTS_GDRIVE_ID = "1ao1ovG_F3fA_kL-2P8VK_z9Dy3km5iQV"
    SUPPORTS_FP16 = False
    DEFAULT_SIZE = 1024

    def build_model(self) -> torch.nn.Module:
        return U2NET(3, 1)

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
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        alpha_img = (alpha.cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(alpha_img, mode="L")


class U2NETPMatting(U2NETMatting):
    MODEL_NAME = "u2netp"
    WEIGHTS_GDRIVE_ID = "1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"
    SUPPORTS_FP16 = False

    def build_model(self) -> torch.nn.Module:
        return U2NETP(3, 1)


class U2NETPortraitMatting(U2NETMatting):
    MODEL_NAME = "u2net_portrait"
    WEIGHTS_GDRIVE_ID = "1IG3HdpcRiDoWNookbncQjeaPN28t90yW"
    SUPPORTS_FP16 = False


class U2NETHumanMatting(U2NETMatting):
    MODEL_NAME = "u2net_human_seg"
    WEIGHTS_GDRIVE_ID = "1N7abitNTB7uHm-bL8hZ5wfQRO_DTrdPp"
    SUPPORTS_FP16 = False
