"""Anime2Sketch model inference helpers.

The model architecture follows the public Anime2Sketch PyTorch generator
(https://github.com/Mukosame/Anime2Sketch, MIT License), but torch is imported
lazily so the rest of the app keeps working without the optional deep-learning
dependency.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


_MODEL_CACHE = {}


def anime2sketch_to_gray(
    rgb: np.ndarray,
    model_path: str,
    input_size: int = 512,
    device_name: str = "auto",
) -> np.ndarray:
    """Run Anime2Sketch and return a grayscale uint8 line-art image."""
    if not model_path:
        raise RuntimeError("请先选择 Anime2Sketch 模型权重文件（netG.pth 或 improved.bin）。")

    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Anime2Sketch 模型文件不存在：{path}")

    torch, nn, f = _import_torch()
    device = _select_device(torch, device_name)
    model = _get_model(torch, nn, f, path, device)

    h, w = rgb.shape[:2]
    size = max(32, int(input_size))
    pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5

    with torch.no_grad():
        out = model(tensor.to(device)).detach().cpu().float().numpy()[0]

    if out.shape[0] == 1:
        gray = out[0]
    else:
        gray = out.mean(axis=0)

    gray_u8 = np.clip((gray + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    if gray_u8.shape[:2] != (h, w):
        gray_u8 = cv2.resize(gray_u8, (w, h), interpolation=cv2.INTER_CUBIC)
    return gray_u8


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as f
    except ImportError as exc:
        raise RuntimeError(
            "Anime2Sketch 模式需要安装 PyTorch。请先运行：pip install torch"
        ) from exc
    return torch, nn, f


def _select_device(torch, device_name: str):
    name = (device_name or "auto").strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 没有可用 CUDA，请把 Anime2Sketch 设备改为 auto 或 cpu。")
    if name not in {"cpu", "cuda"}:
        raise RuntimeError("Anime2Sketch 设备只能填写 auto、cpu 或 cuda。")
    return torch.device(name)


def _get_model(torch, nn, f, model_path: Path, device):
    key = (str(model_path.resolve()), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    model = _build_unet_generator(torch, nn, f)
    try:
        ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise RuntimeError("无法识别 Anime2Sketch 权重格式。")

    ckpt = _strip_module_prefix(dict(ckpt))
    if model_path.suffix.lower() == ".bin":
        _replace_deconv_with_upsample(model, torch, nn, f)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _strip_module_prefix(ckpt: dict) -> dict:
    cleaned = {}
    for key, value in ckpt.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def _build_unet_generator(torch, nn, f):
    class UnetGenerator(nn.Module):
        def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d):
            super().__init__()
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=None,
                norm_layer=norm_layer,
                innermost=True,
            )
            for _ in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(
                    ngf * 8,
                    ngf * 8,
                    input_nc=None,
                    submodule=unet_block,
                    norm_layer=norm_layer,
                )
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            self.model = UnetSkipConnectionBlock(
                output_nc,
                ngf,
                input_nc=input_nc,
                submodule=unet_block,
                outermost=True,
                norm_layer=norm_layer,
            )

        def forward(self, input_tensor):
            return self.model(input_tensor)

    class UnetSkipConnectionBlock(nn.Module):
        def __init__(
            self,
            outer_nc,
            inner_nc,
            input_nc=None,
            submodule=None,
            outermost=False,
            innermost=False,
            norm_layer=nn.BatchNorm2d,
        ):
            super().__init__()
            self.outermost = outermost
            if isinstance(norm_layer, functools.partial):
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d
            if input_nc is None:
                input_nc = outer_nc

            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = norm_layer(inner_nc)
            uprelu = nn.ReLU(True)
            upnorm = norm_layer(outer_nc)

            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
                model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
                model = [downrelu, downconv, uprelu, upconv, upnorm]
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
                model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            self.model = nn.Sequential(*model)

        def forward(self, x):
            if self.outermost:
                return self.model(x)
            return torch.cat([x, self.model(x)], 1)

    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    return UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer)


def _replace_deconv_with_upsample(model, torch, nn, f):
    base = model.model.model[1]
    for _ in range(6):
        inc = base.model[5].in_channels
        outc = base.model[5].out_channels
        base.model[5] = _make_upsample(torch, nn, f, inc, outc)
        base = base.model[3]


def _make_upsample(torch, nn, f, inc: int, outc: int):
    class Smooth(nn.Module):
        def __init__(self):
            super().__init__()
            kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            kernel = torch.tensor([[kernel]], dtype=torch.float)
            kernel /= kernel.sum()
            self.register_buffer("kernel", kernel)
            self.pad = nn.ReplicationPad2d(1)

        def forward(self, x):
            b, c, h, w = x.shape
            x = x.view(-1, 1, h, w)
            x = self.pad(x)
            x = f.conv2d(x, self.kernel)
            return x.view(b, c, h, w)

    class Upsample(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
            self.smooth = Smooth()
            self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
            self.mlp = nn.Sequential(
                nn.Conv2d(outc, 4 * outc, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(4 * outc, outc, kernel_size=1, stride=1, padding=0),
            )

        def forward(self, x):
            x = self.smooth(self.up(x))
            x = self.conv(x)
            return self.mlp(x) + x

    return Upsample()
