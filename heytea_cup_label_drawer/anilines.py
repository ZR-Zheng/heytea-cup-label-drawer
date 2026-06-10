"""AniLines Detail inference adapter.

Architecture and preprocessing follow AniLines by Zhenglin Pan:
https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor (MIT License).
"""

from pathlib import Path

import cv2
import numpy as np

from .anime2sketch import _import_torch, _select_device, _strip_module_prefix


_MODEL_CACHE = {}


def anilines_to_gray(rgb: np.ndarray, model_path: str, input_size: int = 512, device_name: str = "auto") -> np.ndarray:
    if not model_path:
        raise RuntimeError("请先导入 AniLines Detail 模型权重（detail.pth）。")
    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"AniLines 模型文件不存在：{path}")

    torch, nn, f = _import_torch()
    device = _select_device(torch, device_name)
    model = _get_model(torch, nn, f, path, device)

    h, w = rgb.shape[:2]
    size = max(128, int(input_size))
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    inputs = np.stack([gray, sobel]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(inputs).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor).detach().cpu().float().numpy()[0, 0]
    result = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC) if result.shape != (h, w) else result


def _get_model(torch, nn, f, path: Path, device):
    key = (str(path.resolve()), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = _build_line_extractor(torch, nn, f)
    try:
        state = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(_strip_module_prefix(dict(state)))
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _build_line_extractor(torch, nn, f):
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            mid_channels = mid_channels or out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
            x1 = f.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            return self.conv(torch.cat([x2, x1], dim=1))

    class LineExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.inc = DoubleConv(2, 64)
            self.down1, self.down2 = Down(64, 128), Down(128, 256)
            self.down3, self.down4 = Down(256, 512), Down(512, 512)
            self.up1, self.up2 = Up(1024, 256), Up(512, 128)
            self.up3, self.up4 = Up(256, 64), Up(128, 64)
            self.outc = OutConv(64, 1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))

    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x):
            return self.conv(x)

    return LineExtractor()
