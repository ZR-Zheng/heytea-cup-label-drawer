"""Informative Drawings ONNX inference adapter.

The model is from Caroline Chan et al., CVPR 2022 (MIT License).
ONNX conversion: https://huggingface.co/rocca/informative-drawings-line-art-onnx
"""

from pathlib import Path

import cv2
import numpy as np


_SESSION_CACHE = {}


def informative_drawings_to_gray(rgb: np.ndarray, model_path: str, input_size: int = 512, device_name: str = "auto") -> np.ndarray:
    if not model_path:
        raise RuntimeError("请先导入 Informative Drawings ONNX 模型（model.onnx）。")
    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Informative Drawings 模型文件不存在：{path}")
    try:
        import onnxruntime as ort
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"ONNX Runtime 无法加载：{exc}\n请安装 onnxruntime，并确认 Windows Visual C++ 运行库完整。"
        ) from exc

    session = _get_session(ort, path, device_name)
    h, w = rgb.shape[:2]
    size = max(128, int(input_size))
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC)
    tensor = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    result = np.clip(output[0, 0] * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC) if result.shape != (h, w) else result


def _get_session(ort, path: Path, device_name: str):
    providers = ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    if (device_name or "auto").lower() in {"auto", "cuda"} and "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    key = (str(path.resolve()), tuple(providers))
    if key not in _SESSION_CACHE:
        _SESSION_CACHE[key] = ort.InferenceSession(str(path), providers=providers)
    return _SESSION_CACHE[key]
