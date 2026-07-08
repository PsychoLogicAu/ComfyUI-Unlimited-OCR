"""
Unlimited OCR custom nodes for ComfyUI.

This module provides custom nodes for:
- Text extraction
- Document parsing
- General OCR
"""

import os
import json
import re
import tempfile
import logging
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .unlimited_ocr_worker import UnlimitedOCRModel, map_dtype_to_torch

logger = logging.getLogger(__name__)

# Model cache (LRU eviction to prevent unbounded growth)
_WORKER_CACHE_MAX: int = 2  # Max models to keep cached at once
_WORKER_CACHE: OrderedDict[str, "UnlimitedOCRModel"] = OrderedDict()


def _get_or_create_model(
    model_path: str,
    dtype_str: str,
    trust_remote_code: bool,
) -> "UnlimitedOCRModel":
    """Get or create an UnlimitedOCRModel from the cache.

    Uses LRU eviction: when the cache exceeds _WORKER_CACHE_MAX entries,
    the least-recently-used model is evicted and its GPU memory freed.
    """
    # Strip whitespace from model_path (common copy-paste error)
    model_path = model_path.strip()

    cache_key = f"{model_path}:{dtype_str}"

    # Mark as most-recently-used if already cached
    if cache_key in _WORKER_CACHE:
        _WORKER_CACHE.move_to_end(cache_key)
        return _WORKER_CACHE[cache_key]

    # Evict LRU entries if cache is full
    while len(_WORKER_CACHE) > _WORKER_CACHE_MAX:
        evicted_key, evicted_model = _WORKER_CACHE.popitem(last=False)
        try:
            if hasattr(evicted_model, "patcher"):
                del evicted_model.patcher
        except Exception:
            pass
        del evicted_model
        logger.info(f"Evicted LRU model from cache: {evicted_key} "
                    f"({_WORKER_CACHE_MAX+1} > {_WORKER_CACHE_MAX})")

    dtype = map_dtype_to_torch(dtype_str)
    model = UnlimitedOCRModel(
        model_path=model_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    _WORKER_CACHE[cache_key] = model
    logger.info(f"Loaded UnlimitedOCRModel: {model_path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Tensor / PIL helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image (RGB)."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    elif tensor.dim() != 3:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")

    tensor = tensor.clamp(0, 1).mul(255).byte()
    numpy_image = tensor.cpu().numpy().astype(np.uint8)
    return Image.fromarray(numpy_image, mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image back to ComfyUI tensor format [1, H, W, C]."""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _save_image_temporarily(image, suffix=".png"):
    """Save a PIL Image to a temporary file and return the path.

    Args:
        image: PIL Image to save.
        suffix: File extension for the temp file.

    Returns:
        Path to the temporary file.
    """
    temp_dir = tempfile.mkdtemp(prefix="comfy_ocr_")
    temp_path = os.path.join(temp_dir, f"image{suffix}")
    image.save(temp_path)
    return temp_path


# ──────────────────────────────────────────────────────────────────────────────
# Bounding box parsing and drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

# Deterministic color palette for different detection types
_DETECTION_TYPE_COLORS = {
    "title": (255, 0, 0),       # Red
    "header": (0, 255, 0),     # Green
    "footer": (0, 0, 255),     # Blue
    "text": (255, 255, 0),     # Cyan/Yellow
    "image": (255, 0, 255),    # Magenta
    "table": (0, 255, 255),    # Orange-ish
    "list": (128, 0, 0),       # Maroon
    "default": (128, 128, 128), # Gray
}


def _parse_det_annotations(text: str) -> list[dict]:
    """Parse bounding box annotations from model output.

    Extracts <tag>label [x1, y1, x2, y2]</tag> patterns.

    Args:
        text: The raw model output text.

    Returns:
        List of dicts with keys: label, x1, y1, x2, y2
    """
    pattern = r'<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>'
    results = []
    for match in re.finditer(pattern, text):
        label = match.group(1)
        box_str = match.group(2)
        try:
            coords = eval(box_str)
            if isinstance(coords, list) and len(coords) == 4:
                x1, y1, x2, y2 = coords
                results.append({
                    "label": label,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                })
        except Exception:
            continue
    return results


def _draw_boxes_on_image(
    image: Image.Image,
    boxes: list[dict],
) -> Image.Image:
    """Draw bounding boxes on image with different colors per label.

    Args:
        image: PIL Image to draw on.
        boxes: List of dicts with keys: label, x1, y1, x2, y2.

    Returns:
        PIL Image with drawn boxes.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box in boxes:
        label = box["label"]
        color = _DETECTION_TYPE_COLORS.get(label, _DETECTION_TYPE_COLORS["default"])
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background and text
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x1
        text_y = max(0, y1 - 15)
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill=(255, 255, 255, 30),
        )
        draw.text((text_x, text_y), label, font=font, fill=color)

    return image


def _clean_text(text: str) -> str:
    """Remove annotation tags from the model output.

    Strips <tag>label [x1, y1, x2, y2]</tag> patterns, leaving only the
    plain text content.

    Args:
        text: The raw model output text.

    Returns:
        Cleaned text without annotations.
    """
    # Remove all <tag>...</tag> patterns
    cleaned = re.sub(r'<\|det\|>\s*[A-Za-z_][\w-]*\s*\[[^\]]+\]\s*<\|/det\|>', '', text)
    # Remove any remaining <|...|> tags
    cleaned = re.sub(r'<\|[^|]+\|>', '', cleaned)
    return cleaned.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

class UnlimitedOCRLoader:
    """Load Unlimited-OCR model.

    Loads the model from HuggingFace with configurable device, dtype, and
    trust_remote_code settings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "baidu/Unlimited-OCR",
                    "multiline": False,
                    "placeholder": "Model path or HuggingFace repo ID",
                }),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "bfloat16"},
                ),
                "trust_remote_code": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("unlimited_ocr_model",)
    RETURN_NAMES = ("unlimited_ocr",)
    FUNCTION = "load_model"
    CATEGORY = "Unlimited OCR/Loader"

    def load_model(self, model_path, dtype, trust_remote_code):
        model = _get_or_create_model(model_path, dtype, trust_remote_code)
        return (model,)


class UnlimitedOCRConfig:
    """Configure inference parameters.

    Sets parameters for text extraction, document parsing, and general OCR.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_length": ("INT", {"default": 32768, "min": 1, "max": 65536, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "no_repeat_ngram_size": ("INT", {"default": 35, "min": 1, "max": 1024, "step": 1}),
                "ngram_window": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1}),
                "crop_mode": ("BOOLEAN", {"default": True}),
                "base_size": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "image_size": ("INT", {"default": 640, "min": 1, "max": 4096, "step": 1}),
                "tps_interval": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("unlimited_ocr_config",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "Unlimited OCR/Config"

    def configure(
        self,
        max_length,
        temperature,
        no_repeat_ngram_size,
        ngram_window,
        crop_mode,
        base_size=1024,
        image_size=640,
        tps_interval=0,
    ):
        config = {
            "max_length": max_length,
            "temperature": temperature,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "ngram_window": ngram_window,
            "crop_mode": crop_mode,
            "base_size": base_size,
            "image_size": image_size,
            "tps_interval": tps_interval,
        }
        return (config,)


class _InferenceNode:
    """Mixin that resolves the model + config and runs inference."""

    VALID_KEYS = {
        "max_length",
        "temperature",
        "no_repeat_ngram_size",
        "ngram_window",
        "crop_mode",
        "base_size",
        "image_size",
        "tps_interval",
    }

    @staticmethod
    def _resolve_model_and_config(unlimited_ocr, config):
        """Return (UnlimitedOCRModel, inference_kwargs)."""
        if config is not None and not isinstance(config, dict):
            raise TypeError(
                f"Config must be a dict, got {type(config).__name__}. "
                "Did you forget to wire an UnlimitedOCRConfig node?"
            )
        if config is None:
            config = {}

        unknown = set(config.keys()) - _InferenceNode.VALID_KEYS
        if unknown:
            print(
                f"[UnlimitedOCR] WARNING: unknown config keys ignored: "
                f"{', '.join(sorted(unknown))}"
            )

        inference_kw = {
            "max_length": config.get("max_length", 32768),
            "temperature": config.get("temperature", 0.0),
            "no_repeat_ngram_size": config.get("no_repeat_ngram_size", 35),
            "ngram_window": config.get("ngram_window", 128),
            "crop_mode": config.get("crop_mode", True),
            "base_size": config.get("base_size", 1024),
            "image_size": config.get("image_size", 640),
        }
        return unlimited_ocr, inference_kw


class UnlimitedOCRInference(_InferenceNode):
    """General OCR / text extraction with custom prompt.

    Accepts a custom prompt to control the type of OCR output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unlimited_ocr": ("unlimited_ocr_model",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "document parsing.",
                        "multiline": False,
                        "placeholder": "e.g., 'document parsing.', 'extract text.'",
                    },
                ),
            },
            "optional": {"config": ("unlimited_ocr_config",)},
        }

    RETURN_TYPES = ("text", "IMAGE", "text")
    RETURN_NAMES = ("extracted_text", "annotated_image", "debug_text")
    FUNCTION = "ocr"
    CATEGORY = "Unlimited OCR/Inference"

    def ocr(self, unlimited_ocr, image, prompt, config=None):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model, kw = self._resolve_model_and_config(unlimited_ocr, config)

        # Convert ComfyUI tensor to PIL and save to temp file
        pil = _tensor_to_pil(image[0])
        temp_path = _save_image_temporarily(pil)

        # Run inference
        result = model.infer(
            image=temp_path,
            prompt=prompt,
            **kw,
        )

        # Parse bounding boxes from the model output
        boxes = _parse_det_annotations(result)

        # Draw boxes on the image
        annotated_pil = _draw_boxes_on_image(pil, boxes)

        # Clean the text by removing annotations
        cleaned_text = _clean_text(result)

        return (cleaned_text, _pil_to_tensor(annotated_pil), result)


class UnlimitedOCRDebug(_InferenceNode):
    """Debug node to inspect model and inference state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unlimited_ocr": ("unlimited_ocr_model",),
                "image": ("IMAGE",),
                "test_prompt": (
                    "STRING",
                    {"default": "document parsing.", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("text", "IMAGE", "text")
    RETURN_NAMES = ("model_info", "debug_image", "debug_output")
    FUNCTION = "debug_model"
    CATEGORY = "Unlimited OCR/Debug"

    def debug_model(self, unlimited_ocr, image, test_prompt):
        model_info = {
            "model_path": unlimited_ocr.model_path,
            "load_device": str(unlimited_ocr.load_device),
            "offload_device": str(unlimited_ocr.offload_device),
            "dtype": str(unlimited_ocr.dtype),
        }

        model, kw = self._resolve_model_and_config(
            unlimited_ocr, {"max_length": 256, "temperature": 0.0}
        )
        pil = _tensor_to_pil(image[0])
        debug_text = ""

        try:
            temp_path = _save_image_temporarily(pil)
            result = model.infer(
                image=temp_path,
                prompt=test_prompt,
                **kw,
            )
            debug_text = f"Test Result:\n{result[:200]}..."
        except Exception as e:
            debug_text = f"Test Error: {e}"

        return (json.dumps(model_info, indent=2), _pil_to_tensor(pil), debug_text)


# ──────────────────────────────────────────────────────────────────────────────
# Node Registration
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "UnlimitedOCRLoader": UnlimitedOCRLoader,
    "UnlimitedOCRConfig": UnlimitedOCRConfig,
    "UnlimitedOCRInference": UnlimitedOCRInference,
    "UnlimitedOCRDebug": UnlimitedOCRDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnlimitedOCRLoader": "Load Unlimited OCR Model",
    "UnlimitedOCRConfig": "Configure Inference",
    "UnlimitedOCRInference": "OCR / Text Extraction",
    "UnlimitedOCRDebug": "Debug Model",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]