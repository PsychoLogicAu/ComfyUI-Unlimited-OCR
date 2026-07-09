"""
Unlimited OCR Worker module.

This module provides:
- UnlimitedOCRModel wrapper class (ComfyUI-compatible memory management)
- Worker class for inference
- Text extraction, document parsing, and general OCR
"""

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

import comfy.model_management
import comfy.model_patcher


def map_dtype_to_torch(dtype_str):
    """Map dtype config string to torch.dtype.

    Args:
        dtype_str: String like "auto", "fp32", "fp16", "bf16", "int8", or None

    Returns:
        torch.dtype or None (to let ComfyUI decide)
    """
    if dtype_str is None or dtype_str == "auto":
        return None  # Let ComfyUI decide

    dtype_lower = dtype_str.lower()

    if dtype_lower in ("fp32", "float32", "float"):
        return torch.float32
    elif dtype_lower in ("fp16", "float16", "half"):
        return torch.float16
    elif dtype_lower in ("bf16", "bfloat16", "bfloat"):
        return torch.bfloat16
    elif dtype_lower == "int8":
        return torch.int8

    return None


class _HFModelProxy:
    """Thin wrapper around a HuggingFace model that provides a settable 'device' property.

    CoreModelPatcher.load() expects to set model.device = device_to move the
    model between GPU and CPU. HF models expose 'device' as a read-only property
    derived from their parameters, so we intercept the setter and call .to(device).
    All other attribute access is forwarded transparently.
    """

    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    @property
    def device(self):
        return self._model.device

    @device.setter
    def device(self, value):
        self._model.to(value)

    @property
    def training(self):
        return self._model.training

    def __getattr__(self, name):
        return getattr(self._model, name)


class UnlimitedOCRModel:
    """ComfyUI-compatible wrapper for Unlimited-OCR model.

    Follows the same pattern as LocateAnythingModel:
    - Uses CoreModelPatcher for proper GPU memory tracking and eviction.
    - Uses model_management for device/dtype selection.
    - Model is loaded on offload_device and moved to GPU only when needed.
    """

    def __init__(self, model_path, dtype=None, trust_remote_code=True):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code

        # --- Device / dtype selection via ComfyUI model_management ---
        self.load_device = comfy.model_management.text_encoder_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)

        # If the user explicitly requested a different dtype, honour it.
        if dtype is not None:
            self.dtype = dtype

        # --- Load tokenizer (lightweight, keep on CPU) ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        # --- Load model on the offload_device (CPU) ---
        hf_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.offload_device).eval()

        # Wrap in proxy so CoreModelPatcher can set model.device = device_to.
        self.model = _HFModelProxy(hf_model)

        # --- Wrap in CoreModelPatcher for ComfyUI memory management ---
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )

    def _get_model(self):
        """Return the raw model (used internally after loading to GPU)."""
        return self.model

    @torch.no_grad()
    def infer(
        self,
        image,
        prompt,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=128,
        temperature=0.0,
    ):
        """Run inference with ComfyUI-compatible memory management.

        Calls load_model_gpu() before inference so ComfyUI can manage
        GPU memory (evict other models if needed, track usage, etc.).

        The raw prompt is passed directly to the model's infer method,
        which handles conversation formatting internally. Do NOT pre-format
        the prompt here, otherwise it will be formatted twice.
        """
        # --- Load model to GPU via ComfyUI memory management ---
        comfy.model_management.load_model_gpu(self.patcher)

        model = self._get_model()

        # Run inference - pass the raw prompt; the model's infer method
        # builds its own conversation and formats it via format_messages().
        # Pre-formatting here would cause double formatting.
        result = model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ngram_window=ngram_window,
            temperature=temperature,
            eval_mode=True,
        )

        return result