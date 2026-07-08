"""
Unlimited OCR Worker module.

This module provides:
- UnlimitedOCRModel wrapper class (ComfyUI-compatible memory management)
- Worker class for inference
- Text extraction, document parsing, and general OCR
"""

import os
import re
import tempfile
import torch
from PIL import Image, ImageOps
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
        save_results=False,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=128,
        temperature=0.0,
        output_path="",
        eval_mode=True,
    ):
        """Run inference with ComfyUI-compatible memory management.

        Calls load_model_gpu() before inference so ComfyUI can manage
        GPU memory (evict other models if needed, track usage, etc.).
        """
        # --- Load model to GPU via ComfyUI memory management ---
        comfy.model_management.load_model_gpu(self.patcher)

        model = self._get_model()

        # Build conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Format the prompt using the model's conversation template
        from .conversation import get_conv_template
        prompt_formatted = format_messages(
            conversations=conversation,
            sft_format="plain",
            system_prompt="",
        )

        # Load image
        pil_image = load_image(image)

        # Default to a temp directory if no output_path is provided
        if not output_path:
            output_path = tempfile.mkdtemp(prefix="ocr_output_")

        # Run inference
        result = model.infer(
            self.tokenizer,
            prompt=prompt_formatted,
            image_file=image,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=save_results,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ngram_window=ngram_window,
            temperature=temperature,
            eval_mode=eval_mode,
        )

        return result


def load_image(image_path):
    """Load an image from a file path, handling EXIF orientation.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image with corrected orientation.
    """
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def format_messages(conversations, sft_format="deepseek", system_prompt=""):
    """Format conversations using the model's conversation template.

    Args:
        conversations: List of messages.
        sft_format: The format of the SFT template to use.
        system_prompt: The system prompt to use.

    Returns:
        The formatted text.
    """
    from .conversation import get_conv_template
    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    formatted = conv.get_prompt().strip()
    return formatted