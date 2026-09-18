"""Runtime boundary for validated Nemotron Omni + CropGuard LoRA inference.

This module isolates the model-specific sequence validated in ``claude3.py``.
It loads the existing local Nemotron Omni model and the existing remapped LoRA
adapter for inference only; it never trains, remaps, merges, writes, or
downloads model artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor


_SYSTEM_PROMPT = (
    "You are CropGuard, an AI assistant for crop disease detection and "
    "agricultural advisory."
)
_REASONING_END_MARKERS = ("</think>", "</thinking>", "final answer:")
_INNER_GENERATE_STRIP_KEYS = ("num_patches", "num_tokens", "imgs_sizes")


@dataclass(frozen=True)
class NemotronOmniConfig:
    """Configuration for the existing local Nemotron Omni and LoRA assets."""

    base_model_path: Path
    lora_adapter_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 1536
    expected_lora_modules: int = 116


@dataclass(frozen=True)
class NemotronGeneration:
    """Raw and farmer-facing text returned by one Nemotron generation."""

    raw_text: str
    final_text: str
    generated_token_count: int
    truncated: bool
    inference_time_seconds: float


class NemotronOmniVisionModel:
    """Import-safe adapter for validated Nemotron Omni + CropGuard LoRA inference.

    ``load`` owns one base model and one read-only LoRA overlay. ``diagnose``
    preserves the validated RGB image, chat-template, processor, and safe
    inner-generation behavior from ``claude3.py``.
    """

    def __init__(self, config: NemotronOmniConfig):
        self.config = config
        self._processor: Any | None = None
        self._base_model: Any | None = None
        self._model: Any | None = None
        self._patch_installed = False

    def load(self) -> None:
        """Load the local base model and remapped LoRA once for inference."""

        if self._model is not None:
            return

        base_model_path = Path(self.config.base_model_path)
        lora_adapter_path = Path(self.config.lora_adapter_path)

        if not base_model_path.is_dir():
            raise FileNotFoundError(
                f"Base model directory not found:\n{base_model_path}"
            )

        if not lora_adapter_path.is_dir():
            raise FileNotFoundError(
                f"LoRA directory not found:\n{lora_adapter_path}"
            )

        if self.config.dtype != "bfloat16":
            raise ValueError(
                "The validated Nemotron Omni path requires dtype='bfloat16'."
            )

        self._processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

        self._base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        self._model = PeftModel.from_pretrained(
            self._base_model,
            lora_adapter_path,
            is_trainable=False,
        )
        self._model.eval()

        lora_module_count = sum(
            1
            for module in self._model.modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        )
        if lora_module_count != self.config.expected_lora_modules:
            self.close()
            raise RuntimeError(
                "Unexpected LoRA module count: "
                f"expected {self.config.expected_lora_modules}, "
                f"found {lora_module_count}."
            )

        self._install_safe_inner_generation_patch()

    def diagnose(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> NemotronGeneration:
        """Diagnose one PIL image using the validated multimodal inference flow."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image instance.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        self.load()
        assert self._processor is not None
        assert self._model is not None

        generation_limit = (
            self.config.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        if generation_limit <= 0:
            raise ValueError("max_new_tokens must be positive.")

        image = image.convert("RGB")

        conversation = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        chat_text = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        image_token = getattr(self._processor, "image_token", None)
        if image_token is not None and image_token not in chat_text:
            raise RuntimeError("Chat template does not contain image token.")

        inputs = self._processor(
            text=chat_text,
            images=image,
            return_tensors="pt",
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.dtype.is_floating_point:
                    inputs[key] = value.to(
                        self.config.device,
                        dtype=torch.bfloat16,
                    )
                else:
                    inputs[key] = value.to(self.config.device)

        if "input_ids" not in inputs:
            raise RuntimeError("input_ids missing from processor output.")

        image_token_id = getattr(self._processor, "image_token_id", None)
        if image_token_id is not None:
            image_found = (inputs["input_ids"] == image_token_id).any().item()
            if not image_found:
                raise RuntimeError("No image token found in input_ids.")

        started_at = time.time()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=generation_limit,
                do_sample=False,
            )
        inference_time_seconds = time.time() - started_at

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[:, input_length:]
        raw_text = self._processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()
        generated_token_count = generated_tokens.shape[1]

        return NemotronGeneration(
            raw_text=raw_text,
            final_text=self._isolate_final_answer(raw_text),
            generated_token_count=generated_token_count,
            truncated=generated_token_count >= generation_limit,
            inference_time_seconds=inference_time_seconds,
        )

    def close(self) -> None:
        """Release in-memory references without changing model or LoRA files."""

        self._model = None
        self._base_model = None
        self._processor = None
        self._patch_installed = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _install_safe_inner_generation_patch(self) -> None:
        """Shield the inner HF generator from Omni-only metadata exactly once."""

        if self._patch_installed:
            return
        if self._model is None:
            raise RuntimeError("Model must be loaded before installing patch.")

        try:
            multimodal_model = self._model.get_base_model()
            inner_language_model = multimodal_model.language_model
            original_inner_generate = inner_language_model.generate

            def safe_inner_generate(*args: Any, **kwargs: Any) -> Any:
                for key in _INNER_GENERATE_STRIP_KEYS:
                    kwargs.pop(key, None)
                return original_inner_generate(*args, **kwargs)

            inner_language_model.generate = safe_inner_generate
            self._patch_installed = True
        except Exception as exc:
            self.close()
            raise RuntimeError(
                "Failed to install the safe Nemotron inner-generation patch."
            ) from exc

    @staticmethod
    def _isolate_final_answer(response: str) -> str:
        """Return the segment after the last known reasoning end marker."""

        response_lower = response.lower()
        last_cut = -1

        for marker in _REASONING_END_MARKERS:
            index = response_lower.rfind(marker)
            if index != -1:
                cut_point = index + len(marker)
                if cut_point > last_cut:
                    last_cut = cut_point

        if last_cut == -1:
            return response
        return response[last_cut:].strip()
