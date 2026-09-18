from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL = (
    "/home/gsh-ndhmy/.cache/huggingface/models--nvidia--"
    "llama-nemotron-embed-vl-1b-v2/snapshots/"
    "582e3bf72aee355e3c59ed89de53543c5b0657ee"
)


@dataclass(frozen=True)
class VLEmbeddingConfig:
    model_path: str = DEFAULT_MODEL
    device: str = "cuda"
    dtype: str = "bfloat16"
    normalize: bool = True
    pooling: str = "avg"


class NVIDIAVLEmbedder:
    """
    CropGuard adapter for NVIDIA Nemotron VL Embed 1B v2.

    Uses the model's native:
      - encode_queries()
      - encode_documents()

    No model modification or fine-tuning.
    """

    def __init__(
        self,
        config: VLEmbeddingConfig | None = None,
    ) -> None:
        self.config = config or VLEmbeddingConfig()

        self.model_path = Path(self.config.model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"VL Embed model not found: {self.model_path}"
            )

        self.processor = None
        self.model = None
        self._loaded = False

    def load(self) -> "NVIDIAVLEmbedder":
        if self._loaded:
            return self

        dtype = (
            torch.bfloat16
            if self.config.dtype == "bfloat16"
            else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            dtype=dtype,
        )

        self.model = self.model.to(self.config.device).eval()

        self._loaded = True
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @staticmethod
    def _to_numpy(embeddings: Any) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            return embeddings.float().detach().cpu().numpy()

        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms

    def encode_queries(
        self,
        queries: Sequence[str],
    ) -> np.ndarray:
        self._ensure_loaded()

        queries = [str(q) for q in queries]

        with torch.inference_mode():
            embeddings = self.model.encode_queries(
                queries,
                pool_type=self.config.pooling,
            )

        result = self._to_numpy(embeddings)

        if result.ndim == 1:
            result = result.reshape(1, -1)

        if self.config.normalize:
            result = self._normalize(result)

        return result.astype(np.float32, copy=False)

    def encode_documents(
        self,
        texts: Sequence[str] | None = None,
        images: Sequence[Any] | None = None,
    ) -> np.ndarray:
        self._ensure_loaded()

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided.")

        if texts is not None:
            texts = [str(text) for text in texts]

        with torch.inference_mode():
            embeddings = self.model.encode_documents(
                images=images,
                texts=texts,
                pool_type=self.config.pooling,
            )

        result = self._to_numpy(embeddings)

        if result.ndim == 1:
            result = result.reshape(1, -1)

        if self.config.normalize:
            result = self._normalize(result)

        return result.astype(np.float32, copy=False)

    def encode_passages(
        self,
        passages: Sequence[str],
    ) -> np.ndarray:
        return self.encode_documents(texts=passages)

    def encode(
        self,
        texts: Sequence[str],
        *,
        query: bool = False,
    ) -> np.ndarray:
        if query:
            return self.encode_queries(texts)

        return self.encode_documents(texts=texts)

    @property
    def dimension(self) -> int:
        return 2048

    @property
    def info(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path),
            "dimension": self.dimension,
            "pooling": self.config.pooling,
            "normalize": self.config.normalize,
            "query_interface": "encode_queries",
            "document_interface": "encode_documents",
            "model_type": "llama_nemotron_vl",
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
