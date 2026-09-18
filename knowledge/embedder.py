from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class EmbeddingConfig:
    model_path: str = "nvidia/llama-nemotron-embed-1b-v2"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    max_length: int = 8192
    normalize: bool = True


class NVIDIAEmbedder:
    """
    CropGuard dense embedding boundary.

    Uses NVIDIA llama-nemotron-embed-1b-v2.
    No training, fine-tuning, merging, or model modification.

    Query and passage encoding are separate application-level
    interfaces. The underlying NVIDIA model does not expose an
    input_type='query'/'passage' argument in its native forward API.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA GPU is available")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            dtype=self.config.dtype,
        )

        self.model = self.model.to(self.config.device)
        self.model.eval()

        self.dimension = int(
            getattr(self.model.config, "hidden_size", 2048)
        )

        self.pooling = getattr(
            self.model.config,
            "pooling",
            "avg",
        )

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1)

        return summed / counts

    @torch.inference_mode()
    def _encode(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> torch.Tensor:

        if not texts:
            return torch.empty(
                (0, self.dimension),
                dtype=torch.float32,
            )

        results = []

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.config.device)
                for key, value in encoded.items()
            }

            outputs = self.model(**encoded)

            # Pool in the model output dtype, then perform the final
            # normalization explicitly in FP32 for stable retrieval
            # cosine similarity.
            embeddings = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            ).float()

            if self.config.normalize:
                embeddings = F.normalize(
                    embeddings,
                    p=2,
                    dim=1,
                    eps=1e-12,
                )

                # Final FP32 normalization guard.
                embeddings = embeddings / embeddings.norm(
                    p=2,
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-12)

            results.append(
                embeddings.cpu()
            )

        return torch.cat(results, dim=0)

    def encode_query(
        self,
        query: str,
    ) -> torch.Tensor:
        """
        Encode one farmer/search query.

        Returns:
            Tensor of shape [2048].
        """
        return self._encode([query])[0]

    def encode_passages(
        self,
        passages: Sequence[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Encode knowledge passages.

        Returns:
            Tensor of shape [N, 2048].
        """
        return self._encode(
            passages,
            batch_size=batch_size,
        )

    def encode(
        self,
        texts: Sequence[str] | str,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Generic encoding convenience method.
        """
        if isinstance(texts, str):
            return self.encode_query(texts)

        return self.encode_passages(
            texts,
            batch_size=batch_size,
        )

    def info(self) -> dict:
        return {
            "model": self.config.model_path,
            "dimension": self.dimension,
            "pooling": self.pooling,
            "device": self.config.device,
            "dtype": str(self.config.dtype),
            "normalized": self.config.normalize,
            "max_length": self.config.max_length,
        }
