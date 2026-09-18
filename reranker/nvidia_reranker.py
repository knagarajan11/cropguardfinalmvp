from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from nemo_automodel import NeMoAutoModelCrossEncoder
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer


@dataclass
class RerankerResult:
    """
    A reranked CropGuard retrieval candidate.

    The original retrieval metadata is retained so that the complete
    retrieval path remains auditable.
    """

    chunk_id: str
    knowledge_id: str
    content: str
    metadata: dict[str, Any]

    reranker_score: float

    vector_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float | None = None


class NVIDIAReranker:
    """
    NVIDIA Nemotron Reranker adapter for CropGuard.

    Model:
        nvidia/llama-nemotron-rerank-1b-v2

    This is inference-only.

    Pipeline:
        hybrid candidates
            -> query/passage pairs
            -> NVIDIA NeMo AutoModel CrossEncoder
            -> relevance logits
            -> descending score
            -> top_k
    """

    DEFAULT_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_PROMPT = "question:{query} \n \n passage:{passage}"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = DEFAULT_MAX_LENGTH,
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the NVIDIA reranker but is not available."
            )

        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length
        self.prompt_template = prompt_template

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.model = (
            NeMoAutoModelCrossEncoder.from_pretrained(
                model_name_or_path,
                pooling="avg",
                num_labels=1,
                use_liger_kernel=False,
                use_sdpa_patching=False,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        # Reranker is frozen inference-only.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _format_text(self, query: str, passage: str) -> str:
        return self.prompt_template.format(
            query=query,
            passage=passage,
        )

    def _score_pairs(
        self,
        query: str,
        passages: Sequence[str],
    ) -> list[float]:
        if not passages:
            return []

        texts = [
            self._format_text(query, passage)
            for passage in passages
        ]

        # Follow NVIDIA's own implementation:
        # tokenize without padding, then pad separately.
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=False,
            truncation=True,
        )

        token_features = [
            {
                key: encodings[key][i]
                for key in encodings
            }
            for i in range(len(texts))
        ]

        batch = self.tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )

        batch = {
            key: value.to(self.device)
            for key, value in batch.items()
        }

        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda",
                dtype=self.dtype,
            ):
                outputs = self.model(
                    **batch,
                    return_dict=True,
                )

        scores = outputs.logits.view(-1).float().cpu().tolist()

        if len(scores) != len(passages):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores "
                f"for {len(passages)} passages."
            )

        return [float(score) for score in scores]

    @staticmethod
    def _candidate_value(candidate: Any, name: str, default: Any = None) -> Any:
        """
        Read a field from either a retrieval dataclass/object or dict.
        """
        if isinstance(candidate, dict):
            return candidate.get(name, default)

        return getattr(candidate, name, default)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int = 8,
    ) -> list[RerankerResult]:
        """
        Rerank existing hybrid retrieval candidates.

        Candidates can be Vector/BM25/Hybrid retrieval objects or
        dictionaries containing equivalent fields.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if not candidates:
            return []

        passages: list[str] = []

        for candidate in candidates:
            content = self._candidate_value(
                candidate,
                "content",
                "",
            )

            if not content:
                metadata = self._candidate_value(
                    candidate,
                    "metadata",
                    {},
                ) or {}

                content = metadata.get("content", "")

            if not content:
                raise ValueError(
                    "Every reranker candidate must contain "
                    "retrievable text in 'content' or metadata['content']."
                )

            passages.append(str(content))

        scores = self._score_pairs(
            query=query,
            passages=passages,
        )

        reranked: list[RerankerResult] = []

        for candidate, score in zip(candidates, scores):
            metadata = dict(
                self._candidate_value(
                    candidate,
                    "metadata",
                    {},
                ) or {}
            )

            chunk_id = str(
                self._candidate_value(
                    candidate,
                    "chunk_id",
                    metadata.get("chunk_id", ""),
                )
            )

            knowledge_id = str(
                self._candidate_value(
                    candidate,
                    "knowledge_id",
                    metadata.get("knowledge_id", ""),
                )
            )

            content = str(
                self._candidate_value(
                    candidate,
                    "content",
                    metadata.get("content", ""),
                )
            )

            reranked.append(
                RerankerResult(
                    chunk_id=chunk_id,
                    knowledge_id=knowledge_id,
                    content=content,
                    metadata=metadata,
                    reranker_score=float(score),
                    vector_score=self._candidate_value(
                        candidate,
                        "vector_score",
                    ),
                    bm25_score=self._candidate_value(
                        candidate,
                        "bm25_score",
                    ),
                    hybrid_score=self._candidate_value(
                        candidate,
                        "final_score",
                        self._candidate_value(
                            candidate,
                            "hybrid_score",
                        ),
                    ),
                )
            )

        reranked.sort(
            key=lambda result: result.reranker_score,
            reverse=True,
        )

        return reranked[:top_k]

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_name_or_path,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "max_length": self.max_length,
            "prompt_template": self.prompt_template,
            "inference_only": True,
        }
