from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BM25SearchResult:
    chunk_id: str
    knowledge_id: str
    score: float
    content: str
    metadata: dict


class BM25Index:
    """
    Dependency-free BM25 lexical index for CropGuard.

    Uses the same retrieval text representation as the vector
    retrieval pipeline.

    BM25 is used for:
      - exact agricultural terminology
      - disease names
      - crop names
      - identifiers
      - lexical precision
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:

        self.k1 = k1
        self.b = b

        self.records: list[dict] = []
        self.documents: list[list[str]] = []
        self.term_frequencies: list[Counter] = []

        self.document_frequency: Counter = Counter()
        self.avg_document_length: float = 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Lightweight agricultural-text tokenizer.

        Keeps alphanumeric terms and underscore-separated
        identifiers such as:

            northern_leaf_blight
            sample-nlb-002
        """

        text = text.lower()

        return re.findall(
            r"[a-z0-9]+(?:[_-][a-z0-9]+)*",
            text,
        )

    def add(
        self,
        texts: list[str],
        records: list[dict],
    ) -> None:

        if len(texts) != len(records):
            raise ValueError(
                f"Text count {len(texts)} != "
                f"record count {len(records)}"
            )

        for text, record in zip(texts, records):

            tokens = self.tokenize(text)

            self.documents.append(tokens)

            tf = Counter(tokens)
            self.term_frequencies.append(tf)

            for term in tf:
                self.document_frequency[term] += 1

            self.records.append(record)

        if self.documents:
            self.avg_document_length = (
                sum(
                    len(doc)
                    for doc in self.documents
                )
                / len(self.documents)
            )

    def _idf(
        self,
        term: str,
    ) -> float:

        n = len(self.documents)

        df = self.document_frequency.get(
            term,
            0,
        )

        if df == 0:
            return 0.0

        # Standard BM25 IDF with a +1 stabilization.
        return math.log(
            1.0
            + (n - df + 0.5)
            / (df + 0.5)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25SearchResult]:

        if not query.strip():
            return []

        if not self.documents:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = []

        for doc_idx, tokens in enumerate(
            self.documents
        ):

            doc_length = len(tokens)

            tf = self.term_frequencies[
                doc_idx
            ]

            score = 0.0

            for term in query_tokens:

                frequency = tf.get(
                    term,
                    0,
                )

                if frequency == 0:
                    continue

                idf = self._idf(term)

                numerator = (
                    frequency
                    * (self.k1 + 1.0)
                )

                denominator = (
                    frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * (
                            doc_length
                            / max(
                                self.avg_document_length,
                                1e-12,
                            )
                        )
                    )
                )

                score += (
                    idf
                    * numerator
                    / denominator
                )

            # Only retain documents with an actual lexical match.
            # Zero-score documents must not enter the BM25 candidate set.
            if score > 0.0:
                scores.append(
                    (doc_idx, score)
                )

        if not scores:
            return []

        scores.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        ranked_results = scores[:top_k]

        results = []

        for idx, score in ranked_results:

            record = self.records[idx]

            # Expose canonical retrieval fields together with
            # source-specific metadata. This keeps BM25 and
            # vector retrieval metadata consistent.
            metadata = dict(
                record.get(
                    "metadata",
                    {},
                )
            )

            for key in (
                "chunk_id",
                "knowledge_id",
                "source",
                "source_type",
                "crop",
                "disease",
                "evidence_type",
                "recommendation_preference",
                "organic_eligible",
                "ipm_eligible",
                "image_path",
                "original_label",
                "health_status",
                "title",
                "chunk_index",
                "total_chunks",
            ):
                if key in record:
                    metadata[key] = record[key]

            results.append(
                BM25SearchResult(
                    chunk_id=record["chunk_id"],
                    knowledge_id=record["knowledge_id"],
                    score=float(score),
                    content=record["content"],
                    metadata=metadata,
                )
            )

        return results

    def save(
        self,
        directory: str | Path,
    ) -> None:

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            directory / "bm25_data.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "avg_document_length":
                        self.avg_document_length,
                    "document_frequency":
                        dict(self.document_frequency),
                    "records": self.records,
                    "documents": self.documents,
                    "term_frequencies": [
                        dict(tf)
                        for tf in self.term_frequencies
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "BM25Index":

        directory = Path(directory)

        with open(
            directory / "bm25_data.json",
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        index = cls(
            k1=float(data["k1"]),
            b=float(data["b"]),
        )

        index.avg_document_length = float(
            data["avg_document_length"]
        )

        index.document_frequency = Counter(
            data["document_frequency"]
        )

        index.records = data["records"]

        index.documents = data["documents"]

        index.term_frequencies = [
            Counter(tf)
            for tf in data["term_frequencies"]
        ]

        return index
