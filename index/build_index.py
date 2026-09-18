"""Build CropGuard diagnosis and treatment retrieval indexes.

Final MVP rules:
- No model retraining.
- Reuse the validated NVIDIA Embed 1B v2.
- Diagnosis and treatment evidence are indexed separately.
- Knowledge changes require re-indexing only.
- Full-corpus vectors are written through a disk-backed memmap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.chunker import KnowledgeChunker
from knowledge.loaders import (
    load_normalized_dataset,
    load_treatment_knowledge,
)
from knowledge.retrieval_text import build_chunk_retrieval_text
from knowledge.embedder import NVIDIAEmbedder
from index.vector_index import VectorIndex
from index.bm25_index import BM25Index


ROOT = Path("/workspace")

DEFAULT_PLANTVILLAGE = ROOT / "data/processed/plantvillage_normalized.jsonl"
DEFAULT_PLANTDOC = ROOT / "data/processed/plantdoc_normalized.jsonl"
DEFAULT_RICE = ROOT / "data/processed/rice1426_normalized.jsonl"
DEFAULT_TREATMENT = ROOT / "knowledge/sample_data/treatment_knowledge.jsonl"


EXPECTED_COUNTS = {
    "PlantVillage": 108608,
    "PlantDoc": 5156,
    "Rice": 3829,
    "treatment": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        choices=["diagnosis", "treatment", "all"],
        default="all",
    )
    parser.add_argument("--plantvillage", type=Path, default=DEFAULT_PLANTVILLAGE)
    parser.add_argument("--plantdoc", type=Path, default=DEFAULT_PLANTDOC)
    parser.add_argument("--rice", type=Path, default=DEFAULT_RICE)
    parser.add_argument("--treatment", type=Path, default=DEFAULT_TREATMENT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "index",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=None)

    return parser.parse_args()


def load_domain_records(domain: str, args: argparse.Namespace):
    if domain == "diagnosis":
        sources = [
            ("PlantVillage", args.plantvillage),
            ("PlantDoc", args.plantdoc),
            ("Rice", args.rice),
        ]

        records = []

        for source_name, path in sources:
            print(f"Loading: {path}")
            source_records = load_normalized_dataset(path)

            expected = EXPECTED_COUNTS[source_name]

            print(
                f"  {source_name}: {len(source_records):,} "
                f"records"
            )

            if args.max_records is None and len(source_records) != expected:
                raise RuntimeError(
                    f"{source_name} count mismatch: "
                    f"expected {expected:,}, "
                    f"found {len(source_records):,}"
                )

            records.extend(source_records)

        print(
            f"Total diagnosis records: {len(records):,}"
        )

        return records

    if domain == "treatment":
        print(f"Loading: {args.treatment}")

        records = load_treatment_knowledge(
            args.treatment
        )

        if (
            args.max_records is None
            and len(records) != EXPECTED_COUNTS["treatment"]
        ):
            raise RuntimeError(
                "Treatment count mismatch: "
                f"expected {EXPECTED_COUNTS['treatment']}, "
                f"found {len(records)}"
            )

        print(
            f"Treatment records: {len(records):,}"
        )

        return records

    raise ValueError(f"Unsupported domain: {domain}")


def prepare_chunks(records):
    chunker = KnowledgeChunker()
    chunks = []

    for index, record in enumerate(records, start=1):
        chunks.extend(
            chunker.chunk_record(record)
        )

        if index % 10000 == 0:
            print(
                f"  Prepared records: {index:,} "
                f"| chunks: {len(chunks):,}"
            )

    return chunks


def build_records_for_index(chunks):
    records = []

    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "knowledge_id": chunk.knowledge_id,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "crop": chunk.crop,
                "disease": chunk.disease,
                "evidence_type": chunk.evidence_type,
                "content": chunk.content,
                "recommendation_preference":
                    chunk.recommendation_preference,
                "organic_eligible":
                    bool(chunk.organic_eligible),
                "ipm_eligible":
                    bool(chunk.ipm_eligible),
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "metadata": dict(chunk.metadata or {}),
            }
        )

    return records


def build_domain(
    domain: str,
    records,
    output_root: Path,
    batch_size: int,
):
    print()
    print("=" * 70)
    print(f"BUILDING {domain.upper()} INDEX")
    print("=" * 70)

    started = time.time()

    chunks = prepare_chunks(records)

    print(f"Records: {len(records):,}")
    print(f"Chunks : {len(chunks):,}")

    if not chunks:
        raise RuntimeError(
            f"No chunks available for {domain}."
        )

    texts = [
        build_chunk_retrieval_text(chunk)
        for chunk in chunks
    ]

    records_for_index = build_records_for_index(
        chunks
    )

    if len(texts) != len(records_for_index):
        raise RuntimeError(
            "Text/metadata count mismatch."
        )

    print("Loading NVIDIA Embed 1B v2...")
    embedder = NVIDIAEmbedder()

    print(
        f"Embedding dimension: {embedder.dimension}"
    )
    print(
        f"Embedding batch size: {batch_size}"
    )

    vector_dir = output_root / f"{domain}_vector"
    bm25_dir = output_root / f"{domain}_bm25"

    temp_root = (
        output_root /
        f".{domain}_building"
    )

    shutil.rmtree(
        temp_root,
        ignore_errors=True,
    )

    temp_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    memmap_path = (
        temp_root / "vectors.memmap"
    )

    vectors = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(
            len(texts),
            embedder.dimension,
        ),
    )

    try:
        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(texts),
            )

            batch_texts = texts[start:end]

            batch_vectors = (
                embedder
                .encode_passages(
                    batch_texts,
                    batch_size=batch_size,
                )
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            vectors[start:end] = batch_vectors

            if (
                end % 1000 == 0
                or end == len(texts)
            ):
                print(
                    f"  Embedded: "
                    f"{end:,}/{len(texts):,}"
                )

        vectors.flush()

        print("Building vector index...")

        vector_index = VectorIndex(
            dimension=embedder.dimension
        )

        # VectorIndex currently expects an ndarray.
        # The memmap is already ndarray-compatible.
        vector_index.add(
            vectors,
            records_for_index,
        )

        vector_index.save(
            temp_root / "vector"
        )

        print("Building BM25 index...")

        bm25_index = BM25Index()

        bm25_index.add(
            texts,
            records_for_index,
        )

        bm25_index.save(
            temp_root / "bm25"
        )

        metadata = {
            "domain": domain,
            "records": len(records),
            "chunks": len(chunks),
            "vectors": int(len(texts)),
            "dimension": int(embedder.dimension),
            "embedding_model":
                "nvidia/llama-nemotron-embed-1b-v2",
            "embedding_normalized": True,
            "embedding_batch_size": batch_size,
            "sources": sorted(
                {
                    str(record.source)
                    for record in records
                }
            ),
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "build_seconds": round(
                time.time() - started,
                2,
            ),
        }

        with (
            temp_root / "build_metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        # Replace only after the complete build succeeds.
        shutil.rmtree(
            vector_dir,
            ignore_errors=True,
        )
        shutil.rmtree(
            bm25_dir,
            ignore_errors=True,
        )

        (temp_root / "vector").rename(
            vector_dir
        )
        (temp_root / "bm25").rename(
            bm25_dir
        )

        print()
        print(f"Vector index : {vector_dir}")
        print(f"BM25 index   : {bm25_dir}")
        print(
            f"Vectors      : "
            f"{len(texts):,} x "
            f"{embedder.dimension}"
        )
        print(
            f"Elapsed      : "
            f"{time.time() - started:.2f}s"
        )

        return metadata

    finally:
        del vectors

        if memmap_path.exists():
            memmap_path.unlink()

        # Keep completed vector/bm25 directories,
        # but remove temporary build artifacts.
        if temp_root.exists():
            shutil.rmtree(
                temp_root,
                ignore_errors=True,
            )


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero"
        )

    if (
        args.max_records is not None
        and args.max_records <= 0
    ):
        raise ValueError(
            "--max-records must be greater than zero"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.domain == "all":
        domains = [
            "diagnosis",
            "treatment",
        ]
    else:
        domains = [args.domain]

    summary = []

    for domain in domains:
        records = load_domain_records(
            domain,
            args,
        )

        if args.max_records is not None:
            records = records[:args.max_records]

        metadata = build_domain(
            domain,
            records,
            args.output_root,
            args.batch_size,
        )

        summary.append(metadata)

    summary_path = (
        args.output_root /
        "build_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
