"""Maintenance-friendly CropGuard index refresh.

Detects changes in knowledge source files and rebuilds only the
affected retrieval domain.

Final MVP rules:
- Source/data changes require re-indexing, not model retraining.
- Diagnosis sources:
    PlantVillage
    PlantDoc
    Rice
- Treatment source:
    SampleMVP treatment knowledge
- Existing indexes are replaced only after a successful build.
- NVIDIA Embed 1B v2 remains frozen and unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/workspace")
INDEX_ROOT = ROOT / "index"

MANIFEST_PATH = INDEX_ROOT / "source_manifest.json"

SOURCES = {
    "diagnosis": {
        "plantvillage": (
            ROOT / "data/processed/plantvillage_normalized.jsonl"
        ),
        "plantdoc": (
            ROOT / "data/processed/plantdoc_normalized.jsonl"
        ),
        "rice": (
            ROOT / "data/processed/rice1426_normalized.jsonl"
        ),
    },
    "treatment": {
        "treatment": (
            ROOT / "knowledge/sample_data/treatment_knowledge.jsonl"
        ),
    },
}


def sha256_file(path: Path) -> str:
    """Calculate SHA256 for a source file."""

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def source_snapshot() -> dict:
    """Create a source fingerprint snapshot."""

    snapshot = {}

    for domain, sources in SOURCES.items():
        snapshot[domain] = {}

        for name, path in sources.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Knowledge source does not exist: {path}"
                )

            stat = path.stat()

            snapshot[domain][name] = {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }

    return snapshot


def load_manifest() -> dict | None:
    """Load previous source manifest."""

    if not MANIFEST_PATH.exists():
        return None

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def changed_domains(
    previous: dict | None,
    current: dict,
) -> list[str]:
    """Return domains whose source fingerprints changed."""

    if previous is None:
        return ["diagnosis", "treatment"]

    previous_sources = previous.get(
        "sources",
        {},
    )

    changed = []

    for domain in SOURCES:
        if (
            previous_sources.get(domain)
            != current.get(domain)
        ):
            changed.append(domain)

    return changed


def run_build(
    domain: str,
    batch_size: int,
) -> None:
    """Run the validated index builder."""

    command = [
        sys.executable,
        str(ROOT / "index/build_index.py"),
        "--domain",
        domain,
        "--batch-size",
        str(batch_size),
    ]

    print()
    print("=" * 70)
    print(f"REFRESHING {domain.upper()} INDEX")
    print("=" * 70)
    print("Command:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Index build failed for domain '{domain}' "
            f"with exit code {result.returncode}."
        )


def save_manifest(
    snapshot: dict,
    rebuilt_domains: list[str],
) -> None:
    """Persist successful refresh state."""

    payload = {
        "manifest_version": 1,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_model":
            "nvidia/llama-nemotron-embed-1b-v2",
        "model_retraining": False,
        "rebuilt_domains": rebuilt_domains,
        "sources": snapshot,
    }

    temporary = MANIFEST_PATH.with_suffix(
        ".json.tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    temporary.replace(MANIFEST_PATH)


def print_status(
    snapshot: dict,
    previous: dict | None,
    changed: list[str],
) -> None:
    print()
    print("=" * 70)
    print("CROPGUARD INDEX STATUS")
    print("=" * 70)

    for domain, sources in snapshot.items():
        print()
        print(f"{domain.upper()} SOURCES:")

        for name, info in sources.items():
            previous_info = (
                previous.get("sources", {})
                .get(domain, {})
                .get(name, {})
                if previous
                else {}
            )

            if not previous:
                status = "NEW"
            elif (
                previous_info.get("sha256")
                != info["sha256"]
            ):
                status = "CHANGED"
            else:
                status = "UNCHANGED"

            print(
                f"  {name:15} {status:10} "
                f"{info['size']:,} bytes"
            )

    print()
    print(
        "Domains requiring rebuild:",
        ", ".join(changed)
        if changed
        else "NONE",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refresh CropGuard indexes only when "
            "knowledge sources change."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="NVIDIA Embed batch size.",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Detect changes without rebuilding.",
    )

    parser.add_argument(
        "--force",
        choices=[
            "diagnosis",
            "treatment",
            "all",
        ],
        default=None,
        help="Force rebuild of selected domain.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    INDEX_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Calculating knowledge-source fingerprints...")
    current = source_snapshot()

    previous = load_manifest()

    changed = changed_domains(
        previous,
        current,
    )

    if args.force:
        if args.force == "all":
            changed = [
                "diagnosis",
                "treatment",
            ]
        else:
            changed = [args.force]

    print_status(
        current,
        previous,
        changed,
    )

    if args.check_only:
        print()
        print(
            "CHECK ONLY: no indexes were rebuilt."
        )
        return

    if not changed:
        print()
        print(
            "All knowledge sources are unchanged."
        )
        print(
            "No index rebuild is required."
        )
        return

    rebuilt = []

    for domain in changed:
        run_build(
            domain,
            args.batch_size,
        )
        rebuilt.append(domain)

    save_manifest(
        current,
        rebuilt,
    )

    print()
    print("=" * 70)
    print("INDEX REFRESH COMPLETE")
    print("=" * 70)
    print(
        "Rebuilt:",
        ", ".join(rebuilt),
    )
    print(
        f"Manifest: {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
