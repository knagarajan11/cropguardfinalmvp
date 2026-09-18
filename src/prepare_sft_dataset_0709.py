#!/usr/bin/env python3

"""
CropGuard - Prepare SFT Dataset for Nemotron Vision-Language Fine-Tuning

Dataset structure:

/workspace/data/
├── raw/
│   ├── plantdoc/
│   ├── plantvillage/
│   └── rice1426/
│
├── processed/
│   ├── plantdoc_normalized.jsonl
│   ├── plantvillage_normalized.jsonl
│   └── rice1426_normalized.jsonl
│
├── sft/
│   ├── images/
│   ├── train.jsonl
│   └── validation.jsonl
│
└── test_images/
    └── plant.jpg

The normalized JSONL contains image paths such as:

/workspace/data/raw/plantdoc/powdery_mildew/train_001344.jpg

The script:
1. Reads normalized datasets
2. Resolves image paths
3. Selects a small balanced subset
4. Copies selected images to data/sft/images
5. Creates multimodal SFT JSONL
6. Creates train/validation split
7. Reports missing images and class distribution
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

WORKSPACE = Path("/workspace")
DATA_DIR = WORKSPACE / "data"

PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

OUTPUT_DIR = DATA_DIR / "sft"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"

DATASETS = {
    "PlantDoc": PROCESSED_DIR / "plantdoc_normalized.jsonl",
    "PlantVillage": PROCESSED_DIR / "plantvillage_normalized.jsonl",
    "Rice1426": PROCESSED_DIR / "rice1426_normalized.jsonl",
}

# Number of records to use from each dataset
DEFAULT_SAMPLES_PER_DATASET = 500

# Validation percentage
VALIDATION_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def read_jsonl(path):
    """Read JSONL file and return valid records."""

    records = []

    if not path.exists():
        print(f"[ERROR] File does not exist: {path}")
        return records

    print(f"Reading: {path}")

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARNING] Invalid JSON at line {line_number}: "
                    f"{path}"
                )
                continue

            if not isinstance(record, dict):
                continue

            records.append(record)

    return records


def resolve_image_path(image_path):
    """
    Resolve image path from normalized JSONL.

    Handles paths such as:

    /workspace/data/raw/plantdoc/...
    /data/raw/plantdoc/...
    relative/path.jpg
    """

    if not image_path:
        return None

    image_path = str(image_path)

    # --------------------------------------------------------
    # Case 1: absolute path exists directly
    # --------------------------------------------------------

    direct = Path(image_path)

    if direct.exists() and direct.is_file():
        return direct

    # --------------------------------------------------------
    # Case 2: normalized path begins with /workspace
    # --------------------------------------------------------

    if image_path.startswith("/workspace/"):
        candidate = WORKSPACE / image_path[len("/workspace/"):]

        if candidate.exists() and candidate.is_file():
            return candidate

    # --------------------------------------------------------
    # Case 3: path contains data/raw
    # --------------------------------------------------------

    marker = "data/raw/"

    if marker in image_path:

        relative_part = image_path.split(marker, 1)[1]

        candidate = RAW_DIR / relative_part

        if candidate.exists() and candidate.is_file():
            return candidate

    # --------------------------------------------------------
    # Case 4: try filename under raw directories
    # --------------------------------------------------------

    filename = Path(image_path).name

    matches = list(RAW_DIR.rglob(filename))

    if matches:
        return matches[0]

    return None


def get_disease(record):
    """Get disease label from normalized record."""

    disease = record.get("disease")

    if disease:
        return str(disease).strip()

    original_label = record.get("original_label")

    if original_label:
        return str(original_label).strip()

    return "unknown"


def get_crop(record):
    """Get crop name."""

    crop = record.get("crop")

    if crop:
        return str(crop).strip()

    return "unknown"


def get_health_status(record):
    """Get health status."""

    status = record.get("health_status")

    if status:
        return str(status).strip()

    return "Unknown"


# ============================================================
# BALANCED SAMPLING
# ============================================================

def balanced_sample(records, max_samples, seed=42):
    """
    Select approximately balanced samples across disease classes.

    This prevents PlantVillage or one common disease from
    dominating the initial SFT experiment.
    """

    random.seed(seed)

    class_records = {}

    for record in records:

        disease = get_disease(record)

        if disease == "unknown":
            continue

        class_records.setdefault(disease, []).append(record)

    classes = sorted(class_records.keys())

    if not classes:
        return []

    samples_per_class = max(1, max_samples // len(classes))

    selected = []

    for disease in classes:

        candidates = class_records[disease]

        random.shuffle(candidates)

        selected.extend(
            candidates[:samples_per_class]
        )

    # If there is room remaining, fill randomly
    if len(selected) < max_samples:

        selected_ids = {
            r.get("sample_id")
            for r in selected
        }

        remaining = [
            r
            for r in records
            if r.get("sample_id") not in selected_ids
        ]

        random.shuffle(remaining)

        needed = max_samples - len(selected)

        selected.extend(remaining[:needed])

    random.shuffle(selected)

    return selected[:max_samples]


# ============================================================
# CREATE SFT RECORD
# ============================================================

def create_sft_record(record, copied_image_path):

    crop = get_crop(record)
    disease = get_disease(record)
    health_status = get_health_status(record)

    source = record.get("source", "Unknown")

    # --------------------------------------------------------
    # Instruction
    # --------------------------------------------------------

    user_text = (
        "Analyze this plant image for possible disease.\n\n"
        "Identify:\n"
        "1. Crop\n"
        "2. Most likely disease or plant health condition\n"
        "3. Visual symptoms\n"
        "4. Confidence level\n"
        "5. Possible causes\n"
        "6. Organic treatment recommendations\n"
        "7. Preventive measures\n\n"
        "Focus only on what can reasonably be inferred "
        "from the image."
    )

    # --------------------------------------------------------
    # Training answer
    #
    # IMPORTANT:
    # We are using the normalized label as the ground-truth
    # disease. Symptoms/recommendations are deliberately kept
    # conservative for this first dataset preparation stage.
    # --------------------------------------------------------

    assistant_text = (
        f"Crop: {crop}\n"
        f"Most likely disease or condition: {disease}\n"
        f"Health status: {health_status}\n"
        f"Dataset source: {source}\n\n"
        f"The image is associated with the {disease} "
        f"condition in the training dataset.\n\n"
        "For a complete diagnosis, visual symptoms and "
        "field information should also be considered."
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(copied_image_path),
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_text,
                    }
                ],
            },
        ],
        "metadata": {
            "sample_id": record.get("sample_id"),
            "crop": crop,
            "disease": disease,
            "health_status": health_status,
            "source": source,
            "original_label": record.get("original_label"),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=DEFAULT_SAMPLES_PER_DATASET,
        help="Maximum samples to select from each dataset",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=VALIDATION_RATIO,
        help="Validation split ratio",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    print_header(
        "CropGuard - SFT Dataset Preparation"
    )

    print(f"Workspace       : {WORKSPACE}")
    print(f"Data directory  : {DATA_DIR}")
    print(f"Processed data  : {PROCESSED_DIR}")
    print(f"Raw images      : {RAW_DIR}")
    print(f"SFT output      : {OUTPUT_DIR}")
    print(f"Samples/dataset : {args.samples_per_dataset}")

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print_header(
        "[1] LOADING NORMALIZED DATASETS"
    )

    all_sft_records = []

    dataset_summary = []

    for dataset_name, jsonl_path in DATASETS.items():

        records = read_jsonl(jsonl_path)

        print(
            f"{dataset_name}: "
            f"{len(records):,} valid records"
        )

        if not records:
            print(
                f"[WARNING] No records found for "
                f"{dataset_name}"
            )
            continue

        selected = balanced_sample(
            records,
            args.samples_per_dataset,
            args.seed,
        )

        print(
            f"{dataset_name}: "
            f"selected {len(selected):,} records"
        )

        images_found = 0
        images_missing = 0

        for index, record in enumerate(selected):

            original_path = resolve_image_path(
                record.get("image_path")
            )

            if original_path is None:

                images_missing += 1

                continue

            images_found += 1

            sample_id = (
                record.get("sample_id")
                or f"{dataset_name}_{index}"
            )

            extension = original_path.suffix.lower()

            if extension not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            ]:
                extension = ".jpg"

            output_filename = (
                f"{dataset_name.lower()}_"
                f"{sample_id}"
                f"{extension}"
            )

            destination = (
                IMAGE_OUTPUT_DIR /
                output_filename
            )

            # Copy image
            shutil.copy2(
                original_path,
                destination,
            )

            sft_record = create_sft_record(
                record,
                destination,
            )

            all_sft_records.append(
                sft_record
            )

        dataset_summary.append(
            (
                dataset_name,
                len(records),
                len(selected),
                images_found,
                images_missing,
            )
        )

        print(
            f"{dataset_name}: "
            f"{images_found} images found, "
            f"{images_missing} missing"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header(
        "[2] DATASET SUMMARY"
    )

    print(
        f"{'Dataset':15} | "
        f"{'Available':>10} | "
        f"{'Selected':>10} | "
        f"{'Images OK':>10} | "
        f"{'Missing':>10}"
    )

    print("-" * 70)

    for row in dataset_summary:

        print(
            f"{row[0]:15} | "
            f"{row[1]:10,} | "
            f"{row[2]:10,} | "
            f"{row[3]:10,} | "
            f"{row[4]:10,}"
        )

    print("-" * 70)

    print(
        f"Total usable SFT records: "
        f"{len(all_sft_records):,}"
    )

    if not all_sft_records:

        print()
        print(
            "[ERROR] No usable records found."
        )
        print(
            "Check that the raw image directories "
            "contain the images referenced by the "
            "normalized JSONL files."
        )
        return

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(all_sft_records)

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    validation_count = max(
        1,
        int(
            len(all_sft_records)
            * args.validation_ratio
        ),
    )

    validation_records = (
        all_sft_records[:validation_count]
    )

    train_records = (
        all_sft_records[validation_count:]
    )

    # --------------------------------------------------------
    # Write JSONL
    # --------------------------------------------------------

    train_path = OUTPUT_DIR / "train.jsonl"
    validation_path = (
        OUTPUT_DIR / "validation.jsonl"
    )

    with open(
        train_path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in train_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(
        validation_path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in validation_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Disease distribution
    # --------------------------------------------------------

    train_diseases = Counter(
        record["metadata"]["disease"]
        for record in train_records
    )

    validation_diseases = Counter(
        record["metadata"]["disease"]
        for record in validation_records
    )

    print_header(
        "[3] SFT DATASET CREATED"
    )

    print(
        f"Training records   : "
        f"{len(train_records):,}"
    )

    print(
        f"Validation records : "
        f"{len(validation_records):,}"
    )

    print(
        f"Training JSONL     : "
        f"{train_path}"
    )

    print(
        f"Validation JSONL   : "
        f"{validation_path}"
    )

    print(
        f"Training images    : "
        f"{IMAGE_OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Disease distribution
    # --------------------------------------------------------

    print_header(
        "[4] TRAINING DISEASE DISTRIBUTION"
    )

    for disease, count in sorted(
        train_diseases.items()
    ):

        print(
            f"{disease:45} "
            f"{count:6}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print_header(
        "[5] VALIDATION"
    )

    print(
        f"Train JSONL exists: "
        f"{train_path.exists()}"
    )

    print(
        f"Validation JSONL exists: "
        f"{validation_path.exists()}"
    )

    print(
        f"Image directory exists: "
        f"{IMAGE_OUTPUT_DIR.exists()}"
    )

    image_count = len(
        list(
            IMAGE_OUTPUT_DIR.glob("*")
        )
    )

    print(
        f"Copied images: {image_count:,}"
    )

    print_header(
        "SFT DATASET PREPARATION COMPLETE"
    )


if __name__ == "__main__":
    main()
