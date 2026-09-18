#!/usr/bin/env python3

import argparse
import json
import random
import uuid
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

PLANTDOC = "/workspace/data/processed/plantdoc_normalized.jsonl"
PLANTVILLAGE = "/workspace/data/processed/plantvillage_normalized.jsonl"
RICE1426 = "/workspace/data/processed/rice1426_normalized.jsonl"

OUTPUT_DIR = Path("/workspace/data/sft")
IMAGE_DIR = OUTPUT_DIR / "images"

TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation.jsonl"

VALIDATION_RATIO = 0.10
RANDOM_SEED = 42


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Prepare full CropGuard multimodal SFT dataset"
)

parser.add_argument(
    "--full-dataset",
    action="store_true",
    help="Use every valid record from all datasets",
)

parser.add_argument(
    "--seed",
    type=int,
    default=RANDOM_SEED,
)

args = parser.parse_args()


# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET FILES
# ============================================================

DATASETS = [
    ("PlantDoc", PLANTDOC),
    ("PlantVillage", PLANTVILLAGE),
    ("Rice1426", RICE1426),
]


# ============================================================
# HELPERS
# ============================================================

def read_jsonl(path):
    """
    Read JSONL records.
    """

    records = []

    path = Path(path)

    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return records

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as e:

                print(
                    f"[WARNING] Invalid JSON "
                    f"{path}:{line_no}: {e}"
                )

    return records


def get_image_path(record):
    """
    Extract image path from normalized record.
    """

    possible_keys = [
        "image",
        "image_path",
        "filepath",
        "file_path",
        "path",
    ]

    for key in possible_keys:

        value = record.get(key)

        if value:
            return str(value)

    return None


def get_crop(record):
    """
    Extract crop name.
    """

    for key in [
        "crop",
        "crop_name",
        "plant",
        "plant_name",
    ]:

        value = record.get(key)

        if value:
            return str(value)

    return "Unknown"


def get_disease(record):
    """
    Extract disease/condition.
    """

    for key in [
        "disease",
        "disease_name",
        "condition",
        "label",
        "original_label",
    ]:

        value = record.get(key)

        if value:
            return str(value)

    return "unknown"


def normalize_path(image_path):
    """
    Convert dataset image path into an absolute path.
    """

    if image_path is None:
        return None

    p = Path(image_path)

    if p.is_absolute():
        return str(p)

    # Relative paths are interpreted relative to /workspace
    return str(
        Path("/workspace") / p
    )


def create_sample(record, source):
    """
    Convert one normalized dataset record
    into the CropGuard multimodal SFT format.
    """

    image_path = get_image_path(record)

    if not image_path:
        return None

    image_path = normalize_path(image_path)

    if not Path(image_path).exists():
        return None

    crop = get_crop(record)
    disease = get_disease(record)

    sample_id = record.get(
        "sample_id",
        str(uuid.uuid4())
    )

    # --------------------------------------------------------
    # USER MESSAGE
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
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    health_status = (
        "Healthy"
        if str(disease).lower() in [
            "healthy",
            "normal",
            "no disease",
            "no_disease",
        ]
        else "Diseased"
    )

    assistant_text = (
        f"Crop: {crop}\n"
        f"Most likely disease or condition: {disease}\n"
        f"Health status: {health_status}\n"
        f"Dataset source: {source}\n\n"
        f"The image is associated with the "
        f"{disease} condition in the training dataset.\n\n"
        "For a complete diagnosis, visual symptoms "
        "and field information should also be considered."
    )

    # --------------------------------------------------------
    # SFT RECORD
    # --------------------------------------------------------

    sample = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "text": None,
                    },
                    {
                        "type": "text",
                        "image": None,
                        "text": user_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "image": None,
                        "text": assistant_text,
                    }
                ],
            },
        ],
        "metadata": {
            "sample_id": sample_id,
            "crop": crop,
            "disease": disease,
            "health_status": health_status,
            "source": source,
            "original_label": record.get(
                "original_label",
                disease
            ),
        },
    }

    return sample


# ============================================================
# LOAD ALL DATASETS
# ============================================================

print("=" * 70)
print("CropGuard - FULL SFT DATA PREPARATION")
print("=" * 70)

print("\nLoading datasets...")

all_records = []

dataset_counts = {}

for source, path in DATASETS:

    print(f"\n[{source}]")
    print(f"File: {path}")

    records = read_jsonl(path)

    print(
        f"Records loaded: {len(records):,}"
    )

    dataset_counts[source] = len(records)

    for record in records:

        sample = create_sample(
            record,
            source
        )

        if sample is not None:
            all_records.append(sample)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SOURCE DATASET SUMMARY")
print("=" * 70)

for source, count in dataset_counts.items():

    print(
        f"{source:<20}: {count:>10,}"
    )

print(
    f"{'TOTAL SOURCE RECORDS':<20}: "
    f"{sum(dataset_counts.values()):>10,}"
)

print(
    f"{'VALID SFT RECORDS':<20}: "
    f"{len(all_records):>10,}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\nRemoving duplicate samples...")

seen = set()
unique_records = []

for record in all_records:

    metadata = record.get(
        "metadata",
        {}
    )

    sample_id = metadata.get(
        "sample_id"
    )

    image_path = None

    try:
        image_path = (
            record["conversation"][0]
            ["content"][0]
            ["image"]
        )
    except Exception:
        pass

    # Use sample ID primarily.
    # Fall back to image path.
    dedup_key = sample_id or image_path

    if dedup_key in seen:
        continue

    seen.add(dedup_key)
    unique_records.append(record)

all_records = unique_records

print(
    f"Unique valid records: "
    f"{len(all_records):,}"
)


# ============================================================
# SHUFFLE
# ============================================================

random.seed(args.seed)

random.shuffle(all_records)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

validation_count = int(
    len(all_records) * VALIDATION_RATIO
)

train_count = (
    len(all_records)
    - validation_count
)

train_records = all_records[
    :train_count
]

validation_records = all_records[
    train_count:
]


# ============================================================
# SAVE JSONL
# ============================================================

print("\nWriting SFT datasets...")

with open(
    TRAIN_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    for record in train_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


with open(
    VALIDATION_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    for record in validation_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SFT PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Total valid records : "
    f"{len(all_records):,}"
)

print(
    f"Training records    : "
    f"{len(train_records):,}"
)

print(
    f"Validation records  : "
    f"{len(validation_records):,}"
)

print(
    f"Validation ratio    : "
    f"{VALIDATION_RATIO:.0%}"
)

print(
    f"\nTraining file:"
    f"\n{TRAIN_OUTPUT}"
)

print(
    f"\nValidation file:"
    f"\n{VALIDATION_OUTPUT}"
)

print("\nDataset sources:")

for source, count in dataset_counts.items():

    print(
        f"  {source:<20} {count:,}"
    )

print("\nDone.")
