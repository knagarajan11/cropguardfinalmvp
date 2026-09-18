#!/usr/bin/env python3

import json
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROCESSED = Path("/workspace/data/processed")

PV_FILE = PROCESSED / "plantvillage_normalized.jsonl"
PD_FILE = PROCESSED / "plantdoc_normalized.jsonl"

PV_RAW = Path("/workspace/data/raw/plantvillage")
PD_RAW = Path("/workspace/data/raw/plantdoc")

BACKUP_DIR = PROCESSED / "backup_before_final_repair"


# ============================================================
# BACKUP
# ============================================================

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

shutil.copy2(
    PV_FILE,
    BACKUP_DIR / "plantvillage_normalized.jsonl"
)

shutil.copy2(
    PD_FILE,
    BACKUP_DIR / "plantdoc_normalized.jsonl"
)


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(path, records):
    tmp = path.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    tmp.replace(path)


# ============================================================
# BUILD PLANTVILLAGE HEALTHY MAPPING
# ============================================================

print("=" * 80)
print("BUILDING PLANTVILLAGE HEALTHY MAPPING")
print("=" * 80)

healthy_dir = PV_RAW / "healthy"

healthy_mapping = {}

for generic_file in healthy_dir.glob("*"):

    if not generic_file.is_file():
        continue

    matches = []

    for crop_dir in PV_RAW.iterdir():

        if not crop_dir.is_dir():
            continue

        if crop_dir.name == "healthy":
            continue

        candidate = crop_dir / generic_file.name

        if candidate.exists():
            matches.append(crop_dir.name)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one mapping for "
            f"{generic_file.name}, found {matches}"
        )

    original_label = matches[0]

    # PlantVillage labels have forms such as:
    # Tomato___healthy
    # Tomato___Early_blight

    if "___" in original_label:
        crop, disease = original_label.split("___", 1)
    else:
        raise RuntimeError(
            f"Unexpected PlantVillage label: {original_label}"
        )

    if disease.lower() != "healthy":
        raise RuntimeError(
            f"Generic healthy image mapped to non-healthy label: "
            f"{generic_file.name} -> {original_label}"
        )

    healthy_mapping[generic_file.name] = {
        "crop": crop,
        "disease": "healthy",
        "original_label": original_label,
        "health_status": "Healthy",
    }


print("Healthy mappings:", len(healthy_mapping))


# ============================================================
# REPAIR PLANTVILLAGE
# ============================================================

print()
print("=" * 80)
print("REPAIRING PLANTVILLAGE")
print("=" * 80)

pv_records = load_jsonl(PV_FILE)

pv_repaired = 0

for record in pv_records:

    crop = str(record.get("crop", ""))

    if crop != "Unknown":
        continue

    image_path = str(record.get("image_path", ""))

    filename = Path(image_path).name

    mapping = healthy_mapping.get(filename)

    if mapping is None:
        raise RuntimeError(
            f"No healthy mapping for {filename}"
        )

    record["crop"] = mapping["crop"]
    record["disease"] = mapping["disease"]
    record["health_status"] = mapping["health_status"]
    record["original_label"] = mapping["original_label"]

    pv_repaired += 1


save_jsonl(PV_FILE, pv_records)

print("PlantVillage repaired:", pv_repaired)


# ============================================================
# BUILD PLANTDOC MAPPINGS
# ============================================================

print()
print("=" * 80)
print("BUILDING PLANTDOC MAPPINGS")
print("=" * 80)

plantdoc_mapping = {}


def build_pd_mapping(generic_name):

    generic_dir = PD_RAW / generic_name

    mapping = {}

    for generic_file in generic_dir.glob("*"):

        if not generic_file.is_file():
            continue

        matches = []

        for disease_dir in PD_RAW.iterdir():

            if not disease_dir.is_dir():
                continue

            if disease_dir.name == generic_name:
                continue

            candidate = disease_dir / generic_file.name

            if candidate.exists():
                matches.append(disease_dir.name)

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one PlantDoc mapping for "
                f"{generic_file.name}, found {matches}"
            )

        original_label = matches[0]

        # Examples:
        # Apple___rust
        # Corn___rust
        # Apple___scab
        # Tomato___mold

        if "___" not in original_label:
            raise RuntimeError(
                f"Unexpected PlantDoc label: {original_label}"
            )

        crop, disease = original_label.split("___", 1)

        mapping[generic_file.name] = {
            "crop": crop,
            "disease": disease,
            "original_label": original_label,
            "health_status": "Diseased",
        }

    return mapping


for generic_name in ["rust", "scab", "mold"]:

    mapping = build_pd_mapping(generic_name)

    plantdoc_mapping.update(mapping)

    print(
        f"{generic_name:10} : {len(mapping)} mappings"
    )


print(
    "Total PlantDoc mappings:",
    len(plantdoc_mapping)
)


# ============================================================
# REPAIR PLANTDOC
# ============================================================

print()
print("=" * 80)
print("REPAIRING PLANTDOC")
print("=" * 80)

pd_records = load_jsonl(PD_FILE)

pd_repaired = 0

for record in pd_records:

    disease = str(record.get("disease", ""))

    if disease.upper() != "UNKNOWN":
        continue

    image_path = str(record.get("image_path", ""))

    filename = Path(image_path).name

    mapping = plantdoc_mapping.get(filename)

    if mapping is None:
        raise RuntimeError(
            f"No PlantDoc mapping for {filename}"
        )

    record["crop"] = mapping["crop"]
    record["disease"] = mapping["disease"]
    record["health_status"] = mapping["health_status"]
    record["original_label"] = mapping["original_label"]

    pd_repaired += 1


save_jsonl(PD_FILE, pd_records)

print("PlantDoc repaired:", pd_repaired)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("NORMALIZED DATASET REPAIR COMPLETE")
print("=" * 80)

print("PlantVillage repaired:", pv_repaired)
print("PlantDoc repaired     :", pd_repaired)
print("Total repaired        :", pv_repaired + pd_repaired)

print("=" * 80)
