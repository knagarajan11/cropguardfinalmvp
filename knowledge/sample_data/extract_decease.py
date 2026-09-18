#!/usr/bin/env python3

import json
import os


# ============================================================
# EXISTING NORMALIZED DATASETS
# ============================================================

DATASETS = {
    "PlantVillage":
        "/workspace/data_struc/normalized/PlantVillage/plantvillage_normalized.jsonl",

    "PlantDoc":
        "/workspace/data_struc/normalized/PlantDoc/plantdoc_normalized.jsonl",

    "Rice1426":
        "/workspace/data_struc/normalized/Rice1426/rice1426_normalized.jsonl"
}


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = "/workspace/data_struc/crop_disease_names.json"


# ============================================================
# READ CROP + DISEASE
# ============================================================

def extract_crop_disease(dataset_name, path):

    print("\n" + "=" * 80)
    print(dataset_name)
    print("=" * 80)

    print(f"File: {path}")

    if not os.path.exists(path):
        print("WARNING: File not found")
        return []

    results = []
    seen = set()

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                print(
                    f"WARNING: Invalid JSON at line "
                    f"{line_number}"
                )
                continue

            crop = record.get("crop")
            disease = record.get("disease")

            if crop is None or disease is None:
                continue

            # EXACT values from your dataset
            crop = str(crop)
            disease = str(disease)

            key = (crop, disease)

            if key not in seen:

                seen.add(key)

                results.append({
                    "crop": crop,
                    "disease": disease
                })

    # Sort only for easier viewing.
    # Values themselves are NOT changed.
    results.sort(
        key=lambda x: (x["crop"], x["disease"])
    )

    print(f"\nUnique crop + disease combinations: {len(results)}")
    print()

    for i, item in enumerate(results, 1):

        print(
            f"{i:3}. "
            f"Crop: {item['crop']:<20} "
            f"Disease: {item['disease']}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CropGuard - Crop + Disease Extraction")
    print("=" * 80)
    print()
    print("Reading existing normalized JSONL files.")
    print("Crop names: EXACT")
    print("Disease names: EXACT")
    print("No normalization.")
    print("No renaming.")
    print("No hard-coded disease list.")
    print()

    output = {}

    for dataset_name, path in DATASETS.items():

        output[dataset_name] = extract_crop_disease(
            dataset_name,
            path
        )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = sum(
        len(items)
        for items in output.values()
    )

    print()
    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)

    for dataset_name, items in output.items():

        print(
            f"{dataset_name:<20}: "
            f"{len(items)} crop-disease combinations"
        )

    print(
        f"{'TOTAL':<20}: {total}"
    )

    print()
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Crop and disease names were copied exactly")
    print("from your normalized datasets.")


if __name__ == "__main__":
    main()
