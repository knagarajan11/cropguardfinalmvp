from pathlib import Path
import json
import csv
import shutil
from collections import defaultdict, Counter

ROOT = Path("/workspace/CropGuard")
RAW = ROOT / "data/raw"
META = ROOT / "data/metadata"

PLANTVILLAGE_JSON = META / "plantvillage.jsonl"
PLANTDOC_JSON = META / "plantdoc.jsonl"

# ------------------------------------------------------------
# Disease normalization
# ------------------------------------------------------------

def normalize_disease(label):
    if not label:
        return "UNKNOWN"

    x = label.strip().lower()

    mapping = {
        "healthy": "healthy",

        "apple_scab": "apple_scab",
        "scab": "apple_scab",

        "rust": "common_rust",
        "corn_rust": "common_rust",
        "common_rust": "common_rust",
        "apple_rust": "cedar_apple_rust",
        "cedar_apple_rust": "cedar_apple_rust",

        "mold": "leaf_mold",
        "tomato_mold": "leaf_mold",
        "leaf_mold": "leaf_mold",

        "bacterial_spot": "bacterial_spot",
        "early_blight": "early_blight",
        "late_blight": "late_blight",
        "powdery_mildew": "powdery_mildew",
        "leaf_blight": "leaf_blight",
        "septoria_leaf_spot": "septoria_leaf_spot",
        "mosaic_virus": "mosaic_virus",
        "yellow_virus": "yellow_leaf_curl_virus",
        "yellow_leaf_curl_virus": "yellow_leaf_curl_virus",
        "two_spotted_spider_mites": "spider_mites",
        "spider_mites": "spider_mites",
        "black_rot": "black_rot",
        "citrus_greening": "citrus_greening",
        "esca": "esca",
        "leaf_scorch": "leaf_scorch",
        "target_spot": "target_spot",
        "northern_leaf_blight": "northern_leaf_blight",
        "cercospora_leaf_spot": "cercospora_leaf_spot",
        "brown_spot": "brown_spot",
        "bacterial_leaf_blight": "bacterial_leaf_blight",
        "leaf_blast": "leaf_blast",
        "sheath_blight": "sheath_blight",
        "scald": "scald",
    }

    return mapping.get(x, x)


# ------------------------------------------------------------
# Crop normalization
# ------------------------------------------------------------

def normalize_crop(name):
    if not name:
        return "Unknown"

    x = name.strip()

    mapping = {
        "Apple": "Apple",
        "Blueberry": "Blueberry",
        "Cherry": "Cherry",
        "Corn": "Corn",
        "Grape": "Grape",
        "Orange": "Orange",
        "Peach": "Peach",
        "Pepper": "Pepper",
        "Potato": "Potato",
        "Raspberry": "Raspberry",
        "Rice": "Rice",
        "Soybean": "Soybean",
        "Soyabean": "Soybean",
        "Squash": "Squash",
        "Strawberry": "Strawberry",
        "Tomato": "Tomato",
    }

    return mapping.get(x, x)


# ------------------------------------------------------------
# Build filename -> crop mappings
# ------------------------------------------------------------

def build_plantvillage_healthy_mapping():

    base = RAW / "plantvillage"

    generic = base / "healthy"

    generic_files = {
        p.name for p in generic.iterdir()
        if p.is_file()
    }

    mapping = {}
    ambiguous = []

    for p in base.iterdir():

        if not p.is_dir():
            continue

        name = p.name

        # We only need directories ending in ___healthy
        if not name.endswith("___healthy"):
            continue

        crop_part = name.split("___")[0]

        # Normalize special PlantVillage names
        if crop_part == "Cherry_(including_sour)":
            crop = "Cherry"
        elif crop_part == "Corn_(maize)":
            crop = "Corn"
        elif crop_part == "Pepper,_bell":
            crop = "Pepper"
        elif crop_part == "Soybean":
            crop = "Soybean"
        else:
            crop = normalize_crop(crop_part)

        for img in p.iterdir():

            if not img.is_file():
                continue

            if img.name not in generic_files:
                continue

            if img.name in mapping:
                ambiguous.append(img.name)
            else:
                mapping[img.name] = crop

    print("\nPlantVillage generic healthy mapping")
    print("------------------------------------")
    print("Generic images :", len(generic_files))
    print("Mapped images  :", len(mapping))
    print("Ambiguous      :", len(ambiguous))
    print("Unmapped       :", len(generic_files - set(mapping)))

    if ambiguous:
        raise RuntimeError(
            f"Ambiguous PlantVillage healthy mappings: {ambiguous[:10]}"
        )

    if generic_files - set(mapping):
        raise RuntimeError(
            f"Unmapped PlantVillage healthy images: "
            f"{list(generic_files - set(mapping))[:10]}"
        )

    return mapping


def build_plantdoc_generic_mapping():

    base = RAW / "plantdoc"

    mapping = {}

    # These were proven to have unique mappings
    generic_rules = {
        "rust": {
            "Apple___rust": "Apple",
            "Corn___rust": "Corn",
        },
        "scab": {
            "Apple___scab": "Apple",
        },
        "mold": {
            "Tomato___mold": "Tomato",
        },
    }

    for generic, targets in generic_rules.items():

        generic_dir = base / generic

        if not generic_dir.exists():
            raise RuntimeError(f"Missing directory: {generic_dir}")

        generic_files = {
            p.name for p in generic_dir.iterdir()
            if p.is_file()
        }

        found = {}

        for target_dir, crop in targets.items():

            d = base / target_dir

            if not d.exists():
                raise RuntimeError(f"Missing directory: {d}")

            for img in d.iterdir():

                if not img.is_file():
                    continue

                if img.name in generic_files:

                    if img.name in found:
                        raise RuntimeError(
                            f"Ambiguous PlantDoc {generic}/{img.name}"
                        )

                    found[img.name] = (
                        crop,
                        normalize_disease(generic)
                    )

        missing = generic_files - set(found)

        if missing:
            raise RuntimeError(
                f"Unmapped PlantDoc {generic} images: "
                f"{list(missing)[:10]}"
            )

        for filename, value in found.items():
            mapping[(generic, filename)] = value

        print(
            f"PlantDoc {generic}: "
            f"{len(generic_files)} images mapped"
        )

    return mapping


# ------------------------------------------------------------
# Repair PlantVillage metadata
# ------------------------------------------------------------

def repair_plantvillage():

    mapping = build_plantvillage_healthy_mapping()

    records = []
    changed = 0

    with PLANTVILLAGE_JSON.open() as f:

        for line in f:

            r = json.loads(line)

            image_path = r.get("image_path", "")
            path = Path(image_path)

            # Only repair the generic healthy directory
            if (
                path.parent.name == "healthy"
                and path.name in mapping
                and r.get("crop") == "Unknown"
                and r.get("disease") == "healthy"
            ):

                old_crop = r.get("crop")
                r["crop"] = mapping[path.name]
                r["disease"] = "healthy"

                changed += 1

                print(
                    f"PV FIX: {old_crop} -> {r['crop']} "
                    f"{path.name}"
                )

            records.append(r)

    backup = PLANTVILLAGE_JSON.with_suffix(".jsonl.label_fix_backup")

    if not backup.exists():
        shutil.copy2(PLANTVILLAGE_JSON, backup)

    with PLANTVILLAGE_JSON.open("w") as f:

        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nPlantVillage repaired:", changed)

    return changed


# ------------------------------------------------------------
# Repair PlantDoc metadata
# ------------------------------------------------------------

def repair_plantdoc():

    mapping = build_plantdoc_generic_mapping()

    records = []
    changed = 0

    with PLANTDOC_JSON.open() as f:

        for line in f:

            r = json.loads(line)

            image_path = r.get("image_path", "")
            path = Path(image_path)

            generic = path.parent.name

            key = (generic, path.name)

            if (
                key in mapping
                and r.get("crop") in {
                    "Rust",
                    "Scab",
                    "Mold",
                }
                and r.get("disease") == "UNKNOWN"
            ):

                crop, disease = mapping[key]

                r["crop"] = crop
                r["disease"] = disease

                changed += 1

                print(
                    f"PD FIX: {generic} -> "
                    f"{crop}/{disease} {path.name}"
                )

            records.append(r)

    backup = PLANTDOC_JSON.with_suffix(".jsonl.label_fix_backup")

    if not backup.exists():
        shutil.copy2(PLANTDOC_JSON, backup)

    with PLANTDOC_JSON.open("w") as f:

        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nPlantDoc repaired:", changed)

    return changed


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

print("=" * 80)
print("CROPGUARD GENERIC LABEL REPAIR")
print("=" * 80)

pv_changed = repair_plantvillage()
pd_changed = repair_plantdoc()

print("\n" + "=" * 80)
print("REPAIR SUMMARY")
print("=" * 80)
print("PlantVillage records repaired:", pv_changed)
print("PlantDoc records repaired    :", pd_changed)
print("Total repaired               :", pv_changed + pd_changed)
print("=" * 80)
