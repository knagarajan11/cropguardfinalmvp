import os
import json
import csv
import logging
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_directory(raw_dir, source_name):
    metadata = []

    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if "___" in class_name:
            parts = class_name.split("___", 1)
            crop = parts[0]
            disease = parts[1]
        elif "__" in class_name:
            parts = class_name.split("__", 1)
            crop = parts[0]
            disease = parts[1]
        else:
            # Fallback heuristic for directory names that start with a crop
            known_crops = ["Tomato", "Apple", "Potato", "Corn", "Grape", "Peach", "Pepper", "Bell_pepper", "Strawberry", "Cherry", "Blueberry", "Raspberry", "Soybean", "Soyabean", "Squash", "Rice"]
            matched = False
            for kc in known_crops:
                if class_name.lower().startswith(kc.lower()):
                    crop = kc
                    disease = class_name[len(kc):].strip(" _-")
                    matched = True
                    break
            if not matched:
                parts = class_name.split("_")
                first_word = parts[0].lower()
                known_crops_lower = [kc.lower() for kc in known_crops]
                if first_word in known_crops_lower:
                    crop = parts[0]
                    disease = "_".join(parts[1:]) if len(parts) > 1 else "Unknown"
                else:
                    crop = "Unknown"
                    disease = class_name

        # Normalize special crop names
        if crop.lower() in ["bell_pepper", "bell pepper", "pepper"]:
            crop = "Pepper"
        elif crop.lower() in ["soyabean", "soybean"]:
            crop = "Soybean"
        else:
            crop = crop.capitalize()

        disease = disease.strip(" _-")
        if disease.lower() in ["healthy", "leaf", ""] or "healthy" in class_name.lower() or "healthy" in disease.lower():
            health_status = "Healthy"
            disease = "healthy"
        else:
            health_status = "Diseased"

        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            sample_id = str(uuid.uuid4())
            metadata.append({
                "sample_id": sample_id,
                "image_path": str(img_path.absolute()),
                "crop": crop,
                "disease": disease,
                "health_status": health_status,
                "source": source_name,
                "original_label": class_name
            })

    return metadata

def save_metadata(metadata, out_csv, out_jsonl):
    if not metadata:
        logging.warning("No metadata to save!")
        return

    keys = metadata[0].keys()
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadata)

    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    logging.info(f"Saved {len(metadata)} records to {out_csv} and {out_jsonl}")

def main():
    raw_dir = Path(os.getenv("PLANTDOC_DIR", "data/raw/plantdoc"))
    out_dir = Path("data/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Preparing PlantDoc from {raw_dir}")
    metadata = process_directory(raw_dir, "PlantDoc")

    save_metadata(metadata, out_dir / "plantdoc.csv", out_dir / "plantdoc.jsonl")

if __name__ == "__main__":
    main()
