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
        crop = "Rice"
        disease = class_name
        health_status = "Healthy" if "healthy" in class_name.lower() else "Diseased"

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
    raw_dir = Path(os.getenv("RICE1426_DIR", "data/raw/rice1426"))
    out_dir = Path("data/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Preparing Rice1426 from {raw_dir}")
    metadata = process_directory(raw_dir, "Rice1426")

    save_metadata(metadata, out_dir / "rice1426.csv", out_dir / "rice1426.jsonl")

if __name__ == "__main__":
    main()
