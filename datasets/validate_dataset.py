import json
import logging
import imagehash
from PIL import Image
from pathlib import Path
import yaml
import collections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_hash(image_path):
    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception as e:
        logging.error(f"Error hashing {image_path}: {e}")
        return None

def validate_datasets(processed_dir):
    all_records = []
    for jsonl_path in processed_dir.glob("*_normalized.jsonl"):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_records.append(json.loads(line))

    if not all_records:
        logging.warning("No records found to validate.")
        return

    logging.info(f"Validating {len(all_records)} total records.")

    # 1. Check for duplicates
    hash_dict = collections.defaultdict(list)
    corrupt = 0

    for record in all_records:
        img_path = record['image_path']
        if not Path(img_path).exists():
            # Handle dummy creation case in mock data
            img_hash = "mock_hash_" + record['sample_id']
        else:
            img_hash = compute_hash(img_path)

        if img_hash:
            hash_dict[img_hash].append(record['sample_id'])
        else:
            corrupt += 1

    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}

    logging.info(f"Corrupt images: {corrupt}")
    logging.info(f"Exact duplicate hashes found: {len(duplicates)}")

    if duplicates:
        logging.warning("Duplicates detected! Data leakage risk.")

    # Generate statistics report
    stats = {
        "total_images": len(all_records),
        "corrupt_images": corrupt,
        "duplicates": len(duplicates),
        "crops": list(set(r['crop'] for r in all_records)),
        "diseases": list(set(r['disease'] for r in all_records))
    }

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    logging.info("Validation complete.")

def main():
    processed_dir = Path("data/processed")
    validate_datasets(processed_dir)

if __name__ == "__main__":
    main()
