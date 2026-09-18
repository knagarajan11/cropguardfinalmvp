import json
import logging
import yaml
import random
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config("configs/dataset.yaml")
    random.seed(config.get("random_seed", 42))

    processed_dir = Path("data/processed")
    all_records = []

    for jsonl_path in processed_dir.glob("*_normalized.jsonl"):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_records.append(json.loads(line))

    if not all_records:
        logging.warning("No records found to split.")
        return

    # Stratification logic based on config
    # For simplicity in this implementation, we group by crop+disease and split
    grouped = defaultdict(list)
    for r in all_records:
        key = f"{r['crop']}_{r['disease']}_{r['source']}"
        grouped[key].append(r)

    train_records = []
    val_records = []
    test_records = []

    ratios = config.get("split_ratios", {"train": 0.8, "validation": 0.1, "test": 0.1})

    for key, group in grouped.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * ratios['train'])
        n_val = int(n * ratios['validation'])

        train_records.extend(group[:n_train])
        val_records.extend(group[n_train:n_train + n_val])
        test_records.extend(group[n_train + n_val:])

    logging.info(f"Split complete. Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # Save splits
    splits = {
        "data/train/train.jsonl": train_records,
        "data/validation/val.jsonl": val_records,
        "data/test/test.jsonl": test_records
    }

    for out_path, records in splits.items():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    logging.info("Splits saved successfully.")

if __name__ == "__main__":
    main()
