import os
import json
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mapping(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def build_reverse_lookup(mapping):
    """
    Build a flat dict mapping cleaned alias strings -> (crop_key, canonical_disease).
    Used as a global cross-crop fallback when the primary crop lookup fails
    (e.g. when the folder name has no crop prefix and 'Bacterial' becomes the crop).
    First alias encountered wins to keep it deterministic.
    """
    reverse = {}
    for crop_key, diseases in mapping.items():
        for canonical, info in diseases.items():
            canonical_clean = canonical.lower().replace("-", "_").replace(" ", "_").strip(" _")
            all_aliases = [canonical_clean]
            if isinstance(info, dict) and 'aliases' in info:
                all_aliases += [
                    a.lower().replace("-", "_").replace(" ", "_").strip(" _")
                    for a in info['aliases']
                ]
            for alias_clean in all_aliases:
                if alias_clean and alias_clean not in reverse:
                    reverse[alias_clean] = (crop_key, canonical)
    return reverse

def normalize_label(crop, disease, mapping, reverse_lookup=None):
    crop_lower = crop.lower().strip()
    crop_cap = crop_lower.capitalize()

    # Standardize disease string: lowercase, underscore separators, stripped
    disease_clean = disease.lower().replace("-", "_").replace(" ", "_").strip(" _")

    # 1. Universal healthy check across any crop
    if disease_clean == "healthy" or "healthy" in disease_clean:
        return crop_cap, "healthy"

    # 2. Check mapping table using the crop key
    if crop_lower in mapping:
        crop_mapping = mapping[crop_lower]
        for canonical_disease, info in crop_mapping.items():
            canonical_clean = canonical_disease.lower().replace("-", "_").replace(" ", "_").strip(" _")
            aliases = [canonical_clean]
            if isinstance(info, dict) and 'aliases' in info:
                aliases += [a.lower().replace("-", "_").replace(" ", "_").strip(" _") for a in info['aliases']]

            if disease_clean in aliases:
                return crop_cap, canonical_disease
            # Substring check: alias contained in disease_clean (always safe)
            # or disease_clean contained in alias (only when disease_clean is long
            # enough to not be a generic word like 'leaf', 'rust', etc.)
            for a in aliases:
                if len(a) > 4 and a in disease_clean:
                    return crop_cap, canonical_disease
                if len(disease_clean) > 4 and disease_clean in a:
                    return crop_cap, canonical_disease

    # 3. Global reverse lookup fallback.
    # Handles disease-only folder names where a disease word (e.g. 'Bacterial',
    # 'Mold', 'Rust', 'Scab', 'Haunglongbing') was mistakenly parsed as the crop.
    if reverse_lookup:
        combined = f"{crop_lower}_{disease_clean}".strip("_")
        for candidate in [combined, disease_clean, crop_lower]:
            if candidate and candidate != "unknown" and candidate in reverse_lookup:
                found_crop, found_disease = reverse_lookup[candidate]
                return found_crop.capitalize(), found_disease
        # Substring match on combined string or individual tokens
        for alias, (found_crop, found_disease) in reverse_lookup.items():
            if len(alias) >= 4 and (alias in combined or alias in disease_clean or alias in crop_lower):
                return found_crop.capitalize(), found_disease
            if len(combined) > 5 and combined in alias:
                return found_crop.capitalize(), found_disease

    # 4. Clean fallback: produce standardized snake_case disease instead of discarding to UNKNOWN
    cleaned = disease_clean.replace("leaf_", "").replace("_leaf", "").strip(" _")
    if cleaned and cleaned != "unknown":
        return crop_cap, cleaned

    return crop_cap, "UNKNOWN"

def process_file(jsonl_path, mapping, out_path, reverse_lookup=None):
    normalized_data = []

    if not jsonl_path.exists():
        logging.warning(f"File {jsonl_path} does not exist.")
        return

    unknown_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            crop, disease = record['crop'], record['disease']

            norm_crop, norm_disease = normalize_label(crop, disease, mapping, reverse_lookup)

            if norm_disease == "UNKNOWN":
                unknown_count += 1

            record['crop'] = norm_crop
            record['disease'] = norm_disease
            normalized_data.append(record)

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in normalized_data:
            f.write(json.dumps(item) + "\n")

    logging.info(f"Normalized {len(normalized_data)} records to {out_path} (UNKNOWN: {unknown_count})")

def main():
    mapping_path = Path("configs/label_mapping.yaml")
    metadata_dir = Path("data/metadata")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(mapping_path)
    reverse_lookup = build_reverse_lookup(mapping)
    logging.info(f"Built reverse lookup with {len(reverse_lookup)} alias entries.")

    for dataset in ["plantvillage", "plantdoc", "rice1426"]:
        jsonl_path = metadata_dir / f"{dataset}.jsonl"
        out_path = out_dir / f"{dataset}_normalized.jsonl"
        logging.info(f"Normalizing labels for {dataset}")
        process_file(jsonl_path, mapping, out_path, reverse_lookup)

if __name__ == "__main__":
    main()
