"""
Debug script to trace why normalize_labels.py produces so many UNKNOWNs.
Run from cropguard_phase2/ directory after data has been downloaded.

Usage:
    python datasets/debug_normalization.py
"""
import json
import yaml
import sys
from pathlib import Path
from collections import defaultdict

# Add current folder to sys.path to import local normalize_labels without conflicting with HuggingFace 'datasets'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_labels import load_mapping, build_reverse_lookup, normalize_label

def main():
    mapping_path = Path("configs/label_mapping.yaml")
    if not mapping_path.exists():
        print(f"ERROR: {mapping_path} not found. Run from cropguard_phase2/ directory.")
        sys.exit(1)

    mapping = load_mapping(mapping_path)
    reverse_lookup = build_reverse_lookup(mapping)

    # ── 1. Scan actual raw directories ───────────────────────────────────────
    datasets_to_check = {
        "plantvillage": Path("data/metadata/plantvillage.jsonl"),
        "plantdoc":     Path("data/metadata/plantdoc.jsonl"),
        "rice1426":     Path("data/metadata/rice1426.jsonl"),
    }

    for ds_name, jsonl_path in datasets_to_check.items():
        if not jsonl_path.exists():
            print(f"\n[{ds_name}] metadata not found at {jsonl_path}; skipping.")
            continue

        print(f"\n{'='*60}")
        print(f" Dataset: {ds_name}")
        print(f"{'='*60}")

        unknown_entries = []
        mapped_entries  = defaultdict(list)

        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                raw_crop    = r['crop']
                raw_disease = r['disease']
                orig_label  = r.get('original_label', '')

                norm_crop, norm_disease = normalize_label(raw_crop, raw_disease, mapping, reverse_lookup=reverse_lookup)

                if norm_disease == "UNKNOWN":
                    unknown_entries.append({
                        "crop": raw_crop,
                        "disease": raw_disease,
                        "original_label": orig_label,
                    })
                else:
                    mapped_entries[norm_disease].append(raw_disease)

        # Unique UNKNOWN inputs
        unique_unknowns = {}
        for e in unknown_entries:
            key = (e['crop'], e['disease'])
            if key not in unique_unknowns:
                unique_unknowns[key] = e['original_label']

        print(f"\n[UNKNOWN] {len(unknown_entries)} records ({len(unique_unknowns)} unique combos):")
        for (crop, disease), orig in sorted(unique_unknowns.items())[:40]:
            crop_lower = crop.lower().strip()
            in_mapping = crop_lower in mapping
            disease_clean = disease.lower().replace("-","_").replace(" ","_").strip(" _")
            print(f"  crop={repr(crop):<20} disease={repr(disease):<35} orig={repr(orig):<50} "
                  f"crop_in_yaml={in_mapping}  disease_clean={repr(disease_clean)}")

        print(f"\n[MAPPED]  {sum(len(v) for v in mapped_entries.values())} records across {len(mapped_entries)} diseases:")
        for disease, raws in sorted(mapped_entries.items()):
            unique_raws = sorted(set(raws))[:3]
            print(f"  {disease:<35} <- {unique_raws}")

    # ── 2. Quick unit-test of normalize_label with known inputs ──────────────
    print(f"\n{'='*60}")
    print(" normalize_label unit tests")
    print(f"{'='*60}")
    test_cases = [
        # (crop_in, disease_in, expected_disease)
        ("Tomato",       "Early_blight",        "early_blight"),
        ("Tomato",       "early blight",         "early_blight"),
        ("Tomato",       "leaf_early_blight",    "early_blight"),
        ("Tomato",       "Tomato_Early_blight",  "early_blight"),
        ("Pepper",       "Bacterial_spot",       "bacterial_spot"),
        ("Pepper,_bell", "Bacterial_spot",       "bacterial_spot"),
        ("Corn",         "Common_rust_",         "common_rust"),
        ("Corn_(maize)", "Common_rust",          "common_rust"),
        ("Corn",         "Northern_Leaf_Blight", "northern_leaf_blight"),
        ("Apple",        "Apple_scab",           "apple_scab"),
        ("Tomato",       "healthy",              "healthy"),
        # NOTE: 'leaf' alone is handled by prepare_*.py before calling normalize_label.
        # When it reaches here without filtering, the fallback returns 'leaf' (not 'healthy').
        ("Tomato",       "leaf",                 "leaf"),
    ]
    all_pass = True
    for crop, disease, expected in test_cases:
        _, got = normalize_label(crop, disease, mapping)
        status = "PASS" if got == expected else "FAIL"
        if got != expected:
            all_pass = False
        print(f"  [{status}] normalize_label({repr(crop):<20}, {repr(disease):<30}) -> {repr(got)}  (expected {repr(expected)})")

    print()
    if all_pass:
        print("All unit tests PASSED.")
    else:
        print("Some unit tests FAILED -- bugs found in normalize_label or label_mapping.yaml.")

if __name__ == "__main__":
    main()
