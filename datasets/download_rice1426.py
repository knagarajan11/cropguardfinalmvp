import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Downloads a rice leaf disease dataset from HuggingFace.
    Uses Project-AgML/rice_leaf_disease_classification (parquet, 3,829 images, 6 classes).
    Fallback: Project-AgML/rice_disease_classification_bangladesh,
              LeafNet75/Rice-Disease-Classification-Dataset.
    """
    parser = argparse.ArgumentParser(description="Download Rice dataset")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if files already exist")
    args = parser.parse_args()

    dest_dir = Path(os.getenv("RICE1426_DIR", "data/raw/rice1426"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    force_download = args.force or os.getenv("FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes")

    # Check if already downloaded
    existing_images = list(dest_dir.rglob("*.jpg")) + list(dest_dir.rglob("*.jpeg")) + list(dest_dir.rglob("*.png"))
    if len(existing_images) > 100 and not force_download:
        logging.info(f"Rice dataset already downloaded ({len(existing_images)} images found in {dest_dir}). Skipping. (Use --force to re-download)")
        return

    logging.info(f"Downloading Rice disease dataset to {dest_dir}")

    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("'datasets' library not found. Install it: pip install datasets")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")

    # Try multiple dataset sources in order of reliability
    dataset_candidates = [
        "Project-AgML/rice_leaf_disease_classification",
        "Project-AgML/rice_leaf_disease_classification_india",
        "Project-AgML/rice_disease_classification_bangladesh",
        "LeafNet75/Rice-Disease-Classification-Dataset",
    ]

    dataset = None
    for repo_id in dataset_candidates:
        try:
            logging.info(f"Trying HuggingFace dataset: {repo_id}...")
            dataset = load_dataset(repo_id, trust_remote_code=True, token=hf_token)
            logging.info(f"Successfully loaded: {repo_id}")
            break
        except Exception as e:
            logging.warning(f"  Failed to load {repo_id}: {e}")
            continue

    if dataset is None:
        logging.error("Could not download rice disease dataset from any source.")
        logging.error("Please manually download a rice disease dataset and place images in:")
        logging.error(f"  {dest_dir}/<class_name>/<image_files>")
        sys.exit(1)

    # Auto-detect features
    first_split = list(dataset.keys())[0]
    features = dataset[first_split].features
    logging.info(f"Dataset splits: {list(dataset.keys())}")
    logging.info(f"Dataset features: {list(features.keys())}")

    # Find image column
    image_key = None
    for key in ["image", "img", "pixel_values"]:
        if key in features:
            image_key = key
            break
    if image_key is None:
        logging.error(f"Could not find image column. Available: {list(features.keys())}")
        sys.exit(1)

    # Find label column
    label_key = None
    label_names = None
    for key in ["label", "labels", "class", "category", "disease", "class_name", "label_name"]:
        if key in features:
            label_key = key
            if hasattr(features[key], "names"):
                label_names = features[key].names
            break

    if label_key is None:
        logging.error(f"Could not find label column. Available: {list(features.keys())}")
        sys.exit(1)

    logging.info(f"Using image column: '{image_key}', label column: '{label_key}'")
    if label_names:
        logging.info(f"Number of classes: {len(label_names)}")

    total_saved = 0
    for split_name in dataset:
        split = dataset[split_name]
        logging.info(f"Processing split '{split_name}' with {len(split)} samples...")

        for i, sample in enumerate(split):
            image = sample[image_key]
            label_val = sample[label_key]

            # Resolve label name
            if label_names is not None and isinstance(label_val, int):
                label_name = label_names[label_val]
            else:
                label_name = str(label_val)

            # Create class directory
            clean_label = label_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            if not clean_label.lower().startswith("rice"):
                class_dir_name = f"Rice___{clean_label}"
            else:
                class_dir_name = clean_label
            class_dir = dest_dir / class_dir_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save image
            img_filename = f"{split_name}_{i:06d}.jpg"
            img_path = class_dir / img_filename

            if not img_path.exists():
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path, "JPEG", quality=95)

            total_saved += 1
            if total_saved % 500 == 0:
                logging.info(f"  Saved {total_saved} images so far...")

    logging.info(f"Download complete. Total images saved: {total_saved}")

    # Count images per class
    for class_dir in sorted(dest_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.*")))
            logging.info(f"  {class_dir.name}: {count} images")

if __name__ == "__main__":
    main()
