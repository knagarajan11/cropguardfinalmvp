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
    Downloads the PlantDoc dataset from HuggingFace.
    Uses geraldmc/plantdoc-full (parquet format, ~2,569 images, 28 classes).
    Fallback: Project-AgML/plant_doc_classification.
    """
    parser = argparse.ArgumentParser(description="Download PlantDoc dataset")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if files already exist")
    args = parser.parse_args()

    dest_dir = Path(os.getenv("PLANTDOC_DIR", "data/raw/plantdoc"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    force_download = args.force or os.getenv("FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes")

    # Check if already downloaded
    existing_images = list(dest_dir.rglob("*.jpg")) + list(dest_dir.rglob("*.jpeg")) + list(dest_dir.rglob("*.png"))
    if len(existing_images) > 100 and not force_download:
        logging.info(f"PlantDoc already downloaded ({len(existing_images)} images found in {dest_dir}). Skipping. (Use --force to re-download)")
        return

    logging.info(f"Downloading PlantDoc dataset to {dest_dir}")

    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("'datasets' library not found. Install it: pip install datasets")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")

    # Try multiple dataset sources in order of reliability
    dataset_candidates = [
        "geraldmc/plantdoc-full",
        "Project-AgML/plant_doc_classification",
        "LamTNguyen/PlantDoc",
        "agyaatcoder/PlantDoc",
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
        logging.error("Could not download PlantDoc from any source.")
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
    for key in ["class_label", "class_name", "label_name", "label", "labels", "category", "class", "disease"]:
        if key in features:
            label_key = key
            if hasattr(features[key], "names"):
                label_names = features[key].names
            break

    if label_key is None and not ("host" in features and ("disease" in features or "is_healthy" in features)):
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

            # Resolve class directory name preserving both crop and disease
            if "host" in sample and sample["host"]:
                host_str = str(sample["host"]).strip().replace(" ", "_")
                if sample.get("is_healthy"):
                    dis_str = "healthy"
                elif sample.get("disease"):
                    dis_str = str(sample["disease"]).strip().replace(" ", "_")
                elif sample.get("class_label"):
                    dis_str = str(sample["class_label"]).strip().replace(" ", "_")
                else:
                    dis_str = "unknown"
                class_dir_name = f"{host_str}___{dis_str}".replace("/", "_").replace("\\", "_")
            elif "class_label" in sample and sample["class_label"]:
                class_dir_name = str(sample["class_label"]).replace(" ", "_").replace("/", "_").replace("\\", "_")
            elif label_key is not None:
                label_val = sample[label_key]
                if label_names is not None and isinstance(label_val, int):
                    label_name = label_names[label_val]
                else:
                    label_name = str(label_val)
                class_dir_name = label_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            else:
                class_dir_name = "unknown"

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
