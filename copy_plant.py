import os
import shutil
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================
SOURCE_DIR = "/workspace/data/raw/plantvillage"
TARGET_DIR = "/workspace/cropguard_eval_images"

TARGET_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Corn_maize___Common_rust",
    "Corn_maize___Northern_Leaf_Blight",
    "Corn_maize___healthy",
]


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("CropGuard - PlantVillage Evaluation Image Copy")
print("=" * 70)
print()

print("Source directory:", SOURCE_DIR)
print("Target directory:", TARGET_DIR)
print()


# ============================================================
# CHECK SOURCE
# ============================================================

if not os.path.isdir(SOURCE_DIR):
    raise FileNotFoundError(
        f"Source directory not found:\n{SOURCE_DIR}"
    )

os.makedirs(TARGET_DIR, exist_ok=True)


# ============================================================
# FIND AVAILABLE CLASS DIRECTORIES
# ============================================================

print("=" * 70)
print("Finding PlantVillage class directories")
print("=" * 70)

available_dirs = {}

for path in Path(SOURCE_DIR).rglob("*"):

    if path.is_dir():

        available_dirs[path.name] = str(path)


print("Available class directories found:", len(available_dirs))
print()


# ============================================================
# TARGET CLASS MATCHING
# ============================================================

print("=" * 70)
print("Target class matching")
print("=" * 70)

matches = {}

for target in TARGET_CLASSES:

    # --------------------------------------------------------
    # Exact match first
    # --------------------------------------------------------

    if target in available_dirs:

        matches[target] = available_dirs[target]

        print(f"[FOUND] {target}")

        continue

    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    partial = [
        name
        for name in available_dirs
        if target.lower() in name.lower()
        or name.lower() in target.lower()
    ]

    if partial:

        selected = partial[0]

        matches[target] = available_dirs[selected]

        print(f"[FOUND] {target} -> {selected}")

    else:

        print(f"[NOT FOUND] {target}")


print()


# ============================================================
# COPY IMAGES
# ============================================================

print("=" * 70)
print("Copying evaluation images")
print("=" * 70)
print()

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

total_copied = 0


for target_class, source_class_dir in matches.items():

    print()
    print("-" * 70)
    print("Class:", target_class)
    print("Source:", source_class_dir)

    # Create class directory in target
    target_class_dir = Path(TARGET_DIR) / target_class

    target_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    copied_for_class = 0

    source_path = Path(source_class_dir)

    for image_path in sorted(source_path.rglob("*")):

        if not image_path.is_file():
            continue

        if image_path.suffix not in IMAGE_EXTENSIONS:
            continue

        destination = target_class_dir / image_path.name

        # Avoid overwriting an existing file with the same name
        if destination.exists():

            stem = image_path.stem
            suffix = image_path.suffix

            counter = 1

            while destination.exists():

                destination = (
                    target_class_dir
                    / f"{stem}_{counter}{suffix}"
                )

                counter += 1

        shutil.copy2(
            image_path,
            destination
        )

        copied_for_class += 1
        total_copied += 1

    print(
        "Images copied:",
        copied_for_class
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COPY SUMMARY")
print("=" * 70)
print()

print("Source directory :", SOURCE_DIR)
print("Target directory :", TARGET_DIR)
print("Classes requested:", len(TARGET_CLASSES))
print("Classes found    :", len(matches))
print("Total images     :", total_copied)

print()

print("Matched classes:")

for target_class, source_class_dir in matches.items():

    print(
        f"  {target_class:40s} -> {source_class_dir}"
    )

print()

missing = [
    target
    for target in TARGET_CLASSES
    if target not in matches
]

if missing:

    print("Missing classes:")

    for target in missing:

        print(
            f"  [NOT FOUND] {target}"
        )

else:

    print("All target classes were found.")

print()

print("=" * 70)
print("DONE")
print("=" * 70)
print()
