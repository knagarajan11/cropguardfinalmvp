import os
import shutil
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = "/workspace/data/raw/plantvillage"

TARGET_DIR = "/workspace/cropguard_eval_images1"

# EXACTLY 6 evaluation classes
TARGET_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Corn_maize___Common_rust",
    "Corn_maize___Northern_Leaf_Blight",
    "Corn_maize___healthy",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


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
print("Required images :", len(TARGET_CLASSES))
print()


# ============================================================
# CHECK SOURCE DIRECTORY
# ============================================================

if not os.path.isdir(SOURCE_DIR):
    raise FileNotFoundError(
        f"Source directory not found:\n{SOURCE_DIR}"
    )


# ============================================================
# CREATE TARGET DIRECTORY
# ============================================================

os.makedirs(TARGET_DIR, exist_ok=True)


# ============================================================
# FIND AVAILABLE CLASS DIRECTORIES
# ============================================================

print("=" * 70)
print("Finding PlantVillage class directories")
print("=" * 70)
print()

available_dirs = {}

source_root = Path(SOURCE_DIR)

for path in source_root.rglob("*"):

    if path.is_dir():

        available_dirs[path.name] = str(path)


print(
    "Available class directories found:",
    len(available_dirs)
)

print()


# ============================================================
# TARGET CLASS MATCHING
# ============================================================

print("=" * 70)
print("Target class matching")
print("=" * 70)
print()

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

        print(
            f"[FOUND] {target} -> {selected}"
        )

    else:

        print(
            f"[NOT FOUND] {target}"
        )


print()


# ============================================================
# CHECK THAT ALL 6 CLASSES WERE FOUND
# ============================================================

missing_classes = [
    target
    for target in TARGET_CLASSES
    if target not in matches
]

if missing_classes:

    print("=" * 70)
    print("ERROR - REQUIRED CLASSES NOT FOUND")
    print("=" * 70)
    print()

    for target in missing_classes:
        print("  [MISSING]", target)

    print()

    raise RuntimeError(
        "Not all 6 required PlantVillage classes were found."
    )


# ============================================================
# COPY EXACTLY ONE IMAGE FROM EACH CLASS
# ============================================================

print("=" * 70)
print("Copying exactly 6 evaluation images")
print("=" * 70)
print()

total_copied = 0

copied_files = []


for target_class in TARGET_CLASSES:

    source_class_dir = matches[target_class]

    print("-" * 70)
    print("Class :", target_class)
    print("Source:", source_class_dir)

    source_path = Path(source_class_dir)

    # --------------------------------------------------------
    # Find image files in this class
    # --------------------------------------------------------

    images = sorted(
        p
        for p in source_path.iterdir()
        if p.is_file()
        and p.suffix in IMAGE_EXTENSIONS
    )

    if not images:

        raise RuntimeError(
            f"No image files found in:\n{source_class_dir}"
        )

    # --------------------------------------------------------
    # Select EXACTLY ONE image
    # --------------------------------------------------------

    selected_image = images[0]

    # --------------------------------------------------------
    # Create class directory in target
    # --------------------------------------------------------

    target_class_dir = (
        Path(TARGET_DIR) / target_class
    )

    target_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use a standardized filename ending in _sample
    # --------------------------------------------------------

    destination_name = (
        f"{target_class}_sample"
        f"{selected_image.suffix.lower()}"
    )

    destination = (
        target_class_dir / destination_name
    )

    # --------------------------------------------------------
    # Copy
    # --------------------------------------------------------

    shutil.copy2(
        selected_image,
        destination
    )

    total_copied += 1

    copied_files.append(
        {
            "class": target_class,
            "source": str(selected_image),
            "destination": str(destination),
        }
    )

    print(
        "[COPIED]",
        selected_image.name
    )

    print(
        "[TARGET]",
        destination
    )

    print()


# ============================================================
# VERIFY EXACTLY 6 FILES WERE COPIED
# ============================================================

print("=" * 70)
print("VERIFYING EVALUATION SET")
print("=" * 70)
print()

evaluation_files = sorted(
    p
    for p in Path(TARGET_DIR).rglob("*")
    if p.is_file()
    and p.suffix in IMAGE_EXTENSIONS
)

print(
    "Images currently in target:",
    len(evaluation_files)
)

print()


# We expect exactly 6 images.
if len(evaluation_files) != 6:

    raise RuntimeError(
        f"Expected exactly 6 evaluation images, "
        f"but found {len(evaluation_files)}."
    )


# ============================================================
# PRINT FINAL FILE LIST
# ============================================================

print("=" * 70)
print("FINAL 6-IMAGE EVALUATION SET")
print("=" * 70)
print()

for number, image_path in enumerate(
    evaluation_files,
    start=1
):

    relative_path = image_path.relative_to(
        TARGET_DIR
    )

    print(
        f"{number}. {relative_path}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("COPY SUMMARY")
print("=" * 70)
print()

print("Source directory :", SOURCE_DIR)
print("Target directory :", TARGET_DIR)
print("Classes required :", len(TARGET_CLASSES))
print("Images copied    :", total_copied)
print("Images verified  :", len(evaluation_files))

print()

if total_copied == 6:

    print(
        "SUCCESS: Exactly 6 evaluation images are ready."
    )

else:

    print(
        "ERROR: Expected exactly 6 images."
    )

print()

print("=" * 70)
print("DONE")
print("=" * 70)
print()
