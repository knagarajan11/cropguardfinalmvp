from pathlib import Path
import shutil

# ============================================================
# CropGuard - Create Balanced 100 Image Evaluation Dataset
# 20 images per class
# Excludes the problematic 121st image
# Original dataset is NOT modified
# ============================================================

# ============================================================
# CropGuard - Create Balanced Evaluation Dataset
# ============================================================

SRC = Path(
    "/workspace/gsh-team03/images180/eval_cosmos3_180"
)

DST = Path(
    "/workspace/gsh-team03/images100/eval_nemotron_100"
)

EXCLUDE = Path(
    "Rice___Leaf_Blast/Rice___Leaf_Blast_001.jpg"
)

IMAGES_PER_CLASS = 20




EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CLASSES = [
    "Apple___Apple_scab",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Rice___Leaf_Blast",
    "Tomato___Septoria_leaf_spot",
]


def get_images(directory):
    return sorted(
        [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ],
        key=lambda p: p.name
    )


print()
print("=" * 80)
print("CropGuard - BALANCED 100 IMAGE EVALUATION DATASET")
print("=" * 80)

print(f"Source      : {SRC}")
print(f"Destination : {DST}")
print(f"Per class   : {IMAGES_PER_CLASS}")
print(f"Total       : {IMAGES_PER_CLASS * len(CLASSES)}")
print(f"Excluded    : {EXCLUDE}")
print()


# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

if not SRC.exists():
    raise RuntimeError(f"Source dataset does not exist: {SRC}")

if DST.exists():
    raise RuntimeError(
        f"\nDestination already exists:\n{DST}\n"
        "Delete/rename it manually if you want to recreate it."
    )


# ------------------------------------------------------------
# Verify original dataset
# ------------------------------------------------------------

print("Checking source dataset...")
print("-" * 80)

source_total = 0

for cls in CLASSES:

    class_dir = SRC / cls

    if not class_dir.exists():
        raise RuntimeError(
            f"Missing class directory: {class_dir}"
        )

    images = get_images(class_dir)

    # Remove the known problematic image from consideration
    images = [
        p for p in images
        if p.relative_to(SRC) != EXCLUDE
    ]

    print(f"{cls:35s}: {len(images):3d} available")

    if len(images) < IMAGES_PER_CLASS:
        raise RuntimeError(
            f"{cls} has only {len(images)} usable images; "
            f"need {IMAGES_PER_CLASS}"
        )

    source_total += len(images)


print()


# ------------------------------------------------------------
# Create destination
# ------------------------------------------------------------

DST.mkdir(parents=True, exist_ok=False)


# ------------------------------------------------------------
# Select exactly 20 images per class
# ------------------------------------------------------------

selected = []

print("Selecting images...")
print("-" * 80)

for cls in CLASSES:

    class_dir = SRC / cls

    images = get_images(class_dir)

    # Explicitly exclude Image 121
    images = [
        p for p in images
        if p.relative_to(SRC) != EXCLUDE
    ]

    # Deterministic selection:
    # first 20 alphabetically from each class
    chosen = images[:IMAGES_PER_CLASS]

    print(f"{cls:35s}: selecting {len(chosen)}")

    if len(chosen) != IMAGES_PER_CLASS:
        raise RuntimeError(
            f"Selection error for {cls}"
        )

    selected.extend(chosen)


# ------------------------------------------------------------
# Copy files
# ------------------------------------------------------------

print()
print("Copying files...")
print("-" * 80)

for src_file in selected:

    relative_path = src_file.relative_to(SRC)

    dst_file = DST / relative_path

    dst_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Preserve metadata
    shutil.copy2(src_file, dst_file)


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

print()
print("Verifying dataset...")
print("-" * 80)

total_created = 0

for cls in CLASSES:

    class_dir = DST / cls

    images = get_images(class_dir)

    count = len(images)

    print(
        f"{cls:35s}: {count:3d} images"
    )

    if count != IMAGES_PER_CLASS:
        raise RuntimeError(
            f"Verification failed for {cls}: "
            f"expected {IMAGES_PER_CLASS}, got {count}"
        )

    total_created += count


# ------------------------------------------------------------
# Verify excluded image is NOT present
# ------------------------------------------------------------

excluded_destination = DST / EXCLUDE

if excluded_destination.exists():

    raise RuntimeError(
        "ERROR: Excluded Image 121 exists in destination!"
    )


# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

expected_total = IMAGES_PER_CLASS * len(CLASSES)

print()
print("=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

print(f"Expected images : {expected_total}")
print(f"Created images  : {total_created}")
print(f"Classes         : {len(CLASSES)}")
print(f"Images/class    : {IMAGES_PER_CLASS}")
print()
print("Excluded image:")
print(f"  {EXCLUDE}")
print()

if total_created != expected_total:
    raise RuntimeError(
        f"Expected {expected_total}, created {total_created}"
    )

print("PASS")
print("Balanced 100-image evaluation dataset created successfully.")
print("=" * 80)
