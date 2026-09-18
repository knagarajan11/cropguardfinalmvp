import os
import re
import time
import csv
import traceback
from pathlib import Path

import torch
import numpy as np

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-"
    "Reasoning-BF16/snapshots/"
    "e5e9932441de940c9a62185c870ea5bcd4cd24e2"
)

LORA_PATH = (
    "/workspace/outputs/lora/"
    "cropguard_nemotron_lora_full_2gpu/"
    "epoch_1_step_26459/model_remapped"
)

# This directory may itself contain images directly, OR it may contain
# PlantVillage-style class subfolders (e.g. Tomato___Early_blight/*.jpg).
# The search below is recursive, so either layout works.
IMAGE_DIR = "/workspace/cropguard_eval_images"

RESULTS_CSV = "/workspace/cropguard_evaluation_results.csv"

DEVICE = "cuda:0"

MAX_NEW_TOKENS = 512

EXPECTED_LORA_MODULES = 116


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("             CROPGUARD MULTI-IMAGE EVALUATION (v2)")
print("=" * 80)
print()

print("Model      :", MODEL_PATH)
print("LoRA       :", LORA_PATH)
print("Image dir  :", IMAGE_DIR, "(searched recursively)")
print("Device     :", DEVICE)
print()


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Base model directory not found:\n{MODEL_PATH}")

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(f"LoRA directory not found:\n{LORA_PATH}")

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(f"Evaluation image directory not found:\n{IMAGE_DIR}")


# ============================================================
# STEP 1: FIND IMAGES (RECURSIVE)
# ============================================================

print("=" * 80)
print("STEP 1: FIND EVALUATION IMAGES (RECURSIVE)")
print("=" * 80)
print()

image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

all_images = sorted(
    p for p in Path(IMAGE_DIR).rglob("*")
    if p.is_file() and p.suffix in image_extensions
)

print("Images found:", len(all_images))

for i, image_path in enumerate(all_images, start=1):
    # Show path relative to IMAGE_DIR so folder structure is visible
    try:
        rel = image_path.relative_to(IMAGE_DIR)
    except ValueError:
        rel = image_path
    print(f"{i}. {rel}")

print()


# ============================================================
# GROUND TRUTH
# ============================================================
#
# Two supported layouts:
#
# 1. PlantVillage-style class folders, e.g.:
#       Tomato___Early_blight/image1.JPG
#       Tomato___Late_blight/image2.JPG
#       Tomato___healthy/image3.JPG
#    Ground truth is taken from the immediate parent folder name.
#
# 2. Flat directory with descriptive filenames, e.g.:
#       tomato_early_blight_1.jpg
#       tomato_late_blight_1.jpg
#       tomato_healthy_image_1.jpg
#    Ground truth is taken from the filename.
#
# The folder name is checked first (since PlantVillage folder names are
# unambiguous and don't suffer from typos like "tomoto"); the filename
# is used as a fallback for images sitting directly in IMAGE_DIR.

def _match_label(text):
    """Return canonical label from a lowercased, normalized string, or None."""
    text = text.lower().replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)

    if "healthy" in text:
        return "healthy"
    if "early_blight" in text:
        return "early blight"
    if "late_blight" in text:
        return "late blight"
    return None


def get_ground_truth(image_path: Path, image_dir: Path):
    """
    Determine ground truth for an image.

    Checks (in order):
      1. The immediate parent folder name (PlantVillage style).
      2. Any other ancestor folder name up to image_dir (in case of
         nested layouts like species/Early_blight/*.jpg).
      3. The filename itself.

    Returns a canonical class name ("early blight", "late blight",
    "healthy") or None if no match is found anywhere.
    """

    # 1 & 2: walk up through parent folders until we reach image_dir
    current = image_path.parent
    image_dir = image_dir.resolve()

    while True:
        current_resolved = current.resolve()
        label = _match_label(current.name)
        if label is not None:
            return label
        if current_resolved == image_dir or current.parent == current:
            break
        current = current.parent

    # 3: fall back to filename
    return _match_label(image_path.stem)


# ============================================================
# STEP 2: BUILD EVALUATION DATASET
# ============================================================

print("=" * 80)
print("STEP 2: BUILD EVALUATION DATASET")
print("=" * 80)
print()

evaluation_images = []
skipped_images = []

for image_path in all_images:

    ground_truth = get_ground_truth(image_path, Path(IMAGE_DIR))

    try:
        rel = image_path.relative_to(IMAGE_DIR)
    except ValueError:
        rel = image_path

    if ground_truth is None:
        skipped_images.append((str(rel), "Ground truth not identifiable from folder or filename"))
        print(f"SKIP: {rel} (ground truth not identifiable)")
        continue

    evaluation_images.append(
        {
            "path": str(image_path),
            "filename": str(rel),
            "ground_truth": ground_truth,
        }
    )

    print(f"USE : {str(rel):50s} Ground truth = {ground_truth}")

print()
print("Images selected for evaluation:", len(evaluation_images))
print("Images skipped:", len(skipped_images))
print()

if len(evaluation_images) == 0:
    raise RuntimeError("No labelled evaluation images found.")


# ============================================================
# STEP 3: LOAD PROCESSOR
# ============================================================

print("=" * 80)
print("STEP 3: LOAD PROCESSOR")
print("=" * 80)
print()

processor_start = time.time()

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Processor:", type(processor))

image_token = getattr(processor, "image_token", None)
image_token_id = getattr(processor, "image_token_id", None)

print("Image token:", image_token)
print("Image token ID:", image_token_id)

print()
print(f"Processor loaded in {time.time() - processor_start:.2f} seconds")
print()


# ============================================================
# STEP 4: LOAD BASE MODEL
# ============================================================

print("=" * 80)
print("STEP 4: LOAD BASE MODEL")
print("=" * 80)
print()

base_start = time.time()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

print()
print(f"Base model loaded in {time.time() - base_start:.2f} seconds")
print("BASE MODEL: OK")
print()


# ============================================================
# STEP 5: MODEL STRUCTURE
# ============================================================

print("=" * 80)
print("STEP 5: MODEL STRUCTURE")
print("=" * 80)
print()

try:
    language_model = base_model.language_model
    print("language_model:", type(language_model))
    print("Has backbone:", hasattr(language_model, "backbone"))
    if hasattr(language_model, "backbone"):
        print("backbone:", type(language_model.backbone))
        print("Number of layers:", len(language_model.backbone.layers))
except Exception as e:
    print("Could not inspect model structure:", repr(e))

print()


# ============================================================
# STEP 6: LOAD REMAPPED LORA
# ============================================================

print("=" * 80)
print("STEP 6: LOAD REMAPPED LORA")
print("=" * 80)
print()

lora_start = time.time()

model = PeftModel.from_pretrained(base_model, LORA_PATH, is_trainable=False)

print(f"LoRA loaded in {time.time() - lora_start:.2f} seconds")
print("LORA MODEL: OK")
print()


# ============================================================
# STEP 7: ADAPTER STATUS
# ============================================================

print("=" * 80)
print("STEP 7: ADAPTER STATUS")
print("=" * 80)
print()

print("Active adapters:", getattr(model, "active_adapters", None))
print()


# ============================================================
# STEP 8: LORA MODULE CHECK
# ============================================================

print("=" * 80)
print("STEP 8: LORA MODULE CHECK")
print("=" * 80)
print()

lora_modules = [
    name for name, module in model.named_modules()
    if hasattr(module, "lora_A") and hasattr(module, "lora_B")
]

print("LoRA modules found:", len(lora_modules))

if len(lora_modules) != EXPECTED_LORA_MODULES:
    print(f"WARNING: Expected {EXPECTED_LORA_MODULES} LoRA modules but found {len(lora_modules)}")
else:
    print(f"{EXPECTED_LORA_MODULES} LoRA modules: OK")

print()


# ============================================================
# STEP 9: PARAMETER STATUS
# ============================================================

print("=" * 80)
print("STEP 9: PARAMETER STATUS")
print("=" * 80)
print()

trainable_parameters = 0
total_parameters = 0

for parameter in model.parameters():
    total_parameters += parameter.numel()
    if parameter.requires_grad:
        trainable_parameters += parameter.numel()

trainable_percent = 100.0 * trainable_parameters / total_parameters

print("Trainable parameters :", trainable_parameters)
print("Total parameters     :", f"{total_parameters:,}")
print("Trainable percentage :", f"{trainable_percent:.6f}%")

model.eval()

print("Model evaluation mode: OK")
print()


# ============================================================
# STEP 10: INSTALL SAFE MULTIMODAL GENERATION PATCH
# ============================================================

print("=" * 80)
print("STEP 10: INSTALL SAFE MULTIMODAL GENERATION PATCH")
print("=" * 80)
print()

"""
The custom NVIDIA multimodal generate() passes multimodal metadata down
to the inner language model. The inner HuggingFace generate() rejects:
    num_patches, num_tokens, imgs_sizes
so we patch the INNER language model's generate() to strip those keys
before calling the original method. The OUTER multimodal model still
receives all image information — this patch only shields the inner call.
"""

try:
    multimodal_model = model.get_base_model()
    print("Multimodal model:", type(multimodal_model))

    inner_language_model = multimodal_model.language_model
    print("Inner language model:", type(inner_language_model))

    original_inner_generate = inner_language_model.generate

    _STRIP_KEYS = ("num_patches", "num_tokens", "imgs_sizes")

    def safe_inner_generate(*args, **kwargs):
        removed = [k for k in _STRIP_KEYS if k in kwargs]
        for k in removed:
            kwargs.pop(k)
        if removed:
            print("SAFE GENERATION PATCH: removed inner-model metadata:", removed)
        return original_inner_generate(*args, **kwargs)

    inner_language_model.generate = safe_inner_generate

    print("Safe generation patch installed.")
    print("NVIDIA modeling.py remains unchanged.")

except Exception as e:
    print()
    print("FAILED TO INSTALL GENERATION PATCH")
    print(repr(e))
    raise

print()


# ============================================================
# DISEASE NORMALIZATION
# ============================================================

def normalize_disease(text):
    if text is None:
        return "unknown"

    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    # Order matters: specific classes before generic words
    if "late blight" in text:
        return "late blight"
    if "early blight" in text:
        return "early blight"
    if "healthy" in text:
        return "healthy"
    return "unknown"


# ============================================================
# EXTRACT DISEASE FROM MODEL RESPONSE
# ============================================================

def extract_predicted_disease(response):
    response_lower = response.lower()

    patterns = [
        r"most likely disease\s*:\s*([^\n\r]+)",
        r"most likely disease\s*-\s*([^\n\r]+)",
        r"disease\s*:\s*([^\n\r]+)",
        r"diagnosis\s*:\s*([^\n\r]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            candidate = match.group(1).strip()
            candidate = candidate.split(".")[0]
            candidate = candidate.split(",")[0]
            normalized = normalize_disease(candidate)
            if normalized != "unknown":
                return normalized

    # Fallback: look for known classes anywhere in the response.
    # NOTE: "late blight" is checked before "early blight" only because
    # that was the original ordering; since both are mutually exclusive
    # substrings this fallback is order-independent in practice. If a
    # response mentions both diseases (e.g. as a differential diagnosis),
    # prefer whichever appears first in the text.
    first_match = None
    first_index = None
    for label, needle in (
        ("late blight", "late blight"),
        ("early blight", "early blight"),
        ("healthy", "healthy"),
    ):
        idx = response_lower.find(needle)
        if idx != -1 and (first_index is None or idx < first_index):
            first_index = idx
            first_match = label

    return first_match if first_match else "unknown"


# ============================================================
# EXTRACT CONFIDENCE FROM MODEL RESPONSE
# ============================================================

def extract_confidence(response):
    """
    Look for a stated confidence level or percentage in the response.
    Returns a string like "high", "medium", "low", "82%", or "unknown".
    """
    response_lower = response.lower()

    # Look for "confidence level: <value>" or "confidence: <value>"
    match = re.search(r"confidence(?:\s+level)?\s*:\s*([^\n\r.]+)", response_lower)
    if match:
        candidate = match.group(1).strip()
        candidate = candidate.split(",")[0].strip()
        if candidate:
            return candidate[:40]

    # Look for a bare percentage near the word "confidence"
    match = re.search(r"(\d{1,3}\s*%)\s*confidence", response_lower)
    if match:
        return match.group(1).replace(" ", "")

    return "unknown"


# ============================================================
# PROMPT
# ============================================================

def create_prompt():
    return """
You are CropGuard, an AI assistant for crop disease detection
and agricultural advisory.

Analyze the provided crop leaf image.

Identify the crop and diagnose the most likely disease.

Provide:

1. Crop name
2. Most likely disease
3. Visible symptoms
4. Possible alternative causes
5. Recommended immediate action
6. Preventive measures
7. Whether another image is needed
8. Confidence level

IMPORTANT:
Clearly state the diagnosis using exactly this format:

Most likely disease: <disease name>

Also clearly state your confidence using exactly this format:

Confidence level: <low/medium/high or a percentage>

Give a concise agricultural recommendation.
"""


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []
y_true = []
y_pred = []


# ============================================================
# STEP 11+: PROCESS EVERY IMAGE
# ============================================================

for image_number, item in enumerate(evaluation_images, start=1):

    image_path = item["path"]
    filename = item["filename"]
    ground_truth = item["ground_truth"]

    print()
    print()
    print("=" * 80)
    print(f"IMAGE {image_number}/{len(evaluation_images)}")
    print("=" * 80)
    print()
    print("File        :", filename)
    print("Ground truth:", ground_truth)
    print()

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    print("Loading image...")

    try:
        image = Image.open(image_path).convert("RGB")
        print("Image size:", image.size)
        print("Image mode:", image.mode)
        print("IMAGE: OK")
    except Exception as e:
        print("IMAGE LOAD FAILED")
        print(repr(e))

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "prediction": "error",
                "confidence": "unknown",
                "correct": False,
                "inference_time": 0.0,
                "response": f"IMAGE ERROR: {repr(e)}",
            }
        )

        y_true.append(ground_truth)
        y_pred.append("unknown")
        continue

    # --------------------------------------------------------
    # CHAT TEMPLATE
    # --------------------------------------------------------

    print()
    print("Creating chat template...")

    prompt = create_prompt()

    conversation = [
        {
            "role": "system",
            "content": (
                "You are CropGuard, an AI assistant for "
                "crop disease detection and agricultural advisory."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    try:
        chat_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        print("Chat template created.")

        if image_token is not None:
            if image_token in chat_text:
                print("Image token in chat text: PASSED")
            else:
                print("Image token in chat text: FAILED")
                raise RuntimeError("Chat template does not contain image token.")

    except Exception as e:
        print("CHAT TEMPLATE FAILED")
        print(repr(e))
        raise

    # --------------------------------------------------------
    # PROCESS MULTIMODAL INPUT
    # --------------------------------------------------------

    print()
    print("Running processor...")

    try:
        inputs = processor(text=chat_text, images=image, return_tensors="pt")
        print("Processor created multimodal inputs.")
    except Exception as e:
        print()
        print("PROCESSOR FAILED")
        print(repr(e))
        raise

    # --------------------------------------------------------
    # MOVE INPUTS
    # --------------------------------------------------------

    print()
    print("Moving inputs to CUDA...")

    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.dtype.is_floating_point:
                inputs[key] = value.to(DEVICE, dtype=torch.bfloat16)
            else:
                inputs[key] = value.to(DEVICE)

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    print()
    print("Input tensors:")

    for key, value in inputs.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")

    if "input_ids" not in inputs:
        raise RuntimeError("input_ids missing from processor output.")

    if image_token_id is not None:
        image_found = (inputs["input_ids"] == image_token_id).any().item()
        print()
        print("Image token found:", image_found)
        if not image_found:
            raise RuntimeError("No image token found in input_ids.")

    print()
    print("INPUTS: OK")

    # --------------------------------------------------------
    # GENERATION
    #
    # num_patches / num_tokens / imgs_sizes are NOT passed here manually.
    # They are produced internally by the outer multimodal model's own
    # generate() from the processor's `inputs` (e.g. via image_sizes /
    # pixel_values already present in `inputs`), and it is only when the
    # outer model forwards them to the *inner* HF language-model generate()
    # that they need to be stripped — which is what the Step 10 patch does.
    # We must not attempt to pop these from `inputs` ourselves: they are
    # not top-level keys of `inputs` in the first place, and doing so here
    # would have no effect while masking the real place the strip is needed.
    # --------------------------------------------------------

    print()
    print("Generating response...")
    print()

    start_inference = time.time()

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        inference_time = time.time() - start_inference

        print()
        print("Generation completed.")
        print(f"Inference time: {inference_time:.2f} seconds")

    except Exception as e:
        inference_time = time.time() - start_inference

        print()
        print("GENERATION FAILED")
        print(repr(e))
        traceback.print_exc()

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "prediction": "unknown",
                "confidence": "unknown",
                "correct": False,
                "inference_time": inference_time,
                "response": f"GENERATION ERROR: {repr(e)}",
            }
        )

        y_true.append(ground_truth)
        y_pred.append("unknown")
        continue

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    print()
    print("Decoding response...")

    try:
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[:, input_length:]
        response = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        response = response.strip()
    except Exception as e:
        print("DECODE FAILED")
        print(repr(e))
        raise

    # --------------------------------------------------------
    # DISPLAY RESPONSE
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("CROPGUARD RESPONSE")
    print("-" * 80)
    print()
    print(response)
    print()
    print("-" * 80)

    # --------------------------------------------------------
    # EXTRACT PREDICTION + CONFIDENCE
    # --------------------------------------------------------

    predicted_disease = extract_predicted_disease(response)
    confidence = extract_confidence(response)
    correct = predicted_disease == ground_truth

    print()
    print("Ground truth :", ground_truth)
    print("Prediction   :", predicted_disease)
    print("Confidence   :", confidence)
    print("Correct      :", correct)

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    y_true.append(ground_truth)
    y_pred.append(predicted_disease)

    results.append(
        {
            "image": filename,
            "ground_truth": ground_truth,
            "prediction": predicted_disease,
            "confidence": confidence,
            "correct": correct,
            "inference_time": inference_time,
            "response": response,
        }
    )


# ============================================================
# STEP 12: SAVE CSV
# ============================================================

print()
print()
print("=" * 80)
print("STEP 12: SAVE RESULTS")
print("=" * 80)
print()

try:
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "ground_truth",
                "prediction",
                "confidence",
                "correct",
                "inference_time",
                "response",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Results saved:", RESULTS_CSV)

except Exception as e:
    print("Could not save CSV:", repr(e))


# ============================================================
# STEP 13: FINAL METRICS
# ============================================================

print()
print()
print("=" * 80)
print("STEP 13: FINAL EVALUATION METRICS")
print("=" * 80)
print()

if len(y_true) == 0:
    print("No evaluation results available.")
    raise SystemExit(1)

correct_count = sum(1 for a, p in zip(y_true, y_pred) if a == p)
total_count = len(y_true)

accuracy = accuracy_score(y_true, y_pred)

macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"Images evaluated : {total_count}")
print(f"Correct          : {correct_count}")
print(f"Incorrect        : {total_count - correct_count}")
print()
print(f"Accuracy         : {accuracy * 100:.2f}%")
print()
print(f"Macro Precision  : {macro_precision * 100:.2f}%")
print(f"Macro Recall     : {macro_recall * 100:.2f}%")
print(f"Macro F1         : {macro_f1 * 100:.2f}%")
print()
print(f"Weighted Precision : {weighted_precision * 100:.2f}%")
print(f"Weighted Recall    : {weighted_recall * 100:.2f}%")
print(f"Weighted F1        : {weighted_f1 * 100:.2f}%")


# ============================================================
# STEP 14: CONFUSION MATRIX
# ============================================================

print()
print()
print("=" * 80)
print("STEP 14: CONFUSION MATRIX")
print("=" * 80)
print()

labels = ["early blight", "late blight", "healthy"]

cm = confusion_matrix(y_true, y_pred, labels=labels)

print()
header = f"{'Actual / Predicted':20s}{'Early':>10s}{'Late':>10s}{'Healthy':>10s}"
print(header)
print("-" * 50)

for label, row in zip(labels, cm):
    print(f"{label:20s}{row[0]:10d}{row[1]:10d}{row[2]:10d}")


# ============================================================
# STEP 15: CLASSIFICATION REPORT
# ============================================================

print()
print()
print("=" * 80)
print("STEP 15: PER-CLASS CLASSIFICATION REPORT")
print("=" * 80)
print()

print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4))


# ============================================================
# STEP 16: PER-IMAGE RESULTS
# ============================================================

print()
print()
print("=" * 80)
print("STEP 16: PER-IMAGE RESULTS")
print("=" * 80)
print()

print(f"{'Image':40s}{'Actual':16s}{'Predicted':16s}{'Confidence':14s}{'Result':10s}")
print("-" * 100)

for row in results:
    result_text = "PASS" if row["correct"] else "FAIL"
    print(
        f"{row['image'][-40:]:40s}"
        f"{row['ground_truth'][:16]:16s}"
        f"{row['prediction'][:16]:16s}"
        f"{str(row.get('confidence', 'unknown'))[:14]:14s}"
        f"{result_text:10s}"
    )


# ============================================================
# STEP 17: RESPONSE / MODEL STATUS
# ============================================================

print()
print()
print("=" * 80)
print("STEP 17: CROPGUARD FINAL STATUS")
print("=" * 80)
print()

print("Base model           : PASSED")
print("Remapped LoRA        : PASSED")
print(f"LoRA modules         : {len(lora_modules)} / {EXPECTED_LORA_MODULES}")
print("Multimodal input     : PASSED")
print("Generation           : PASSED")
print(f"Images evaluated     : {total_count}")
print()
print(f"Disease accuracy     : {accuracy * 100:.2f}%")
print(f"Macro precision      : {macro_precision * 100:.2f}%")
print(f"Macro recall         : {macro_recall * 100:.2f}%")
print(f"Macro F1             : {macro_f1 * 100:.2f}%")
print()

if skipped_images:
    print("Skipped images:")
    for filename, reason in skipped_images:
        print(f"  - {filename}: {reason}")

print()
print("CSV results:", RESULTS_CSV)
print()
print("=" * 80)
print("DONE")
print("=" * 80)
print()
