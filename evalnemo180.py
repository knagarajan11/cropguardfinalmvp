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

# EXACT 6-image evaluation directory


IMAGE_DIR = "/workspace/gsh-team03/CropGuard/images180/eval_cosmos3_180"
RESULTS_CSV = "/workspace/cropguard_nemotron_180_results.csv"
METRICS_CSV = "/workspace/cropguard_nemotron_180_metrics.csv"

DEVICE = "cuda:0"

MAX_NEW_TOKENS = 512

EXPECTED_LORA_MODULES = 116

REASONING_END_MARKERS = [
    "</think>",
    "</thinking>",
    "final answer:",
]


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("       CROPGUARD NEMOTRON OMNI BASE vs BASE+LoRA")
print("                 180-IMAGE EVALUATION")
print("=" * 80)
print()

print("Model      :", MODEL_PATH)
print("LoRA       :", LORA_PATH)
print("Image dir  :", IMAGE_DIR)
print("Device     :", DEVICE)
print("Max tokens :", MAX_NEW_TOKENS)
print(
    "Metrics    : Accuracy, Macro/Weighted Precision, "
    "Recall, F1"
)
print()

print(
    "Base and Base+LoRA will use the exact same processed "
    "image inputs and generation settings."
)

print()


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Base model directory not found:\n{MODEL_PATH}"
    )

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(
        f"LoRA directory not found:\n{LORA_PATH}"
    )

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"Evaluation image directory not found:\n{IMAGE_DIR}"
    )


# ============================================================
# STEP 1: FIND IMAGES
# ============================================================

print("=" * 80)
print("STEP 1: FIND 180 EVALUATION IMAGES")
print("=" * 80)
print()

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

all_images = sorted(
    p
    for p in Path(IMAGE_DIR).rglob("*")
    if p.is_file()
    and p.suffix in image_extensions
)

print("Images found:", len(all_images))
print()

for i, image_path in enumerate(
    all_images,
    start=1
):
    try:
        rel = image_path.relative_to(
            IMAGE_DIR
        )
    except ValueError:
        rel = image_path

    print(f"{i}. {rel}")

print()

if len(all_images) != 180:
    raise RuntimeError(
        f"Expected exactly 180 evaluation images, "
        f"but found {len(all_images)}."
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def match_label(text):

    text = text.lower()

    text = text.replace("-", "_")

    text = re.sub(
        r"[^a-z0-9_]+",
        "_",
        text
    )

    # Six evaluation classes
    if "apple_scab" in text or "apple_apple_scab" in text:
        return "apple scab"

    if "common_rust" in text or "corn_common_rust" in text:
        return "common rust"

    if (
        "northern_leaf_blight" in text
        or "corn_northern_leaf_blight" in text
    ):
        return "northern leaf blight"

    if "black_rot" in text or "grape_black_rot" in text:
        return "black rot"

    if (
        "septoria_leaf_spot" in text
        or "tomato_septoria_leaf_spot" in text
    ):
        return "septoria leaf spot"

    if "leaf_blast" in text or "rice_leaf_blast" in text:
        return "leaf blast"

    return None


def get_ground_truth(
    image_path: Path,
    image_dir: Path
):

    current = image_path.parent

    image_dir = image_dir.resolve()

    while True:

        current_resolved = current.resolve()

        label = match_label(
            current.name
        )

        if label is not None:
            return label

        if (
            current_resolved == image_dir
            or current.parent == current
        ):
            break

        current = current.parent

    return match_label(
        image_path.stem
    )


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

    ground_truth = get_ground_truth(
        image_path,
        Path(IMAGE_DIR)
    )

    try:
        rel = image_path.relative_to(
            IMAGE_DIR
        )
    except ValueError:
        rel = image_path

    if ground_truth is None:

        skipped_images.append(
            (
                str(rel),
                "Ground truth not identifiable"
            )
        )

        print(
            f"SKIP: {rel}"
        )

        continue

    evaluation_images.append(
        {
            "path": str(image_path),
            "filename": str(rel),
            "ground_truth": ground_truth,
        }
    )

    print(
        f"USE : {str(rel):60s} "
        f"Ground truth = {ground_truth}"
    )

print()

print(
    "Images selected for evaluation:",
    len(evaluation_images)
)

print(
    "Images skipped:",
    len(skipped_images)
)

print()

if len(evaluation_images) != 180:

    raise RuntimeError(
        "Expected exactly 180 labelled evaluation images."
    )

# Require exactly 30 images for each of the six classes.
EXPECTED_CLASS_COUNTS = {
    "apple scab": 30,
    "common rust": 30,
    "northern leaf blight": 30,
    "black rot": 30,
    "septoria leaf spot": 30,
    "leaf blast": 30,
}

actual_class_counts = {}
for item in evaluation_images:
    label = item["ground_truth"]
    actual_class_counts[label] = actual_class_counts.get(label, 0) + 1

print("CLASS COUNTS:")
for label, expected_count in EXPECTED_CLASS_COUNTS.items():
    actual_count = actual_class_counts.get(label, 0)
    print(f"  {label:25s}: {actual_count}")

if actual_class_counts != EXPECTED_CLASS_COUNTS:
    raise RuntimeError(
        "Evaluation dataset is not exactly balanced at 30 images per class. "
        f"Actual counts: {actual_class_counts}"
    )

print("180-image dataset balance: OK (6 classes x 30 images)")


# ============================================================
# STEP 3: LOAD PROCESSOR
# ============================================================

print("=" * 80)
print("STEP 3: LOAD PROCESSOR")
print("=" * 80)
print()

processor_start = time.time()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print(
    "Processor:",
    type(processor)
)

image_token = getattr(
    processor,
    "image_token",
    None
)

image_token_id = getattr(
    processor,
    "image_token_id",
    None
)

print(
    "Image token:",
    image_token
)

print(
    "Image token ID:",
    image_token_id
)

print()

print(
    f"Processor loaded in "
    f"{time.time() - processor_start:.2f} seconds"
)

print()


# ============================================================
# STEP 4: LOAD BASE MODEL
# ============================================================

print("=" * 80)
print("STEP 4: LOAD NEMOTRON OMNI BASE MODEL")
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

print(
    f"Base model loaded in "
    f"{time.time() - base_start:.2f} seconds"
)

print(
    "BASE MODEL: OK"
)

print()


# ============================================================
# STEP 5: LOAD LoRA
# ============================================================

print("=" * 80)
print("STEP 5: LOAD LoRA")
print("=" * 80)
print()

lora_start = time.time()

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False
)

print(
    f"LoRA loaded in "
    f"{time.time() - lora_start:.2f} seconds"
)

print(
    "LORA MODEL: OK"
)

print()

print(
    "Active adapters:",
    getattr(
        model,
        "active_adapters",
        None
    )
)

print()


# ============================================================
# CHECK LoRA MODULES
# ============================================================

lora_modules = [
    name
    for name, module in model.named_modules()
    if hasattr(module, "lora_A")
    and hasattr(module, "lora_B")
]

print(
    "LoRA modules found:",
    len(lora_modules)
)

if len(lora_modules) != EXPECTED_LORA_MODULES:

    print(
        f"WARNING: Expected "
        f"{EXPECTED_LORA_MODULES} LoRA modules "
        f"but found {len(lora_modules)}"
    )

else:

    print(
        f"{EXPECTED_LORA_MODULES} LoRA modules: OK"
    )

print()


# ============================================================
# PARAMETER INFORMATION
# ============================================================

trainable_parameters = 0

total_parameters = 0

for parameter in model.parameters():

    total_parameters += parameter.numel()

    if parameter.requires_grad:
        trainable_parameters += parameter.numel()

print(
    "Trainable parameters :",
    trainable_parameters
)

print(
    "Total parameters     :",
    f"{total_parameters:,}"
)

print(
    "Trainable percentage :",
    f"{100.0 * trainable_parameters / total_parameters:.6f}%"
)

model.eval()

print(
    "Model evaluation mode: OK"
)

print()


# ============================================================
# STEP 6: SAFE MULTIMODAL GENERATION PATCH
# ============================================================

print("=" * 80)
print("STEP 6: INSTALL SAFE MULTIMODAL GENERATION PATCH")
print("=" * 80)
print()

try:

    multimodal_model = model.get_base_model()

    print(
        "Multimodal model:",
        type(multimodal_model)
    )

    inner_language_model = (
        multimodal_model.language_model
    )

    print(
        "Inner language model:",
        type(inner_language_model)
    )

    original_inner_generate = (
        inner_language_model.generate
    )

    STRIP_KEYS = (
        "num_patches",
        "num_tokens",
        "imgs_sizes",
    )

    def safe_inner_generate(
        *args,
        **kwargs
    ):

        removed = [
            key
            for key in STRIP_KEYS
            if key in kwargs
        ]

        for key in removed:
            kwargs.pop(key)

        if removed:

            print(
                "SAFE GENERATION PATCH: "
                "removed inner-model metadata:",
                removed
            )

        return original_inner_generate(
            *args,
            **kwargs
        )

    inner_language_model.generate = (
        safe_inner_generate
    )

    print(
        "Safe generation patch installed."
    )

    print(
        "NVIDIA modeling.py remains unchanged."
    )

except Exception as e:

    print()
    print(
        "FAILED TO INSTALL GENERATION PATCH"
    )

    print(
        repr(e)
    )

    raise


print()


# ============================================================
# DISEASE NORMALIZATION
# ============================================================

def normalize_disease(text):

    if text is None:
        return "unknown"

    text = text.lower().strip()

    text = (
        text
        .replace("_", " ")
        .replace("-", " ")
    )

    text = re.sub(
        r"\\s+",
        " ",
        text
    )

    if "apple scab" in text:
        return "apple scab"

    if "common rust" in text:
        return "common rust"

    if "northern leaf blight" in text:
        return "northern leaf blight"

    if "black rot" in text:
        return "black rot"

    if "septoria leaf spot" in text:
        return "septoria leaf spot"

    if "leaf blast" in text:
        return "leaf blast"

    return "unknown"


# ============================================================
# ISOLATE FINAL ANSWER
# ============================================================

def isolate_final_answer(response):

    response_lower = response.lower()

    last_cut = -1

    for marker in REASONING_END_MARKERS:

        idx = response_lower.rfind(
            marker
        )

        if idx != -1:

            cut_point = (
                idx + len(marker)
            )

            if cut_point > last_cut:
                last_cut = cut_point

    if last_cut == -1:
        return response

    return response[last_cut:].strip()


# ============================================================
# EXTRACT DISEASE
# ============================================================

def extract_predicted_disease(
    response,
    truncated=False
):

    response_lower = response.lower()

    patterns = [

        r"most likely disease\s*:\s*([^\n\r]+)",

        r"most likely disease\s*-\s*([^\n\r]+)",

        r"disease\s*:\s*([^\n\r]+)",

        r"diagnosis\s*:\s*([^\n\r]+)",

    ]

    for pattern in patterns:

        matches = list(
            re.finditer(
                pattern,
                response_lower
            )
        )

        for match in reversed(matches):

            candidate = (
                match.group(1)
                .strip()
            )

            candidate = (
                candidate
                .split(".")[0]
            )

            candidate = (
                candidate
                .split(",")[0]
            )

            normalized = normalize_disease(
                candidate
            )

            if normalized != "unknown":

                return normalized

    if truncated:
        return "incomplete"

    last_match = None

    last_index = -1

    for label, needle in (

        (
            "apple scab",
            "apple scab"
        ),

        (
            "common rust",
            "common rust"
        ),

        (
            "northern leaf blight",
            "northern leaf blight"
        ),

        (
            "black rot",
            "black rot"
        ),

        (
            "septoria leaf spot",
            "septoria leaf spot"
        ),

        (
            "leaf blast",
            "leaf blast"
        ),

    ):

        idx = response_lower.rfind(
            needle
        )

        if idx != -1 and idx > last_index:

            last_index = idx

            last_match = label

    return (
        last_match
        if last_match
        else "unknown"
    )


# ============================================================
# EXTRACT CONFIDENCE
# ============================================================

def extract_confidence(response):

    response_lower = response.lower()

    def clean_candidate(text):

        return (
            text
            .strip(" *_\"'")
            .strip()
        )

    matches = list(
        re.finditer(
            r"confidence(?:\s+level)?\s*:\s*([^\n\r.]+)",
            response_lower
        )
    )

    if matches:

        candidate = (
            matches[-1]
            .group(1)
            .strip()
        )

        candidate = (
            candidate
            .split(",")[0]
        )

        candidate = clean_candidate(
            candidate
        )

        if candidate:

            return candidate[:40]

    matches = list(
        re.finditer(
            r"(\d{1,3}\s*%)\s*confidence",
            response_lower
        )
    )

    if matches:

        return (
            matches[-1]
            .group(1)
            .replace(" ", "")
        )

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

The possible disease classes for this evaluation are:

1. early blight
2. late blight
3. common rust
4. northern leaf blight
5. healthy

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

Use one of the six disease classes listed above.

Also clearly state your confidence using exactly this format:

Confidence level: <low/medium/high or a percentage>

Give a concise agricultural recommendation.
"""


# ============================================================
# GENERATION HELPER
# ============================================================

def run_generation(
    inputs,
    label
):

    start_inference = time.time()

    try:

        with torch.inference_mode():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        inference_time = (
            time.time()
            - start_inference
        )

        input_length = (
            inputs["input_ids"]
            .shape[1]
        )

        generated_tokens = (
            output_ids[
                :,
                input_length:
            ]
        )

        truncated = (
            generated_tokens.shape[1]
            >= MAX_NEW_TOKENS
        )

        response = (
            processor
            .batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]
        )

        response = response.strip()

        print(
            f"[{label}] Generation completed "
            f"in {inference_time:.2f}s "
            f"({generated_tokens.shape[1]} tokens"
            + (
                ", TRUNCATED"
                if truncated
                else ""
            )
            + ")"
        )

        return (
            response,
            inference_time,
            truncated,
            None
        )

    except Exception as e:

        inference_time = (
            time.time()
            - start_inference
        )

        print(
            f"[{label}] GENERATION FAILED:",
            repr(e)
        )

        traceback.print_exc()

        return (
            "",
            inference_time,
            False,
            repr(e)
        )


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []

y_true = []

y_pred_base = []

y_pred_lora = []


# ============================================================
# STEP 7: PROCESS EVERY IMAGE
# ============================================================

for image_number, item in enumerate(
    evaluation_images,
    start=1
):

    image_path = item["path"]

    filename = item["filename"]

    ground_truth = item["ground_truth"]

    print()
    print()
    print("=" * 80)

    print(
        f"IMAGE {image_number}/"
        f"{len(evaluation_images)}"
    )

    print("=" * 80)

    print()

    print(
        "File        :",
        filename
    )

    print(
        "Ground truth:",
        ground_truth
    )

    print()

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    print(
        "Loading image..."
    )

    try:

        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )

        print(
            "Image size:",
            image.size
        )

        print(
            "Image mode:",
            image.mode
        )

    except Exception as e:

        print(
            "IMAGE LOAD FAILED:",
            repr(e)
        )

        error_row = {

            "image": filename,

            "ground_truth": ground_truth,

            "base_prediction": "error",

            "base_confidence": "unknown",

            "base_truncated": False,

            "base_correct": False,

            "base_inference_time": 0.0,

            "base_response":
                f"IMAGE ERROR: {repr(e)}",

            "lora_prediction": "error",

            "lora_confidence": "unknown",

            "lora_truncated": False,

            "lora_correct": False,

            "lora_inference_time": 0.0,

            "lora_response":
                f"IMAGE ERROR: {repr(e)}",

            "verdict":
                "Image load error",
        }

        results.append(
            error_row
        )

        y_true.append(
            ground_truth
        )

        y_pred_base.append(
            "unknown"
        )

        y_pred_lora.append(
            "unknown"
        )

        continue

    # --------------------------------------------------------
    # CHAT TEMPLATE
    # --------------------------------------------------------

    print()

    print(
        "Creating chat template and "
        "processing inputs..."
    )

    prompt = create_prompt()

    conversation = [

        {
            "role": "system",

            "content": (
                "You are CropGuard, an AI "
                "assistant for crop disease "
                "detection and agricultural advisory."
            ),
        },

        {
            "role": "user",

            "content": [

                {
                    "type": "image"
                },

                {
                    "type": "text",
                    "text": prompt
                },

            ],
        },

    ]

    try:

        chat_text = (
            processor
            .apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

        if (
            image_token is not None
            and image_token not in chat_text
        ):

            raise RuntimeError(
                "Chat template does not "
                "contain image token."
            )

        inputs = processor(
            text=chat_text,
            images=image,
            return_tensors="pt"
        )

    except Exception as e:

        print(
            "CHAT TEMPLATE / PROCESSOR FAILED:",
            repr(e)
        )

        raise

    # --------------------------------------------------------
    # MOVE INPUTS TO GPU
    # --------------------------------------------------------

    for key, value in inputs.items():

        if torch.is_tensor(value):

            if value.dtype.is_floating_point:

                inputs[key] = value.to(
                    DEVICE,
                    dtype=torch.bfloat16
                )

            else:

                inputs[key] = value.to(
                    DEVICE
                )

    if "input_ids" not in inputs:

        raise RuntimeError(
            "input_ids missing from processor output."
        )

    if image_token_id is not None:

        image_found = (
            inputs["input_ids"]
            == image_token_id
        ).any().item()

        if not image_found:

            raise RuntimeError(
                "No image token found in input_ids."
            )

    print(
        "INPUTS: OK"
    )

    # --------------------------------------------------------
    # PASS 1: BASE
    # --------------------------------------------------------

    print()

    print("-" * 80)

    print(
        "PASS 1: BASE MODEL "
        "(LoRA disabled)"
    )

    print("-" * 80)

    with model.disable_adapter():

        (
            base_response,
            base_inference_time,
            base_truncated,
            base_error,
        ) = run_generation(
            inputs,
            "BASE"
        )

    if base_error is not None:

        base_predicted_disease = "unknown"

        base_confidence = "unknown"

        base_response_stored = (
            f"GENERATION ERROR: {base_error}"
        )

    else:

        base_final_text = (
            isolate_final_answer(
                base_response
            )
        )

        base_predicted_disease = (
            extract_predicted_disease(
                base_final_text,
                truncated=base_truncated
            )
        )

        base_confidence = (
            extract_confidence(
                base_final_text
            )
        )

        base_response_stored = (
            base_response
        )

    base_correct = (
        base_predicted_disease
        == ground_truth
    )

    print(
        "BASE prediction :",
        base_predicted_disease
    )

    print(
        "BASE confidence :",
        base_confidence
    )

    print(
        "BASE correct    :",
        base_correct
    )

    # --------------------------------------------------------
    # PASS 2: LoRA
    # --------------------------------------------------------

    print()

    print("-" * 80)

    print(
        "PASS 2: BASE + LoRA "
        "(adapter enabled)"
    )

    print("-" * 80)

    (
        lora_response,
        lora_inference_time,
        lora_truncated,
        lora_error,
    ) = run_generation(
        inputs,
        "LoRA"
    )

    if lora_error is not None:

        lora_predicted_disease = "unknown"

        lora_confidence = "unknown"

        lora_response_stored = (
            f"GENERATION ERROR: {lora_error}"
        )

    else:

        lora_final_text = (
            isolate_final_answer(
                lora_response
            )
        )

        lora_predicted_disease = (
            extract_predicted_disease(
                lora_final_text,
                truncated=lora_truncated
            )
        )

        lora_confidence = (
            extract_confidence(
                lora_final_text
            )
        )

        lora_response_stored = (
            lora_response
        )

    lora_correct = (
        lora_predicted_disease
        == ground_truth
    )

    print(
        "LoRA prediction :",
        lora_predicted_disease
    )

    print(
        "LoRA confidence :",
        lora_confidence
    )

    print(
        "LoRA correct    :",
        lora_correct
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    if base_correct and not lora_correct:

        verdict = (
            "LoRA REGRESSED this image"
        )

    elif lora_correct and not base_correct:

        verdict = (
            "LoRA IMPROVED this image"
        )

    elif base_correct and lora_correct:

        verdict = "Both correct"

    else:

        verdict = "Both incorrect"

    print()

    print(
        "Verdict:",
        verdict
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    y_true.append(
        ground_truth
    )

    y_pred_base.append(
        base_predicted_disease
    )

    y_pred_lora.append(
        lora_predicted_disease
    )

    results.append(
        {

            "image": filename,

            "ground_truth": ground_truth,

            "base_prediction":
                base_predicted_disease,

            "base_confidence":
                base_confidence,

            "base_truncated":
                base_truncated,

            "base_correct":
                base_correct,

            "base_inference_time":
                base_inference_time,

            "base_response":
                base_response_stored,

            "lora_prediction":
                lora_predicted_disease,

            "lora_confidence":
                lora_confidence,

            "lora_truncated":
                lora_truncated,

            "lora_correct":
                lora_correct,

            "lora_inference_time":
                lora_inference_time,

            "lora_response":
                lora_response_stored,

            "verdict":
                verdict,

        }
    )


# ============================================================
# STEP 8: SAVE RESULTS CSV
# ============================================================

print()
print()
print("=" * 80)
print("STEP 8: SAVE RESULTS")
print("=" * 80)
print()

with open(
    RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "ground_truth",
            "base_prediction",
            "base_confidence",
            "base_truncated",
            "base_correct",
            "base_inference_time",
            "base_response",
            "lora_prediction",
            "lora_confidence",
            "lora_truncated",
            "lora_correct",
            "lora_inference_time",
            "lora_response",
            "verdict",
        ],
    )

    writer.writeheader()

    for row in results:

        writer.writerow(row)

print(
    "Results saved:",
    RESULTS_CSV
)

print()


# ============================================================
# STEP 9: SAME METRICS AS COSMOS3
# ============================================================

print("=" * 80)
print("STEP 9: METRICS -- BASE vs BASE+LoRA")
print("=" * 80)
print()


if len(y_true) == 0:

    print(
        "No evaluation results available."
    )

    raise SystemExit(1)


labels = [
    "apple scab",
    "common rust",
    "northern leaf blight",
    "black rot",
    "septoria leaf spot",
    "leaf blast",
]


def compute_metrics(
    y_true,
    y_pred
):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "macro_precision":
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0
            ),

        "macro_recall":
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0
            ),

        "weighted_precision":
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0
            ),

        "weighted_recall":
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0
            ),

        "weighted_f1":
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0
            ),

    }


base_metrics = compute_metrics(
    y_true,
    y_pred_base
)

lora_metrics = compute_metrics(
    y_true,
    y_pred_lora
)


# ------------------------------------------------------------
# EXACT SAME DISPLAY ORDER AS COSMOS3
# ------------------------------------------------------------

metric_rows = [

    (
        "Accuracy",
        "accuracy"
    ),

    (
        "Precision (Macro)",
        "macro_precision"
    ),

    (
        "Precision (Weighted)",
        "weighted_precision"
    ),

    (
        "Recall (Macro)",
        "macro_recall"
    ),

    (
        "Recall (Weighted)",
        "weighted_recall"
    ),

    (
        "F1 (Macro)",
        "macro_f1"
    ),

    (
        "F1 (Weighted)",
        "weighted_f1"
    ),

]


print(
    f"{'Metric':36s}"
    f"{'Base':>12s}"
    f"{'Base + LoRA':>16s}"
    f"{'Delta':>12s}"
)

print("-" * 80)


for display_name, key in metric_rows:

    base_val = (
        base_metrics[key] * 100
    )

    lora_val = (
        lora_metrics[key] * 100
    )

    delta = (
        lora_val - base_val
    )

    delta_str = (
        f"{'+' if delta >= 0 else ''}"
        f"{delta:.2f}%"
    )

    print(
        f"{display_name:36s}"
        f"{base_val:11.2f}%"
        f"{lora_val:15.2f}%"
        f"{delta_str:>12s}"
    )

print()


# ============================================================
# SAVE METRICS CSV
# ============================================================

with open(
    METRICS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "base",
            "base_plus_lora",
            "delta",
        ]
    )

    for display_name, key in metric_rows:

        base_val = base_metrics[key]

        lora_val = lora_metrics[key]

        writer.writerow(
            [
                display_name,
                f"{base_val:.6f}",
                f"{lora_val:.6f}",
                f"{lora_val - base_val:.6f}",
            ]
        )

print(
    "Metrics saved:",
    METRICS_CSV
)

print()


# ============================================================
# STEP 10: CONFUSION MATRICES
# ============================================================

print("=" * 80)
print("STEP 10: CONFUSION MATRICES")
print("=" * 80)
print()


def print_confusion_matrix(
    y_true,
    y_pred,
    title
):

    matrix_labels = labels + [
        "other"
    ]

    y_pred_for_matrix = [

        p
        if p in labels
        else "other"

        for p in y_pred

    ]

    cm = confusion_matrix(
        y_true,
        y_pred_for_matrix,
        labels=matrix_labels
    )

    print(title)

    column_headers = [
        "Apple Scab",
        "Common Rust",
        "Northern LB",
        "Black Rot",
        "Septoria",
        "Leaf Blast",
        "Other",
    ]

    header = (
        f"{'Actual / Predicted':24s}"
        + "".join(
            f"{name:>16s}"
            for name in column_headers
        )
    )

    print(header)

    print("-" * 145)

    for label, row in zip(
        labels,
        cm
    ):

        print(
            f"{label:24s}"
            + "".join(
                f"{value:16d}"
                for value in row
            )
        )

    if any(
        p not in labels
        for p in y_pred
    ):

        print(
            "Note: 'Other' covers predictions "
            "that did not resolve to one of "
            "the five known classes."
        )

    print()


print_confusion_matrix(
    y_true,
    y_pred_base,
    "BASE MODEL"
)

print_confusion_matrix(
    y_true,
    y_pred_lora,
    "BASE + LoRA"
)


# ============================================================
# STEP 11: CLASSIFICATION REPORTS
# ============================================================

print("=" * 80)
print("STEP 11: PER-CLASS CLASSIFICATION REPORTS")
print("=" * 80)
print()

print(
    "BASE MODEL"
)

print(
    classification_report(
        y_true,
        y_pred_base,
        labels=labels,
        zero_division=0,
        digits=4
    )
)

print(
    "BASE + LoRA"
)

print(
    classification_report(
        y_true,
        y_pred_lora,
        labels=labels,
        zero_division=0,
        digits=4
    )
)


# ============================================================
# STEP 12: PER-IMAGE COMPARISON
# ============================================================

print("=" * 80)
print("STEP 12: PER-IMAGE COMPARISON")
print("=" * 80)
print()

print(
    f"{'Image':40s} "
    f"{'Actual':22s} "
    f"{'Base Pred':22s} "
    f"{'LoRA Pred':22s} "
    f"{'Verdict'}"
)

print("-" * 140)


for row in results:

    base_mark = (
        "PASS"
        if row["base_correct"]
        else "FAIL"
    )

    lora_mark = (
        "PASS"
        if row["lora_correct"]
        else "FAIL"
    )

    print(
        f"{row['image'][-40:]:40s} "
        f"{row['ground_truth'][:22]:22s} "
        f"{row['base_prediction'][:22]:22s} "
        f"{base_mark:4s} "
        f"{row['lora_prediction'][:22]:22s} "
        f"{lora_mark:4s} "
        f"{row['verdict']}"
    )

print()


# ============================================================
# SUMMARY
# ============================================================

improved = sum(
    1
    for r in results
    if r["verdict"]
    == "LoRA IMPROVED this image"
)

regressed = sum(
    1
    for r in results
    if r["verdict"]
    == "LoRA REGRESSED this image"
)

both_correct = sum(
    1
    for r in results
    if r["verdict"]
    == "Both correct"
)

both_wrong = sum(
    1
    for r in results
    if r["verdict"]
    == "Both incorrect"
)


base_correct_count = sum(
    1
    for r in results
    if r["base_correct"]
)

lora_correct_count = sum(
    1
    for r in results
    if r["lora_correct"]
)


base_times = [
    r["base_inference_time"]
    for r in results
    if r["base_inference_time"] > 0
]

lora_times = [
    r["lora_inference_time"]
    for r in results
    if r["lora_inference_time"] > 0
]


print("=" * 80)
print("FINAL EVALUATION SUMMARY")
print("=" * 80)
print()

print(
    "Total images evaluated :",
    len(results)
)

print(
    "Base correct           :",
    f"{base_correct_count}/{len(results)}"
)

print(
    "Base + LoRA correct    :",
    f"{lora_correct_count}/{len(results)}"
)

print()

print(
    "Base accuracy          :",
    f"{base_metrics['accuracy'] * 100:.2f}%"
)

print(
    "LoRA accuracy          :",
    f"{lora_metrics['accuracy'] * 100:.2f}%"
)

print(
    "Accuracy improvement   :",
    f"{(lora_metrics['accuracy'] - base_metrics['accuracy']) * 100:+.2f} percentage points"
)

print()

if base_times:

    print(
        "Base average inference:",
        f"{np.mean(base_times):.2f}s"
    )

if lora_times:

    print(
        "LoRA average inference:",
        f"{np.mean(lora_times):.2f}s"
    )

print()

print(
    "LoRA improved images   :",
    improved
)

print(
    "LoRA regressed images  :",
    regressed
)

print(
    "Both correct           :",
    both_correct
)

print(
    "Both incorrect         :",
    both_wrong
)

print()

print(
    "Results CSV:",
    RESULTS_CSV
)

print(
    "Metrics CSV:",
    METRICS_CSV
)

print()

print("=" * 80)
print("DONE")
print("=" * 80)
print()
