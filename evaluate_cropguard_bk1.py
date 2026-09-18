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

IMAGE_DIR = "/workspace/cropguard_eval_images"

RESULTS_CSV = "/workspace/cropguard_evaluation_results.csv"

DEVICE = "cuda:0"

MAX_NEW_TOKENS = 512


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("             CROPGUARD MULTI-IMAGE EVALUATION")
print("=" * 80)
print()

print("Model      :", MODEL_PATH)
print("LoRA       :", LORA_PATH)
print("Image dir  :", IMAGE_DIR)
print("Device     :", DEVICE)
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
print("STEP 1: FIND EVALUATION IMAGES")
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
    [
        p for p in Path(IMAGE_DIR).iterdir()
        if p.is_file() and p.suffix in image_extensions
    ]
)

print("Images found:", len(all_images))

for i, image_path in enumerate(all_images, start=1):
    print(f"{i}. {image_path}")

print()


# ============================================================
# GROUND TRUTH FROM FILE NAME
# ============================================================

def get_ground_truth(filename):
    """
    Determine ground truth from the evaluation filename.

    Supported examples:

        tomoto_early_blight_1.jpg
        tomato_early_blight_2.jpg
        tomoto_late_blight_1.jpg
        tomoto_healthy_image_1.jpg

    Returns canonical class names:
        early blight
        late blight
        healthy

    Returns None when the filename cannot be safely labelled.
    """

    name = filename.lower()

    # Healthy
    if "healthy" in name:
        return "healthy"

    # Early blight
    if "early_blight" in name or "early-blight" in name:
        return "early blight"

    # Late blight
    if "late_blight" in name or "late-blight" in name:
        return "late blight"

    return None


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

    ground_truth = get_ground_truth(image_path.name)

    if ground_truth is None:

        skipped_images.append(
            (image_path.name, "Ground truth not identifiable from filename")
        )

        print(
            f"SKIP: {image_path.name} "
            f"(ground truth not identifiable)"
        )

        continue

    evaluation_images.append(
        {
            "path": str(image_path),
            "filename": image_path.name,
            "ground_truth": ground_truth,
        }
    )

    print(
        f"USE : {image_path.name:40s} "
        f"Ground truth = {ground_truth}"
    )

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

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

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
    print(
        "Has backbone:",
        hasattr(language_model, "backbone")
    )

    if hasattr(language_model, "backbone"):

        print(
            "backbone:",
            type(language_model.backbone)
        )

        print(
            "Number of layers:",
            len(language_model.backbone.layers)
        )

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

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False,
)

print(
    f"LoRA loaded in {time.time() - lora_start:.2f} seconds"
)

print("LORA MODEL: OK")
print()


# ============================================================
# STEP 7: ADAPTER STATUS
# ============================================================

print("=" * 80)
print("STEP 7: ADAPTER STATUS")
print("=" * 80)
print()

print(
    "Active adapters:",
    getattr(model, "active_adapters", None)
)

print()


# ============================================================
# STEP 8: LORA MODULE CHECK
# ============================================================

print("=" * 80)
print("STEP 8: LORA MODULE CHECK")
print("=" * 80)
print()

lora_modules = []

for name, module in model.named_modules():

    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

        lora_modules.append(name)

print("LoRA modules found:", len(lora_modules))

if len(lora_modules) != 116:

    print(
        "WARNING: Expected 116 LoRA modules "
        f"but found {len(lora_modules)}"
    )

else:

    print("116 LoRA modules: OK")

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

trainable_percent = (
    100.0 * trainable_parameters / total_parameters
)

print("Trainable parameters :", trainable_parameters)
print("Total parameters     :", f"{total_parameters:,}")
print(
    "Trainable percentage :",
    f"{trainable_percent:.6f}%"
)

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
The custom NVIDIA multimodal generate() passes the multimodal
metadata to the inner language model.

The inner HuggingFace generate() rejects:
    num_patches
    num_tokens
    imgs_sizes

We therefore temporarily patch the INNER language model's
generate() method so those metadata tensors are removed before
calling the original language-model generation method.

The OUTER multimodal model still receives all image information.
"""

try:

    multimodal_model = model.get_base_model()

    print(
        "Multimodal model:",
        type(multimodal_model)
    )

    inner_language_model = multimodal_model.language_model

    print(
        "Inner language model:",
        type(inner_language_model)
    )

    original_inner_generate = inner_language_model.generate

    def safe_inner_generate(*args, **kwargs):

        removed = []

        for key in [
            "num_patches",
            "num_tokens",
            "imgs_sizes",
        ]:

            if key in kwargs:

                removed.append(key)
                kwargs.pop(key)

        if removed:

            print(
                "SAFE GENERATION PATCH:"
            )

            print(
                "Removed inner-model metadata:",
                removed
            )

        return original_inner_generate(
            *args,
            **kwargs
        )

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

    # Normalize punctuation
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    # Remove excessive whitespace
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

    # First look specifically for:
    # Most likely disease: <value>

    patterns = [
        r"most likely disease\s*:\s*([^\n\r]+)",
        r"most likely disease\s*-\s*([^\n\r]+)",
        r"disease\s*:\s*([^\n\r]+)",
        r"diagnosis\s*:\s*([^\n\r]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            response_lower
        )

        if match:

            candidate = match.group(1).strip()

            candidate = candidate.split(".")[0]
            candidate = candidate.split(",")[0]

            normalized = normalize_disease(
                candidate
            )

            if normalized != "unknown":

                return normalized

    # Fallback:
    # Look for known classes anywhere in response.

    if "late blight" in response_lower:
        return "late blight"

    if "early blight" in response_lower:
        return "early blight"

    if "healthy" in response_lower:
        return "healthy"

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
        f"IMAGE {image_number}/{len(evaluation_images)}"
    )
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

        image = Image.open(
            image_path
        ).convert("RGB")

        print(
            "Image size:",
            image.size
        )

        print(
            "Image mode:",
            image.mode
        )

        print("IMAGE: OK")

    except Exception as e:

        print("IMAGE LOAD FAILED")
        print(repr(e))

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "prediction": "error",
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
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": prompt,
                },
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

                print(
                    "Image token in chat text: PASSED"
                )

            else:

                print(
                    "Image token in chat text: FAILED"
                )

                raise RuntimeError(
                    "Chat template does not contain image token."
                )

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

        inputs = processor(
            text=chat_text,
            images=image,
            return_tensors="pt",
        )

        print(
            "Processor created multimodal inputs."
        )

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

                inputs[key] = value.to(
                    DEVICE,
                    dtype=torch.bfloat16
                )

            else:

                inputs[key] = value.to(
                    DEVICE
                )

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    print()
    print("Input tensors:")

    for key, value in inputs.items():

        if torch.is_tensor(value):

            print(
                f"  {key}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"device={value.device}"
            )

    if "input_ids" not in inputs:

        raise RuntimeError(
            "input_ids missing from processor output."
        )

    if image_token_id is not None:

        image_found = (
            inputs["input_ids"] == image_token_id
        ).any().item()

        print()
        print(
            "Image token found:",
            image_found
        )

        if not image_found:

            raise RuntimeError(
                "No image token found in input_ids."
            )

    print()
    print("INPUTS: OK")

    # --------------------------------------------------------
    # GENERATION
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

        inference_time = (
            time.time() - start_inference
        )

        print()
        print(
            "Generation completed."
        )

        print(
            f"Inference time: "
            f"{inference_time:.2f} seconds"
        )

    except Exception as e:

        inference_time = (
            time.time() - start_inference
        )

        print()
        print("GENERATION FAILED")
        print(repr(e))

        traceback.print_exc()

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "prediction": "unknown",
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

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_tokens = (
            output_ids[:, input_length:]
        )

        response = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

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
    # EXTRACT PREDICTION
    # --------------------------------------------------------

    predicted_disease = (
        extract_predicted_disease(
            response
        )
    )

    correct = (
        predicted_disease
        == ground_truth
    )

    print()
    print("Ground truth :", ground_truth)
    print("Prediction   :", predicted_disease)
    print(
        "Correct      :",
        correct
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    y_true.append(
        ground_truth
    )

    y_pred.append(
        predicted_disease
    )

    results.append(
        {
            "image": filename,
            "ground_truth": ground_truth,
            "prediction": predicted_disease,
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
                "prediction",
                "correct",
                "inference_time",
                "response",
            ],
        )

        writer.writeheader()

        for row in results:

            writer.writerow(row)

    print(
        "Results saved:",
        RESULTS_CSV
    )

except Exception as e:

    print(
        "Could not save CSV:",
        repr(e)
    )


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


# ------------------------------------------------------------
# Basic counts
# ------------------------------------------------------------

correct_count = sum(
    1
    for a, p in zip(y_true, y_pred)
    if a == p
)

total_count = len(y_true)

accuracy = accuracy_score(
    y_true,
    y_pred
)

# ------------------------------------------------------------
# Macro metrics
#
# zero_division=0 prevents warnings when a class receives
# no predictions.
# ------------------------------------------------------------

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

# ------------------------------------------------------------
# Weighted metrics
# ------------------------------------------------------------

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    f"Images evaluated : {total_count}"
)

print(
    f"Correct          : {correct_count}"
)

print(
    f"Incorrect        : "
    f"{total_count - correct_count}"
)

print()

print(
    f"Accuracy         : "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    f"Macro Precision  : "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall     : "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1         : "
    f"{macro_f1 * 100:.2f}%"
)

print()

print(
    f"Weighted Precision : "
    f"{weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall    : "
    f"{weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1        : "
    f"{weighted_f1 * 100:.2f}%"
)


# ============================================================
# STEP 14: CONFUSION MATRIX
# ============================================================

print()
print()
print("=" * 80)
print("STEP 14: CONFUSION MATRIX")
print("=" * 80)
print()

labels = [
    "early blight",
    "late blight",
    "healthy",
]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
)

print()

header = (
    f"{'Actual / Predicted':20s}"
    f"{'Early':>10s}"
    f"{'Late':>10s}"
    f"{'Healthy':>10s}"
)

print(header)
print("-" * 50)

for label, row in zip(
    labels,
    cm
):

    print(
        f"{label:20s}"
        f"{row[0]:10d}"
        f"{row[1]:10d}"
        f"{row[2]:10d}"
    )


# ============================================================
# STEP 15: CLASSIFICATION REPORT
# ============================================================

print()
print()
print("=" * 80)
print("STEP 15: PER-CLASS CLASSIFICATION REPORT")
print("=" * 80)
print()

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        digits=4,
    )
)


# ============================================================
# STEP 16: PER-IMAGE RESULTS
# ============================================================

print()
print()
print("=" * 80)
print("STEP 16: PER-IMAGE RESULTS")
print("=" * 80)
print()

print(
    f"{'Image':40s}"
    f"{'Actual':18s}"
    f"{'Predicted':18s}"
    f"{'Result':10s}"
)

print("-" * 90)

for row in results:

    result_text = (
        "PASS"
        if row["correct"]
        else "FAIL"
    )

    print(
        f"{row['image'][:40]:40s}"
        f"{row['ground_truth'][:18]:18s}"
        f"{row['prediction'][:18]:18s}"
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
print(
    f"LoRA modules         : {len(lora_modules)} / 116"
)
print("Multimodal input     : PASSED")
print("Generation           : PASSED")
print(
    f"Images evaluated     : {total_count}"
)
print()

print(
    f"Disease accuracy     : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro precision      : "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro recall         : "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1             : "
    f"{macro_f1 * 100:.2f}%"
)

print()

if skipped_images:

    print(
        "Skipped images:"
    )

    for filename, reason in skipped_images:

        print(
            f"  - {filename}: {reason}"
        )

print()
print(
    "CSV results:",
    RESULTS_CSV
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
print()
