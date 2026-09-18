import os
import re
import time
import csv
import traceback
import signal
from pathlib import Path

import torch
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

# =============================================================================
# CROPGUARD - NEMOTRON 120-IMAGE BASE vs BASE+LoRA EVALUATION
# =============================================================================
#
# Purpose:
#   Evaluate the same 180-image dataset with:
#       1. Nemotron Base
#       2. Nemotron Base + CropGuard LoRA
#
# Methodology:
#   - Same image
#   - Same prompt
#   - Same processed inputs
#   - Same generation settings
#   - Base pass uses model.disable_adapter()
#   - LoRA pass uses the adapter enabled
#   - One 30B model copy is loaded; LoRA is attached to it
#
# IMPORTANT:
#   Do not change MODEL_PATH or LORA_PATH.
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

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

IMAGE_DIR = (
    "/workspace/gsh-team03/images100/"
    "eval_nemotron_100"
)

RESULTS_CSV = "/workspace/cropguard_nemotron_120_base_vs_lora_results.csv"

DEVICE = "cuda:0"

# 1024 was validated in the one-image diagnostic.
MAX_NEW_TOKENS = 1024

# Maximum time allowed for ONE Base or LoRA generation.
# If generation exceeds this limit, that pass is marked as a timeout
# and evaluation continues with the next pass/image.
GENERATION_TIMEOUT_SECONDS = 180

EXPECTED_TOTAL_IMAGES = 120
EXPECTED_IMAGES_PER_CLASS = 20

LABELS = [
    "apple scab",
    "common rust",
    "northern leaf blight",
    "black rot",
    "leaf blast",
    "septoria leaf spot",
]

LABEL_TO_NUMBER = {
    "apple scab": 1,
    "common rust": 2,
    "northern leaf blight": 3,
    "black rot": 4,
    "leaf blast": 5,
    "septoria leaf spot": 6,
}

NUMBER_TO_LABEL = {v: k for k, v in LABEL_TO_NUMBER.items()}

REASONING_END_MARKERS = [
    "</think>",
    "</thinking>",
    "final answer:",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 88)
print("        CROPGUARD NEMOTRON 120-IMAGE BASE vs BASE+LoRA EVALUATION")
print("=" * 88)
print()
print("Base model :", MODEL_PATH)
print("LoRA       :", LORA_PATH)
print("Image dir  :", IMAGE_DIR)
print("Device     :", DEVICE)
print("Max tokens :", MAX_NEW_TOKENS)
print()
print("Classes:")
for number, label in NUMBER_TO_LABEL.items():
    print(f"  {number}. {label}")
print()
print("Evaluation design:")
print("  120 images = 6 classes x 20 images")
print("  Each image is evaluated twice: Base and Base+LoRA")
print("  Same processed inputs are reused for both passes.")
print()


# =============================================================================
# CHECK PATHS
# =============================================================================

print("=" * 88)
print("STEP 0: CHECK PATHS")
print("=" * 88)

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Base model directory not found:\n{MODEL_PATH}")

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(f"LoRA directory not found:\n{LORA_PATH}")

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(f"Evaluation image directory not found:\n{IMAGE_DIR}")

print("BASE MODEL PATH : OK")
print("LORA PATH        : OK")
print("IMAGE DIRECTORY  : OK")
print()


# =============================================================================
# GROUND TRUTH
# =============================================================================

def normalize_folder_label(folder_name):
    """
    Convert PlantVillage-style folder names to one of the six disease labels.

    For folders such as:
        Apple___Apple_scab
        Corn___Common_rust
        Corn___Northern_Leaf_Blight

    only the portion AFTER "___" is the disease/class.
    """
    text = folder_name.lower().strip()

    # PlantVillage format: Crop___Disease
    if "___" in text:
        text = text.split("___", 1)[1]

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()

    aliases = {
        "apple scab": "apple scab",
        "common rust": "common rust",
        "northern leaf blight": "northern leaf blight",
        "black rot": "black rot",
        "leaf blast": "leaf blast",
        "septoria leaf spot": "septoria leaf spot",
    }

    return aliases.get(text)


def get_ground_truth(image_path, image_dir):
    """
    Ground truth is taken from the nearest class folder.

    Expected layout:
        eval_cosmos3_180/
          Apple___Apple_scab/
          Corn___Common_rust/
          Corn___Northern_Leaf_Blight/
          Grape___Black_rot/
          Rice___Leaf_Blast/
          Tomato___Septoria_leaf_spot/
    """
    image_path = Path(image_path)
    image_dir = Path(image_dir).resolve()

    current = image_path.parent

    while True:
        label = normalize_folder_label(current.name)
        if label is not None:
            return label

        current_resolved = current.resolve()

        if current_resolved == image_dir or current.parent == current:
            break

        current = current.parent

    # Filename fallback
    return normalize_folder_label(image_path.stem)


# =============================================================================
# STEP 1: FIND AND VALIDATE EXACTLY 120 IMAGES
# =============================================================================

print("=" * 88)
print("STEP 1: FIND AND VALIDATE 120 IMAGES")
print("=" * 88)
print()

all_images = sorted(
    p for p in Path(IMAGE_DIR).rglob("*")
    if p.is_file() and p.suffix in IMAGE_EXTENSIONS
)

print("Total image files found:", len(all_images))
print()

if len(all_images) != EXPECTED_TOTAL_IMAGES:
    raise RuntimeError(
        f"Expected exactly {EXPECTED_TOTAL_IMAGES} images, "
        f"but found {len(all_images)} in {IMAGE_DIR}"
    )

evaluation_images = []

for image_path in all_images:
    ground_truth = get_ground_truth(image_path, IMAGE_DIR)

    if ground_truth not in LABELS:
        raise RuntimeError(
            f"Could not identify one of the six ground-truth classes for:\n"
            f"{image_path}"
        )

    evaluation_images.append(
        {
            "path": str(image_path),
            "filename": str(image_path.relative_to(Path(IMAGE_DIR))),
            "ground_truth": ground_truth,
        }
    )

class_counts = {label: 0 for label in LABELS}

for item in evaluation_images:
    class_counts[item["ground_truth"]] += 1

print("Class distribution:")
for label in LABELS:
    print(f"  {label:25s}: {class_counts[label]}")

print()

bad_counts = [
    (label, count)
    for label, count in class_counts.items()
    if count != EXPECTED_IMAGES_PER_CLASS
]

if bad_counts:
    raise RuntimeError(
        "Dataset is not balanced at 20 images per class: "
        + str(bad_counts)
    )

print("DATASET VALIDATION: PASS")
print("Total:", len(evaluation_images))
print("Each class:", EXPECTED_IMAGES_PER_CLASS)
print()


# =============================================================================
# STEP 2: LOAD PROCESSOR
# =============================================================================

print("=" * 88)
print("STEP 2: LOAD PROCESSOR")
print("=" * 88)
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
print(f"Processor loaded in {time.time() - processor_start:.2f}s")
print()


# =============================================================================
# STEP 3: LOAD BASE MODEL
# =============================================================================

print("=" * 88)
print("STEP 3: LOAD BASE MODEL")
print("=" * 88)
print()

base_start = time.time()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

print()
print(f"Base model loaded in {time.time() - base_start:.2f}s")
print("BASE MODEL: OK")
print()


# =============================================================================
# STEP 4: LOAD LoRA ON THE SAME BASE MODEL
# =============================================================================

print("=" * 88)
print("STEP 4: LOAD LoRA")
print("=" * 88)
print()

lora_start = time.time()

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False,
)

print(f"LoRA loaded in {time.time() - lora_start:.2f}s")
print("LORA MODEL: OK")
print("Active adapters:", getattr(model, "active_adapters", None))
print()

lora_modules = [
    name
    for name, module in model.named_modules()
    if hasattr(module, "lora_A") and hasattr(module, "lora_B")
]

print("LoRA modules found:", len(lora_modules))

EXPECTED_LORA_MODULES = 116

if len(lora_modules) != EXPECTED_LORA_MODULES:
    print(
        f"WARNING: Expected {EXPECTED_LORA_MODULES} LoRA modules "
        f"but found {len(lora_modules)}"
    )
else:
    print(f"{EXPECTED_LORA_MODULES} LoRA modules: OK")

model.eval()
print("Model evaluation mode: OK")
print()


# =============================================================================
# STEP 5: SAFE MULTIMODAL GENERATION PATCH
# =============================================================================

print("=" * 88)
print("STEP 5: INSTALL SAFE MULTIMODAL GENERATION PATCH")
print("=" * 88)
print()

multimodal_model = model.get_base_model()
inner_language_model = multimodal_model.language_model

original_inner_generate = inner_language_model.generate

_STRIP_KEYS = (
    "num_patches",
    "num_tokens",
    "imgs_sizes",
)


def safe_inner_generate(*args, **kwargs):
    removed = [key for key in _STRIP_KEYS if key in kwargs]

    for key in removed:
        kwargs.pop(key)

    if removed:
        print(
            "SAFE GENERATION PATCH: removed inner-model metadata:",
            removed,
        )

    return original_inner_generate(*args, **kwargs)


inner_language_model.generate = safe_inner_generate

print("Safe generation patch installed.")
print()


# =============================================================================
# PROMPT
# =============================================================================

def create_prompt():
    return """
You are CropGuard, an AI assistant for crop disease detection.

Analyze the provided crop leaf image and identify the most likely disease.

The evaluation has exactly six possible disease classes:

1. apple scab
2. common rust
3. northern leaf blight
4. black rot
5. leaf blast
6. septoria leaf spot

IMPORTANT:
- Select exactly ONE of the six classes.
- At the end of your response, give the final answer as a number from 1 to 6.
- The final answer must appear after your reasoning.
- Use this format for the final answer:

Answer: <number>

You may explain the symptoms briefly before the final answer.
Do not select a class outside the six listed above.
"""


# =============================================================================
# RESPONSE HELPERS
# =============================================================================

def isolate_final_answer(response):
    """
    Return text after the final reasoning delimiter.
    """
    if not response:
        return ""

    text = response.strip()
    lower = text.lower()

    last_cut = -1

    for marker in REASONING_END_MARKERS:
        idx = lower.rfind(marker)
        if idx != -1:
            cut_point = idx + len(marker)
            if cut_point > last_cut:
                last_cut = cut_point

    if last_cut == -1:
        return text

    return text[last_cut:].strip()


def normalize_label_text(text):
    if not text:
        return None

    text = text.lower().strip()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    # Longest/specific labels first.
    for label in sorted(LABELS, key=len, reverse=True):
        if label in text:
            return label

    return None


def extract_predicted_disease(response, truncated=False):
    """
    Robust Nemotron extraction.

    Priority:
      1. Answer: <number>
      2. Final standalone number 1-6
      3. Disease/class text
      4. Number immediately before a disease name
      5. unknown/incomplete

    This specifically handles both diagnostic outputs:

        </think>
        1. apple scab

    and:

        </think>
        1
    """

    if not response:
        return "incomplete" if truncated else "unknown"

    text = response.strip()
    lower = text.lower()

    # ---------------------------------------------------------
    # 1. Explicit "Answer: N"
    # ---------------------------------------------------------
    answer_matches = list(
        re.finditer(
            r"\banswer\s*:\s*([1-6])\b",
            lower,
        )
    )

    if answer_matches:
        number = int(answer_matches[-1].group(1))
        return NUMBER_TO_LABEL[number]

    # ---------------------------------------------------------
    # 2. Final text after </think> often contains:
    #       1
    #       1. apple scab
    # ---------------------------------------------------------
    final_text = isolate_final_answer(text)

    final_number_matches = list(
        re.finditer(
            r"(?<!\d)([1-6])(?:\s*[\.\):\-])?(?:\s|$)",
            final_text.lower(),
        )
    )

    if final_number_matches:
        number = int(final_number_matches[-1].group(1))

        # Make sure this is genuinely a final answer and not a
        # numbered sentence in a longer paragraph.
        tail = final_text[
            final_number_matches[-1].end():
        ].strip()

        # If the tail starts with a matching disease name, this is
        # definitely the class answer.
        mapped_tail = normalize_label_text(tail)

        if mapped_tail is not None:
            return mapped_tail

        # A short final response such as "1" is also valid.
        if len(final_text) <= 40:
            return NUMBER_TO_LABEL[number]

        # Handle "1. apple scab" even if punctuation/spacing differs.
        if re.search(
            rf"^\s*{number}\s*[\.\):\-]?\s*"
            rf"{re.escape(NUMBER_TO_LABEL[number])}\b",
            final_text.lower(),
        ):
            return NUMBER_TO_LABEL[number]

    # ---------------------------------------------------------
    # 3. Explicit disease/diagnosis field
    # ---------------------------------------------------------
    field_patterns = [
        r"most likely disease\s*:\s*([^\n\r]+)",
        r"disease\s*:\s*([^\n\r]+)",
        r"diagnosis\s*:\s*([^\n\r]+)",
        r"class\s*:\s*([^\n\r]+)",
    ]

    candidates = []

    for pattern in field_patterns:
        matches = list(re.finditer(pattern, lower))
        for match in reversed(matches):
            candidates.append(match.group(1).strip())

    for candidate in candidates:
        label = normalize_label_text(candidate)
        if label is not None:
            return label

    # ---------------------------------------------------------
    # 4. Search final answer text for disease names.
    # ---------------------------------------------------------
    label = normalize_label_text(final_text)
    if label is not None:
        return label

    # ---------------------------------------------------------
    # 5. Last-resort search in complete response.
    # ---------------------------------------------------------
    found = []

    for label_name in LABELS:
        idx = lower.rfind(label_name)
        if idx != -1:
            found.append((idx, label_name))

    if found:
        found.sort(key=lambda x: x[0])
        return found[-1][1]

    return "incomplete" if truncated else "unknown"


def extract_confidence(response):
    if not response:
        return "unknown"

    text = response.lower()

    matches = list(
        re.finditer(
            r"confidence(?:\s+level)?\s*:\s*([^\n\r]+)",
            text,
        )
    )

    if matches:
        value = matches[-1].group(1).strip()
        value = value.strip(" *_\"'")
        return value[:40]

    percentage_matches = list(
        re.finditer(
            r"(\d{1,3}\s*%)\s*confidence",
            text,
        )
    )

    if percentage_matches:
        return percentage_matches[-1].group(1).replace(" ", "")

    return "unknown"


# =============================================================================
# GENERATION HELPER
# =============================================================================

def _generation_timeout_handler(signum, frame):
    raise TimeoutError(
        f"Generation exceeded "
        f"{GENERATION_TIMEOUT_SECONDS} seconds"
    )


def run_generation(inputs, label):
    start = time.time()

    previous_handler = signal.signal(
        signal.SIGALRM,
        _generation_timeout_handler,
    )

    signal.setitimer(
        signal.ITIMER_REAL,
        GENERATION_TIMEOUT_SECONDS,
    )

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        inference_time = time.time() - start

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[:, input_length:]

        generated_count = generated_tokens.shape[1]
        truncated = generated_count >= MAX_NEW_TOKENS

        response = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

        print(
            f"[{label}] Generation completed in "
            f"{inference_time:.2f}s "
            f"({generated_count} tokens"
            + (", TRUNCATED" if truncated else "")
            + ")"
        )

        return response, inference_time, truncated, None

    except TimeoutError as exc:
        inference_time = time.time() - start

        print()
        print(
            f"[{label}] GENERATION TIMEOUT after "
            f"{inference_time:.2f}s"
        )
        print(
            f"[{label}] Timeout limit: "
            f"{GENERATION_TIMEOUT_SECONDS}s"
        )

        return (
            "",
            inference_time,
            False,
            f"TIMEOUT: {GENERATION_TIMEOUT_SECONDS}s",
        )

    except Exception as exc:
        inference_time = time.time() - start

        print(
            f"[{label}] GENERATION FAILED: "
            f"{repr(exc)}"
        )

        traceback.print_exc()

        return (
            "",
            inference_time,
            False,
            repr(exc),
        )

    finally:
        signal.setitimer(
            signal.ITIMER_REAL,
            0,
        )

        signal.signal(
            signal.SIGALRM,
            previous_handler,
        )


# =============================================================================
# RESULTS
# =============================================================================

results = []

y_true = []
y_pred_base = []
y_pred_lora = []


# =============================================================================
# STEP 6: RUN ALL 120 IMAGES
# =============================================================================

print("=" * 88)
print("STEP 6: RUN 120 IMAGES - BASE + BASE LoRA")
print("=" * 88)
print()

overall_start = time.time()

for image_number, item in enumerate(evaluation_images, start=1):

    image_path = item["path"]
    filename = item["filename"]
    ground_truth = item["ground_truth"]

    print()
    print("=" * 88)
    print(f"IMAGE {image_number}/{EXPECTED_TOTAL_IMAGES}")
    print("=" * 88)
    print("File        :", filename)
    print("Ground truth:", ground_truth)
    print()

    # -------------------------------------------------------------------------
    # LOAD IMAGE
    # -------------------------------------------------------------------------

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print("IMAGE LOAD FAILED:", repr(exc))

        y_true.append(ground_truth)
        y_pred_base.append("unknown")
        y_pred_lora.append("unknown")

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "base_prediction": "error",
                "base_confidence": "unknown",
                "base_truncated": False,
                "base_correct": False,
                "base_inference_time": 0.0,
                "base_response": f"IMAGE ERROR: {repr(exc)}",
                "lora_prediction": "error",
                "lora_confidence": "unknown",
                "lora_truncated": False,
                "lora_correct": False,
                "lora_inference_time": 0.0,
                "lora_response": f"IMAGE ERROR: {repr(exc)}",
                "verdict": "Image error",
            }
        )
        continue

    print("Image size:", image.size)
    print("Image mode:", image.mode)

    # -------------------------------------------------------------------------
    # SAME PROMPT + SAME PROCESSED INPUTS FOR BOTH PASSES
    # -------------------------------------------------------------------------

    conversation = [
        {
            "role": "system",
            "content": (
                "You are CropGuard, an AI assistant for crop "
                "disease detection."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": create_prompt()},
            ],
        },
    ]

    try:
        chat_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        if image_token is not None and image_token not in chat_text:
            raise RuntimeError(
                "Chat template does not contain the expected image token."
            )

        inputs = processor(
            text=chat_text,
            images=image,
            return_tensors="pt",
        )

    except Exception as exc:
        print("PROCESSING FAILED:", repr(exc))
        raise

    # Move once; these inputs are reused for both passes.
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.dtype.is_floating_point:
                inputs[key] = value.to(
                    DEVICE,
                    dtype=torch.bfloat16,
                )
            else:
                inputs[key] = value.to(DEVICE)

    if "input_ids" not in inputs:
        raise RuntimeError("input_ids missing from processor output.")

    if image_token_id is not None:
        image_found = (
            inputs["input_ids"] == image_token_id
        ).any().item()

        if not image_found:
            raise RuntimeError(
                "No image token found in input_ids."
            )

    print("INPUTS: OK")

    # -------------------------------------------------------------------------
    # PASS 1: BASE
    # -------------------------------------------------------------------------

    print()
    print("-" * 88)
    print("PASS: BASE (LoRA disabled)")
    print("-" * 88)

    with model.disable_adapter():
        (
            base_response,
            base_time,
            base_truncated,
            base_error,
        ) = run_generation(inputs, "BASE")

    if base_error is not None:
        base_prediction = "unknown"
        base_confidence = "unknown"
        base_response_stored = (
            f"GENERATION ERROR: {base_error}"
        )
    else:
        base_final = isolate_final_answer(base_response)
        base_prediction = extract_predicted_disease(
            base_response,
            truncated=base_truncated,
        )
        base_confidence = extract_confidence(base_response)
        base_response_stored = base_response

        print("BASE final text :", repr(base_final))

    base_correct = base_prediction == ground_truth

    print("BASE prediction :", base_prediction)
    print("BASE confidence :", base_confidence)
    print("BASE correct    :", base_correct)

    # -------------------------------------------------------------------------
    # PASS 2: BASE + LoRA
    # -------------------------------------------------------------------------

    print()
    print("-" * 88)
    print("PASS: BASE + LoRA (adapter enabled)")
    print("-" * 88)

    (
        lora_response,
        lora_time,
        lora_truncated,
        lora_error,
    ) = run_generation(inputs, "LoRA")

    if lora_error is not None:
        lora_prediction = "unknown"
        lora_confidence = "unknown"
        lora_response_stored = (
            f"GENERATION ERROR: {lora_error}"
        )
    else:
        lora_final = isolate_final_answer(lora_response)
        lora_prediction = extract_predicted_disease(
            lora_response,
            truncated=lora_truncated,
        )
        lora_confidence = extract_confidence(lora_response)
        lora_response_stored = lora_response

        print("LoRA final text :", repr(lora_final))

    lora_correct = lora_prediction == ground_truth

    print("LoRA prediction :", lora_prediction)
    print("LoRA confidence :", lora_confidence)
    print("LoRA correct    :", lora_correct)

    # -------------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------------

    if base_correct and not lora_correct:
        verdict = "LoRA REGRESSED this image"
    elif lora_correct and not base_correct:
        verdict = "LoRA IMPROVED this image"
    elif base_correct and lora_correct:
        verdict = "Both correct"
    else:
        verdict = "Both incorrect"

    print("Verdict         :", verdict)

    # -------------------------------------------------------------------------
    # STORE
    # -------------------------------------------------------------------------

    y_true.append(ground_truth)
    y_pred_base.append(base_prediction)
    y_pred_lora.append(lora_prediction)

    results.append(
        {
            "image": filename,
            "ground_truth": ground_truth,

            "base_prediction": base_prediction,
            "base_confidence": base_confidence,
            "base_truncated": base_truncated,
            "base_correct": base_correct,
            "base_inference_time": round(base_time, 4),
            "base_response": base_response_stored,

            "lora_prediction": lora_prediction,
            "lora_confidence": lora_confidence,
            "lora_truncated": lora_truncated,
            "lora_correct": lora_correct,
            "lora_inference_time": round(lora_time, 4),
            "lora_response": lora_response_stored,

            "verdict": verdict,
        }
    )


overall_time = time.time() - overall_start


# =============================================================================
# STEP 7: SAVE CSV
# =============================================================================

print()
print("=" * 88)
print("STEP 7: SAVE RESULTS")
print("=" * 88)
print()

fieldnames = [
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
]

with open(
    RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(results)

print("Results saved:", RESULTS_CSV)
print()


# =============================================================================
# STEP 8: METRICS
# =============================================================================

print("=" * 88)
print("STEP 8: METRICS - BASE vs BASE+LoRA")
print("=" * 88)
print()


def compute_metrics(y_actual, y_predicted):
    return {
        "accuracy": accuracy_score(
            y_actual,
            y_predicted,
        ),
        "macro_precision": precision_score(
            y_actual,
            y_predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_actual,
            y_predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_actual,
            y_predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_precision": precision_score(
            y_actual,
            y_predicted,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            y_actual,
            y_predicted,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_actual,
            y_predicted,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
    }


base_metrics = compute_metrics(
    y_true,
    y_pred_base,
)

lora_metrics = compute_metrics(
    y_true,
    y_pred_lora,
)

metric_rows = [
    ("Accuracy", "accuracy"),
    ("Macro Precision", "macro_precision"),
    ("Macro Recall", "macro_recall"),
    ("Macro F1", "macro_f1"),
    ("Weighted Precision", "weighted_precision"),
    ("Weighted Recall", "weighted_recall"),
    ("Weighted F1", "weighted_f1"),
]

print(
    f"{'Metric':24s}"
    f"{'Base':>12s}"
    f"{'Base+LoRA':>14s}"
    f"{'Delta':>12s}"
)
print("-" * 64)

for display_name, key in metric_rows:
    base_value = base_metrics[key] * 100
    lora_value = lora_metrics[key] * 100
    delta = lora_value - base_value

    print(
        f"{display_name:24s}"
        f"{base_value:11.2f}%"
        f"{lora_value:13.2f}%"
        f"{delta:+11.2f}%"
    )

print()


# =============================================================================
# STEP 9: CONFUSION MATRICES
# =============================================================================

print("=" * 88)
print("STEP 9: CONFUSION MATRICES")
print("=" * 88)
print()


def print_confusion_matrix(y_actual, y_predicted, title):

    matrix_labels = LABELS + ["other"]

    matrix_predictions = [
        prediction
        if prediction in LABELS
        else "other"
        for prediction in y_predicted
    ]

    cm = confusion_matrix(
        y_actual,
        matrix_predictions,
        labels=matrix_labels,
    )

    print(title)
    print()

    short_labels = [
        "Apple scab",
        "Common rust",
        "N. leaf blight",
        "Black rot",
        "Leaf blast",
        "Septoria",
        "Other",
    ]

    print(
        f"{'Actual / Pred':20s}"
        + "".join(
            f"{name:>15s}"
            for name in short_labels
        )
    )
    print("-" * 125)

    for actual_label, row in zip(LABELS, cm):
        print(
            f"{actual_label:20s}"
            + "".join(
                f"{value:15d}"
                for value in row
            )
        )

    print()


print_confusion_matrix(
    y_true,
    y_pred_base,
    "BASE MODEL",
)

print_confusion_matrix(
    y_true,
    y_pred_lora,
    "BASE + LoRA",
)


# =============================================================================
# STEP 10: PER-CLASS REPORTS
# =============================================================================

print("=" * 88)
print("STEP 10: PER-CLASS CLASSIFICATION REPORTS")
print("=" * 88)
print()

print("BASE MODEL")
print(
    classification_report(
        y_true,
        y_pred_base,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
        digits=4,
    )
)

print("BASE + LoRA")
print(
    classification_report(
        y_true,
        y_pred_lora,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
        digits=4,
    )
)


# =============================================================================
# STEP 11: PER-CLASS ACCURACY COUNTS
# =============================================================================

print("=" * 88)
print("STEP 11: PER-CLASS RESULTS")
print("=" * 88)
print()

print(
    f"{'Class':25s}"
    f"{'Total':>8s}"
    f"{'Base':>10s}"
    f"{'LoRA':>10s}"
)

print("-" * 55)

for label in LABELS:

    indices = [
        i
        for i, truth in enumerate(y_true)
        if truth == label
    ]

    base_correct_count = sum(
        y_pred_base[i] == label
        for i in indices
    )

    lora_correct_count = sum(
        y_pred_lora[i] == label
        for i in indices
    )

    print(
        f"{label:25s}"
        f"{len(indices):8d}"
        f"{base_correct_count:4d}"
        f"/{len(indices):<5d}"
        f"{lora_correct_count:4d}"
        f"/{len(indices):<5d}"
    )

print()


# =============================================================================
# STEP 12: BASE vs LoRA SUMMARY
# =============================================================================

improved = sum(
    1
    for row in results
    if row["verdict"] == "LoRA IMPROVED this image"
)

regressed = sum(
    1
    for row in results
    if row["verdict"] == "LoRA REGRESSED this image"
)

both_correct = sum(
    1
    for row in results
    if row["verdict"] == "Both correct"
)

both_wrong = sum(
    1
    for row in results
    if row["verdict"] == "Both incorrect"
)

base_correct_total = sum(
    1
    for prediction, truth in zip(
        y_pred_base,
        y_true,
    )
    if prediction == truth
)

lora_correct_total = sum(
    1
    for prediction, truth in zip(
        y_pred_lora,
        y_true,
    )
    if prediction == truth
)

base_times = [
    row["base_inference_time"]
    for row in results
    if row["base_inference_time"] > 0
]

lora_times = [
    row["lora_inference_time"]
    for row in results
    if row["lora_inference_time"] > 0
]

base_truncated_count = sum(
    1
    for row in results
    if row["base_truncated"]
)

lora_truncated_count = sum(
    1
    for row in results
    if row["lora_truncated"]
)

print("=" * 88)
print("FINAL 120-IMAGE SUMMARY")
print("=" * 88)
print()

print("Images evaluated                  :", len(results))
print("Expected images                   :", EXPECTED_TOTAL_IMAGES)
print()

print(
    f"BASE correct                      : "
    f"{base_correct_total}/{len(y_true)} "
    f"({base_metrics['accuracy'] * 100:.2f}%)"
)

print(
    f"BASE + LoRA correct               : "
    f"{lora_correct_total}/{len(y_true)} "
    f"({lora_metrics['accuracy'] * 100:.2f}%)"
)

print()

print("LoRA improved images              :", improved)
print("LoRA regressed images             :", regressed)
print("Both correct                      :", both_correct)
print("Both incorrect                    :", both_wrong)

print()

if base_times:
    print(
        f"BASE average inference time      : "
        f"{sum(base_times) / len(base_times):.2f}s"
    )

if lora_times:
    print(
        f"LoRA average inference time      : "
        f"{sum(lora_times) / len(lora_times):.2f}s"
    )

print()

print("BASE truncated responses          :", base_truncated_count)
print("LoRA truncated responses          :", lora_truncated_count)

print()

print(
    f"Total evaluation wall time       : "
    f"{overall_time / 60:.2f} minutes"
)

print()
print("CSV:", RESULTS_CSV)
print()

print("=" * 88)
print("CROPGUARD NEMOTRON 120-IMAGE EVALUATION COMPLETE")
print("=" * 88)
print()
