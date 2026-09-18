import os
import re
import time
import csv
import traceback
import gc
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

# This is a "-Reasoning" model, which typically emits a chain-of-thought
# scratchpad BEFORE its final structured answer. 512 tokens is often not
# enough room for both the reasoning trace and the full 8-item answer,
# which causes generation to cut off mid-reasoning -- before the model
# ever states "Most likely disease:" or "Confidence level:". When that
# happens, fallback text-search extraction can grab a disease name the
# model was only reasoning about, not its actual conclusion. Raise this
# if you see truncated=True rows in the results.
MAX_NEW_TOKENS = 1536

# Reasoning models commonly wrap their scratchpad in a delimiter like
# "<think>...</think>". If present, extraction uses only the text AFTER
# the last closing marker (the final answer), never the scratchpad, even
# though the full raw response is still saved to the CSV either way.
REASONING_END_MARKERS = ["</think>", "</thinking>", "final answer:"]

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
# ISOLATE FINAL ANSWER FROM REASONING SCRATCHPAD
# ============================================================

def isolate_final_answer(response):
    """
    If the response contains a reasoning-scratchpad delimiter, return only
    the text AFTER the last occurrence of that delimiter -- this is the
    model's actual conclusion, not its intermediate thinking. Disease and
    confidence extraction should run on this, not on the raw scratchpad,
    since the scratchpad may casually mention multiple disease names
    before the model settles on one.

    If no delimiter is found, returns the response unchanged (nothing to
    strip -- either the model didn't reason aloud, or its reasoning wasn't
    wrapped in a marker we recognize).
    """
    response_lower = response.lower()

    last_cut = -1
    for marker in REASONING_END_MARKERS:
        idx = response_lower.rfind(marker)
        if idx != -1:
            cut_point = idx + len(marker)
            if cut_point > last_cut:
                last_cut = cut_point

    if last_cut == -1:
        return response

    return response[last_cut:].strip()


# ============================================================
# EXTRACT DISEASE FROM MODEL RESPONSE
# ============================================================

def extract_predicted_disease(response, truncated=False):
    """
    Extract the model's stated diagnosis.

    Reasoning models often say the disease name more than once while
    thinking out loud -- a tentative guess ("most likely disease: early
    blight?") followed later by a settled conclusion ("Most likely
    disease: late_blight"). re.search only returns the FIRST match, which
    would grab the tentative guess and miss the real conclusion. We
    instead collect every match for a given pattern and use the LAST one,
    since later statements in a reasoning trace are more likely to be the
    model's actual, settled answer.
    """
    response_lower = response.lower()

    patterns = [
        r"most likely disease\s*:\s*([^\n\r]+)",
        r"most likely disease\s*-\s*([^\n\r]+)",
        r"disease\s*:\s*([^\n\r]+)",
        r"diagnosis\s*:\s*([^\n\r]+)",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, response_lower))
        for match in reversed(matches):  # last match first
            candidate = match.group(1).strip()
            candidate = candidate.split(".")[0]
            candidate = candidate.split(",")[0]
            normalized = normalize_disease(candidate)
            if normalized != "unknown":
                return normalized

    # No explicit "disease: <value>" statement found anywhere.
    #
    # If generation was truncated, the model may simply never have
    # reached its conclusion -- in that case, scanning the scratchpad for
    # any mention of a class name is unreliable, since reasoning text
    # casually uses words like "healthy" (e.g. "green healthy tissue")
    # or floats a disease name only to reject it a sentence later. Rather
    # than manufacture a confident-looking wrong answer from that noise,
    # report the response as incomplete.
    if truncated:
        return "incomplete"

    # For a COMPLETE response that just never used the requested format,
    # a loose scan for the last-mentioned class name is a reasonable
    # last resort (the model likely did reach a conclusion, just not in
    # the exact phrasing we asked for).
    last_match = None
    last_index = -1
    for label, needle in (
        ("late blight", "late blight"),
        ("early blight", "early blight"),
        ("healthy", "healthy"),
    ):
        idx = response_lower.rfind(needle)
        if idx != -1 and idx > last_index:
            last_index = idx
            last_match = label

    return last_match if last_match else "unknown"


# ============================================================
# EXTRACT CONFIDENCE FROM MODEL RESPONSE
# ============================================================

def extract_confidence(response):
    """
    Look for a stated confidence level or percentage in the response.
    Returns a string like "high", "medium", "low", "82%", or "unknown".

    Uses the LAST match for the same reason as extract_predicted_disease:
    a reasoning trace may state a tentative confidence before settling
    on its final one.
    """
    response_lower = response.lower()

    # Look for "confidence level: <value>" or "confidence: <value>"
    matches = list(re.finditer(r"confidence(?:\s+level)?\s*:\s*([^\n\r.]+)", response_lower))
    if matches:
        candidate = matches[-1].group(1).strip()
        candidate = candidate.split(",")[0].strip()
        if candidate:
            return candidate[:40]

    # Look for a bare percentage near the word "confidence"
    matches = list(re.finditer(r"(\d{1,3}\s*%)\s*confidence", response_lower))
    if matches:
        return matches[-1].group(1).replace(" ", "")

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
y_pred_base = []
y_pred_lora = []


# ============================================================
# STEP 11+: PROCESS EVERY IMAGE -- BASE THEN BASE+LORA
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
        image = Image.open(image_path).convert("RGB").copy()
        print("Image size:", image.size)
        print("Image mode:", image.mode)
        print("IMAGE: OK")
    except Exception as e:
        print("IMAGE LOAD FAILED")
        print(repr(e))

        row = {
            "image": filename,
            "ground_truth": ground_truth,
            "base_prediction": "error",
            "base_confidence": "unknown",
            "base_truncated": False,
            "base_correct": False,
            "base_inference_time": 0.0,
            "base_response": f"IMAGE ERROR: {repr(e)}",
            "lora_prediction": "error",
            "lora_confidence": "unknown",
            "lora_truncated": False,
            "lora_correct": False,
            "lora_inference_time": 0.0,
            "lora_response": f"IMAGE ERROR: {repr(e)}",
            "verdict": "Both incorrect",
        }
        results.append(row)
        y_true.append(ground_truth)
        y_pred_base.append("unknown")
        y_pred_lora.append("unknown")
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
        print("Processor created multimodal inputs.")
    except Exception as e:
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
            print(
                f"  {key}: shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}"
            )

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
    # GENERATION HELPER
    # --------------------------------------------------------
    def generate_and_extract(label, adapter_disabled=False):
        print()
        print("-" * 80)
        if adapter_disabled:
            print("PASS 1: BASE MODEL (LoRA disabled)")
        else:
            print("PASS 2: BASE + LoRA (adapter enabled)")
        print("-" * 80)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        start_inference = time.time()

        try:
            context = model.disable_adapter() if adapter_disabled else None

            if context is not None:
                context.__enter__()

            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )
            finally:
                if context is not None:
                    context.__exit__(None, None, None)

            inference_time = time.time() - start_inference

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output_ids[:, input_length:]

            generated_token_count = generated_tokens.shape[1]
            truncated = generated_token_count >= MAX_NEW_TOKENS

            response = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0].strip()

            final_answer_text = isolate_final_answer(response)
            predicted_disease = extract_predicted_disease(
                final_answer_text,
                truncated=truncated,
            )
            confidence = extract_confidence(final_answer_text)

            print(
                f"[{label}] Generation completed in "
                f"{inference_time:.2f}s "
                f"({generated_token_count} tokens"
                + (", TRUNCATED" if truncated else "")
                + ")"
            )

            print(f"{label} prediction :", predicted_disease)
            print(f"{label} confidence :", confidence)

            # Free generation tensors immediately.
            del generated_tokens
            del output_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return (
                predicted_disease,
                confidence,
                truncated,
                inference_time,
                response,
                None,
            )

        except Exception as e:
            inference_time = time.time() - start_inference
            print(f"[{label}] GENERATION FAILED")
            print(repr(e))
            traceback.print_exc()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return (
                "unknown",
                "unknown",
                False,
                inference_time,
                f"GENERATION ERROR: {repr(e)}",
                repr(e),
            )

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------
    (
        base_predicted_disease,
        base_confidence,
        base_truncated,
        base_inference_time,
        base_response,
        base_error,
    ) = generate_and_extract("BASE", adapter_disabled=True)

    base_correct = base_predicted_disease == ground_truth
    print("BASE correct    :", base_correct)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------
    (
        lora_predicted_disease,
        lora_confidence,
        lora_truncated,
        lora_inference_time,
        lora_response,
        lora_error,
    ) = generate_and_extract("LoRA", adapter_disabled=False)

    lora_correct = lora_predicted_disease == ground_truth
    print("LoRA correct    :", lora_correct)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------
    if base_correct and not lora_correct:
        verdict = "LoRA REGRESSED this image"
    elif lora_correct and not base_correct:
        verdict = "LoRA IMPROVED this image"
    elif base_correct and lora_correct:
        verdict = "Both correct"
    else:
        verdict = "Both incorrect"

    print()
    print("Verdict:", verdict)

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------
    y_true.append(ground_truth)
    y_pred_base.append(base_predicted_disease)
    y_pred_lora.append(lora_predicted_disease)

    results.append(
        {
            "image": filename,
            "ground_truth": ground_truth,
            "base_prediction": base_predicted_disease,
            "base_confidence": base_confidence,
            "base_truncated": base_truncated,
            "base_correct": base_correct,
            "base_inference_time": base_inference_time,
            "base_response": base_response,
            "lora_prediction": lora_predicted_disease,
            "lora_confidence": lora_confidence,
            "lora_truncated": lora_truncated,
            "lora_correct": lora_correct,
            "lora_inference_time": lora_inference_time,
            "lora_response": lora_response,
            "verdict": verdict,
        }
    )

    # Release per-image tensors and image objects.
    try:
        del inputs
    except Exception:
        pass

    try:
        del image
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    print("Results saved:", RESULTS_CSV)

except Exception as e:
    print("Could not save CSV:", repr(e))


# ============================================================
# STEP 13: FINAL METRICS -- BASE vs BASE+LoRA
# ============================================================

print()
print()
print("=" * 80)
print("STEP 13: METRICS -- BASE vs BASE+LoRA")
print("=" * 80)
print()

labels = ["early blight", "late blight", "healthy"]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


base_metrics = compute_metrics(y_true, y_pred_base)
lora_metrics = compute_metrics(y_true, y_pred_lora)

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
    f"{'Metric':22s}"
    f"{'Base':>12s}"
    f"{'Base+LoRA':>12s}"
    f"{'Delta':>12s}"
)
print("-" * 58)

for display_name, key in metric_rows:
    base_val = base_metrics[key] * 100
    lora_val = lora_metrics[key] * 100
    delta = lora_val - base_val
    sign = "+" if delta >= 0 else ""
    print(
        f"{display_name:22s}"
        f"{base_val:11.2f}%"
        f"{lora_val:11.2f}%"
        f"{sign}{delta:10.2f}%"
    )


# ============================================================
# STEP 14: CONFUSION MATRICES
# ============================================================

print()
print()
print("=" * 80)
print("STEP 14: CONFUSION MATRICES")
print("=" * 80)
print()


def print_confusion_matrix(y_true_local, y_pred_local, title):
    matrix_labels = labels + ["other"]
    y_pred_for_matrix = [
        p if p in labels else "other"
        for p in y_pred_local
    ]

    cm = confusion_matrix(
        y_true_local,
        y_pred_for_matrix,
        labels=matrix_labels,
    )

    print(title)
    header = (
        f"{'Actual / Predicted':20s}"
        f"{'Early':>10s}"
        f"{'Late':>10s}"
        f"{'Healthy':>10s}"
        f"{'Other':>10s}"
    )
    print(header)
    print("-" * 60)

    for label, row in zip(labels, cm):
        print(
            f"{label:20s}"
            f"{row[0]:10d}"
            f"{row[1]:10d}"
            f"{row[2]:10d}"
            f"{row[3]:10d}"
        )

    print()


print_confusion_matrix(y_true, y_pred_base, "BASE MODEL")
print_confusion_matrix(y_true, y_pred_lora, "BASE + LoRA")


# ============================================================
# STEP 15: CLASSIFICATION REPORTS
# ============================================================

print("=" * 80)
print("STEP 15: PER-CLASS CLASSIFICATION REPORTS")
print("=" * 80)
print()

print("BASE MODEL")
print(
    classification_report(
        y_true,
        y_pred_base,
        labels=labels,
        zero_division=0,
        digits=4,
    )
)

print("BASE + LoRA")
print(
    classification_report(
        y_true,
        y_pred_lora,
        labels=labels,
        zero_division=0,
        digits=4,
    )
)


# ============================================================
# STEP 16: PER-IMAGE COMPARISON
# ============================================================

print("=" * 80)
print("STEP 16: PER-IMAGE COMPARISON")
print("=" * 80)
print()

print(
    f"{'Image':32s} {'Actual':13s} {'Base Pred':13s} {'B':4s} "
    f"{'LoRA Pred':13s} {'L':4s} {'Verdict':26s}"
)
print("-" * 110)

for row in results:
    base_mark = "PASS" if row["base_correct"] else "FAIL"
    lora_mark = "PASS" if row["lora_correct"] else "FAIL"

    print(
        f"{row['image'][-32:]:32s} "
        f"{row['ground_truth'][:13]:13s} "
        f"{row['base_prediction'][:13]:13s} "
        f"{base_mark:4s} "
        f"{row['lora_prediction'][:13]:13s} "
        f"{lora_mark:4s} "
        f"{row['verdict']:26s}"
    )

print()

improved = sum(
    1 for r in results
    if r["verdict"] == "LoRA IMPROVED this image"
)
regressed = sum(
    1 for r in results
    if r["verdict"] == "LoRA REGRESSED this image"
)
both_correct = sum(
    1 for r in results
    if r["verdict"] == "Both correct"
)
both_wrong = sum(
    1 for r in results
    if r["verdict"] == "Both incorrect"
)

print(f"Images where LoRA improved on base : {improved}")
print(f"Images where LoRA regressed vs base: {regressed}")
print(f"Images both got right              : {both_correct}")
print(f"Images both got wrong               : {both_wrong}")

if skipped_images:
    print()
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
