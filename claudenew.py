import os

# CUBLAS_WORKSPACE_CONFIG must be set before CUDA initializes, so this
# happens before importing torch or doing anything else. It's required
# for torch.use_deterministic_algorithms(True) to have any effect on
# CUBLAS operations (matmuls) on GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import re
import sys
import time
import csv
import random
import traceback
from pathlib import Path
from collections import Counter

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

# May contain images directly or PlantVillage-style class subfolders --
# searched recursively either way.
IMAGE_DIR = "/workspace/cropguard_eval_images"

RESULTS_CSV = "/workspace/cropguard_base_vs_lora_results.csv"

DEVICE = "cuda:0"

# Reasoning models need room for a scratchpad AND the structured answer.
MAX_NEW_TOKENS = 1536

EXPECTED_LORA_MODULES = 116

REASONING_END_MARKERS = ["</think>", "</thinking>", "final answer:"]

SEED = 42

# Whether to force torch.use_deterministic_algorithms(True) / cuDNN
# determinism. This is the top suspect for the device-side assert seen
# in the vision tower's custom patch-embedding kernel (RADIOv4-H,
# vit_patch_generator.py) -- forcing a deterministic kernel path on a
# third-party, unvetted custom op is a known way to surface latent bugs
# that never trip under the default (non-deterministic) path. Seeding
# alone (below) still meaningfully reduces run-to-run variance without
# forcing kernel selection, so if you hit CUDA device-side asserts with
# this on, set it to False and rely on repeat-and-vote (REPEATS_PER_IMAGE)
# alone to characterize stability instead.
ENABLE_DETERMINISTIC_ALGORITHMS = False

# How many times to generate per image, per model variant (base / LoRA).
# The reported prediction is the MAJORITY vote across these repeats, and
# the per-image "agreement" (e.g. "3/3" or "2/3") tells you how stable
# that image's answer actually is. Total generations = images * 2 * this
# number, so raising it directly multiplies runtime -- 3 is a reasonable
# starting point; use 5 if you want tighter confidence on borderline
# images at the cost of roughly 5/3x more generation time.
REPEATS_PER_IMAGE = 3


# ============================================================
# DETERMINISM
# ============================================================
#
# do_sample=False (greedy decoding) is only bit-for-bit reproducible if
# every underlying GPU kernel is deterministic. By default many aren't --
# cuBLAS matmuls and some attention kernels can pick different reduction
# orders between runs, which can flip a token at a close-probability
# decision point and, in a long reasoning trace, cascade into a
# completely different final answer. Seeding always helps; forcing
# deterministic algorithms helps further but is gated behind
# ENABLE_DETERMINISTIC_ALGORITHMS above since it can surface kernel bugs
# in custom/fused ops that otherwise lie dormant. warn_only=True means an
# op with no deterministic implementation falls back with a warning
# rather than crashing outright -- it does NOT protect against a kernel
# that has a deterministic path but that path itself is buggy.

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if ENABLE_DETERMINISTIC_ALGORITHMS:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("Deterministic algorithms: ENABLED (torch.use_deterministic_algorithms=True)")
else:
    print("Deterministic algorithms: DISABLED (seeding only) -- "
          "set ENABLE_DETERMINISTIC_ALGORITHMS=True to re-enable if you "
          "want to test whether it's implicated in a CUDA crash")


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("  CROPGUARD BASE vs BASE+LoRA -- DETERMINISTIC, REPEAT-AND-VOTE EVAL")
print("=" * 80)
print()

print("Model         :", MODEL_PATH)
print("LoRA          :", LORA_PATH)
print("Image dir     :", IMAGE_DIR, "(searched recursively)")
print("Device        :", DEVICE)
print("Max tokens    :", MAX_NEW_TOKENS)
print("Seed          :", SEED)
print("Repeats/image :", REPEATS_PER_IMAGE, "(per model variant)")
print()

print(
    "Both the base model and the base+LoRA model are run on the exact "
    "same processed inputs for each image, and each is generated "
    f"{REPEATS_PER_IMAGE} times. The reported prediction per variant is "
    "the majority vote across those repeats, with an agreement count "
    "showing how stable that image's answer actually is."
)
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
    try:
        rel = image_path.relative_to(IMAGE_DIR)
    except ValueError:
        rel = image_path
    print(f"{i}. {rel}")

print()


# ============================================================
# GROUND TRUTH
# ============================================================

def _match_label(text):
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
print(
    f"Total generations to run: {len(evaluation_images)} images x 2 variants x "
    f"{REPEATS_PER_IMAGE} repeats = {len(evaluation_images) * 2 * REPEATS_PER_IMAGE}"
)
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

image_token = getattr(processor, "image_token", None)
image_token_id = getattr(processor, "image_token_id", None)

print("Processor:", type(processor))
print("Image token:", image_token)
print("Image token ID:", image_token_id)
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

print(f"Base model loaded in {time.time() - base_start:.2f} seconds")
print("BASE MODEL: OK")
print()


# ============================================================
# STEP 5: LOAD REMAPPED LORA ON TOP OF THE SAME MODEL
# ============================================================

print("=" * 80)
print("STEP 5: LOAD REMAPPED LORA (SHARED WEIGHTS)")
print("=" * 80)
print()

lora_start = time.time()

model = PeftModel.from_pretrained(base_model, LORA_PATH, is_trainable=False)

print(f"LoRA loaded in {time.time() - lora_start:.2f} seconds")
print("Active adapters:", getattr(model, "active_adapters", None))

lora_modules = [
    name for name, module in model.named_modules()
    if hasattr(module, "lora_A") and hasattr(module, "lora_B")
]

print("LoRA modules found:", len(lora_modules))

if len(lora_modules) != EXPECTED_LORA_MODULES:
    print(f"WARNING: Expected {EXPECTED_LORA_MODULES} LoRA modules but found {len(lora_modules)}")
else:
    print(f"{EXPECTED_LORA_MODULES} LoRA modules: OK")

model.eval()

print("Model evaluation mode: OK")
print()


# ============================================================
# STEP 6: INSTALL SAFE MULTIMODAL GENERATION PATCH
# ============================================================

print("=" * 80)
print("STEP 6: INSTALL SAFE MULTIMODAL GENERATION PATCH")
print("=" * 80)
print()

try:
    multimodal_model = model.get_base_model()
    inner_language_model = multimodal_model.language_model

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

except Exception as e:
    print("FAILED TO INSTALL GENERATION PATCH:", repr(e))
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
    response_lower = response.lower()

    patterns = [
        r"most likely disease\s*:\s*([^\n\r]+)",
        r"most likely disease\s*-\s*([^\n\r]+)",
        r"disease\s*:\s*([^\n\r]+)",
        r"diagnosis\s*:\s*([^\n\r]+)",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, response_lower))
        for match in reversed(matches):
            candidate = match.group(1).strip()
            candidate = candidate.split(".")[0]
            candidate = candidate.split(",")[0]
            normalized = normalize_disease(candidate)
            if normalized != "unknown":
                return normalized

    if truncated:
        return "incomplete"

    last_match, last_index = None, -1
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
    response_lower = response.lower()

    def clean_candidate(text):
        return text.strip(" *_\"'").strip()

    matches = list(re.finditer(r"confidence(?:\s+level)?\s*:\s*([^\n\r.]+)", response_lower))
    if matches:
        candidate = matches[-1].group(1).strip()
        candidate = candidate.split(",")[0]
        candidate = clean_candidate(candidate)
        if candidate:
            return candidate[:40]

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
# SINGLE GENERATION HELPER
# ============================================================

def run_generation(inputs, label):
    """
    Runs model.generate() once on already-prepared, on-device inputs.
    Returns (response_text, inference_time, truncated, error_text).
    """

    start_inference = time.time()

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        inference_time = time.time() - start_inference

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[:, input_length:]

        truncated = generated_tokens.shape[1] >= MAX_NEW_TOKENS

        response = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        response = response.strip()

        print(f"    [{label}] {inference_time:6.2f}s, {generated_tokens.shape[1]:4d} tokens"
              + (" -- TRUNCATED" if truncated else ""))

        return response, inference_time, truncated, None

    except Exception as e:
        inference_time = time.time() - start_inference
        print(f"    [{label}] GENERATION FAILED: {repr(e)}")
        traceback.print_exc()
        return "", inference_time, False, repr(e)


# ============================================================
# REPEAT-AND-VOTE HELPER
# ============================================================

def run_repeated_generations(inputs, label, n_repeats):
    """
    Runs run_generation() n_repeats times and returns a summary dict:

        predictions       -- list of the extracted disease per repeat
        confidences       -- list of the extracted confidence per repeat
        majority          -- the most common prediction (ties broken by
                              whichever majority value was seen first)
        agreement         -- "k/n" repeats that matched the majority
        majority_confidence -- confidence from the first repeat whose
                              prediction equals the majority (a
                              representative value, not an average --
                              confidence strings like "high"/"72%" can't
                              be meaningfully averaged)
        any_truncated     -- True if ANY repeat hit the token cap
        total_time        -- summed inference time across all repeats
        representative_response -- raw response text from the first
                              repeat matching the majority (stored in the
                              CSV instead of all n raw responses, to keep
                              the file a reasonable size)
    """

    predictions = []
    confidences = []
    responses = []
    truncations = []
    total_time = 0.0

    for repeat_index in range(1, n_repeats + 1):
        response, inference_time, truncated, error = run_generation(inputs, f"{label} #{repeat_index}")
        total_time += inference_time

        if error is not None:
            predictions.append("unknown")
            confidences.append("unknown")
            responses.append(f"GENERATION ERROR: {error}")
            truncations.append(False)
            continue

        final_text = isolate_final_answer(response)
        predicted = extract_predicted_disease(final_text, truncated=truncated)
        confidence = extract_confidence(final_text)

        predictions.append(predicted)
        confidences.append(confidence)
        responses.append(response)
        truncations.append(truncated)

    vote_counts = Counter(predictions)
    majority, majority_count = vote_counts.most_common(1)[0]
    agreement = f"{majority_count}/{n_repeats}"

    # representative confidence/response: first repeat that agrees with
    # the majority prediction
    majority_index = predictions.index(majority)
    majority_confidence = confidences[majority_index]
    representative_response = responses[majority_index]

    print(f"    [{label}] repeats: {predictions}  ->  majority = {majority} ({agreement})")

    return {
        "predictions": predictions,
        "confidences": confidences,
        "majority": majority,
        "agreement": agreement,
        "majority_confidence": majority_confidence,
        "any_truncated": any(truncations),
        "total_time": total_time,
        "representative_response": representative_response,
    }


# ============================================================
# CSV SCHEMA + SAVE/LOAD HELPERS (used for resumability)
# ============================================================
#
# Progress is written to disk after EVERY image, not just once at the
# end, and on startup any existing RESULTS_CSV is read back in so a
# crash (CUDA or otherwise) only loses at most the one in-flight image,
# never the whole run. Re-running the script after a crash automatically
# skips images already present in the CSV.

CSV_FIELDNAMES = [
    "image",
    "ground_truth",
    "base_repeat_predictions",
    "base_majority_prediction",
    "base_agreement",
    "base_confidence",
    "base_any_truncated",
    "base_correct",
    "base_total_inference_time",
    "base_response",
    "lora_repeat_predictions",
    "lora_majority_prediction",
    "lora_agreement",
    "lora_confidence",
    "lora_any_truncated",
    "lora_correct",
    "lora_total_inference_time",
    "lora_response",
    "verdict",
]

_BOOL_FIELDS = ("base_any_truncated", "base_correct", "lora_any_truncated", "lora_correct")


def save_results_csv(results, path):
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        return True
    except Exception as e:
        print("Could not save CSV:", repr(e))
        return False


def load_existing_results_csv(path):
    """
    If a results CSV from a previous (possibly crashed) run exists,
    load it back in. CSV round-trips everything as strings, so boolean
    fields are converted back to actual bools. Returns an empty list if
    the file doesn't exist or can't be parsed.
    """
    if not os.path.isfile(path):
        return []

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Could not read existing results CSV ({path}): {repr(e)} -- starting fresh.")
        return []

    for row in rows:
        for field in _BOOL_FIELDS:
            row[field] = str(row.get(field, "")).strip().lower() == "true"

    return rows


def is_unrecoverable_cuda_error(exc):
    """
    Once a CUDA device-side assert fires, the CUDA context is poisoned
    for the rest of the process -- every subsequent CUDA call fails too,
    including completely unrelated ones (a plain .to(device) tensor
    move). There is no supported in-process recovery from this; the
    process must restart. This checks whether an exception looks like
    that specific failure mode, as opposed to an ordinary, recoverable
    error (a bad image file, a transient OOM, etc.).
    """
    message = str(exc)
    type_name = type(exc).__name__
    return (
        "device-side assert" in message.lower()
        or "cuda error" in message.lower()
        or type_name == "AcceleratorError"
    )


# ============================================================
# RESULTS STORAGE (with resume from a previous run, if any)
# ============================================================

results = load_existing_results_csv(RESULTS_CSV)

if results:
    print(f"Resuming: found {len(results)} already-completed image(s) in {RESULTS_CSV}.")

already_processed = {row["image"] for row in results}

y_true = [row["ground_truth"] for row in results]
y_pred_base = [row["base_majority_prediction"] for row in results]
y_pred_lora = [row["lora_majority_prediction"] for row in results]


# ============================================================
# STEP 7+: PROCESS EVERY IMAGE -- BASE THEN BASE+LORA, N REPEATS EACH
# ============================================================

for image_number, item in enumerate(evaluation_images, start=1):

    if item["filename"] in already_processed:
        print(f"\nSKIP (already processed in a previous run): {item['filename']}")
        continue

    image_path = item["path"]
    filename = item["filename"]
    ground_truth = item["ground_truth"]

    print()
    print()
    print("=" * 80)
    print(f"IMAGE {image_number}/{len(evaluation_images)}: {filename}")
    print("=" * 80)
    print("Ground truth:", ground_truth)
    print()

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print("IMAGE LOAD FAILED:", repr(e))

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "base_repeat_predictions": "",
                "base_majority_prediction": "error",
                "base_agreement": "0/0",
                "base_confidence": "unknown",
                "base_any_truncated": False,
                "base_correct": False,
                "base_total_inference_time": 0.0,
                "base_response": f"IMAGE ERROR: {repr(e)}",
                "lora_repeat_predictions": "",
                "lora_majority_prediction": "error",
                "lora_agreement": "0/0",
                "lora_confidence": "unknown",
                "lora_any_truncated": False,
                "lora_correct": False,
                "lora_total_inference_time": 0.0,
                "lora_response": f"IMAGE ERROR: {repr(e)}",
                "verdict": "Image error",
            }
        )

        y_true.append(ground_truth)
        y_pred_base.append("unknown")
        y_pred_lora.append("unknown")
        save_results_csv(results, RESULTS_CSV)
        continue

    # --------------------------------------------------------
    # CHAT TEMPLATE THROUGH STORE -- wrapped so that ANY failure here
    # (including a CUDA device-side assert) is handled explicitly rather
    # than crashing the whole batch. A device-side assert poisons the
    # CUDA context for the rest of the process -- there is no supported
    # in-process recovery -- so on detecting one we save everything
    # collected so far and exit cleanly rather than limping on with a
    # broken context (which, as observed, fails even on trivial
    # unrelated calls like a plain .to(device)).
    # --------------------------------------------------------

    try:
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

        chat_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        if image_token is not None and image_token not in chat_text:
            raise RuntimeError("Chat template does not contain image token.")

        inputs = processor(text=chat_text, images=image, return_tensors="pt")

        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.dtype.is_floating_point:
                    inputs[key] = value.to(DEVICE, dtype=torch.bfloat16)
                else:
                    inputs[key] = value.to(DEVICE)

        if "input_ids" not in inputs:
            raise RuntimeError("input_ids missing from processor output.")

        if image_token_id is not None:
            if not (inputs["input_ids"] == image_token_id).any().item():
                raise RuntimeError("No image token found in input_ids.")

        # ----------------------------------------------------
        # PASS 1: BASE MODEL, N REPEATS (adapter disabled)
        # ----------------------------------------------------

        print(f"  BASE MODEL ({REPEATS_PER_IMAGE} repeats, LoRA disabled):")

        with model.disable_adapter():
            base_summary = run_repeated_generations(inputs, "BASE", REPEATS_PER_IMAGE)

        base_correct = base_summary["majority"] == ground_truth

        # ----------------------------------------------------
        # PASS 2: BASE + LoRA, N REPEATS (adapter enabled)
        # ----------------------------------------------------

        print(f"  BASE + LoRA ({REPEATS_PER_IMAGE} repeats):")

        lora_summary = run_repeated_generations(inputs, "LoRA", REPEATS_PER_IMAGE)

        lora_correct = lora_summary["majority"] == ground_truth

        # ----------------------------------------------------
        # COMPARE
        # ----------------------------------------------------

        if base_correct and not lora_correct:
            verdict = "LoRA REGRESSED this image"
        elif lora_correct and not base_correct:
            verdict = "LoRA IMPROVED this image"
        elif base_correct and lora_correct:
            verdict = "Both correct"
        else:
            verdict = "Both incorrect"

        print(f"  Base majority: {base_summary['majority']} ({base_summary['agreement']})  |  "
              f"LoRA majority: {lora_summary['majority']} ({lora_summary['agreement']})  |  "
              f"Verdict: {verdict}")

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        y_true.append(ground_truth)
        y_pred_base.append(base_summary["majority"])
        y_pred_lora.append(lora_summary["majority"])

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "base_repeat_predictions": "|".join(base_summary["predictions"]),
                "base_majority_prediction": base_summary["majority"],
                "base_agreement": base_summary["agreement"],
                "base_confidence": base_summary["majority_confidence"],
                "base_any_truncated": base_summary["any_truncated"],
                "base_correct": base_correct,
                "base_total_inference_time": base_summary["total_time"],
                "base_response": base_summary["representative_response"],
                "lora_repeat_predictions": "|".join(lora_summary["predictions"]),
                "lora_majority_prediction": lora_summary["majority"],
                "lora_agreement": lora_summary["agreement"],
                "lora_confidence": lora_summary["majority_confidence"],
                "lora_any_truncated": lora_summary["any_truncated"],
                "lora_correct": lora_correct,
                "lora_total_inference_time": lora_summary["total_time"],
                "lora_response": lora_summary["representative_response"],
                "verdict": verdict,
            }
        )

        # Persist after every image -- a crash on the NEXT image should
        # never cost the results already earned on this one.
        save_results_csv(results, RESULTS_CSV)

    except Exception as e:
        if is_unrecoverable_cuda_error(e):
            print()
            print("=" * 80)
            print("UNRECOVERABLE CUDA ERROR -- STOPPING")
            print("=" * 80)
            print(repr(e))
            print()
            print(
                f"A CUDA device-side assert was hit while processing '{filename}'. "
                "The CUDA context is now poisoned for the rest of this process -- "
                "every further CUDA call will fail too, so continuing would not "
                "produce valid results. This is NOT something a try/except can "
                "recover from; the process must restart."
            )
            print()
            print(f"Progress so far ({len(results)} image(s)) has been saved to {RESULTS_CSV}.")
            print(
                "Re-running this script will automatically skip those already-"
                f"completed images and resume from '{filename}'. If '{filename}' "
                "fails again on restart, that points to something specific about "
                "this image/model interaction rather than a one-off GPU glitch -- "
                "worth checking its pixel dimensions against the vision tower's "
                "expected patch size, and re-testing with "
                "ENABLE_DETERMINISTIC_ALGORITHMS=False if it isn't already."
            )
            sys.exit(1)

        # A non-CUDA-corrupting error (bad chat template, unexpected
        # processor output, etc.) -- log it, record this image as failed,
        # and move on to the next one instead of losing the whole batch.
        print("IMAGE PROCESSING FAILED (non-fatal, continuing):", repr(e))
        traceback.print_exc()

        results.append(
            {
                "image": filename,
                "ground_truth": ground_truth,
                "base_repeat_predictions": "",
                "base_majority_prediction": "error",
                "base_agreement": "0/0",
                "base_confidence": "unknown",
                "base_any_truncated": False,
                "base_correct": False,
                "base_total_inference_time": 0.0,
                "base_response": f"PROCESSING ERROR: {repr(e)}",
                "lora_repeat_predictions": "",
                "lora_majority_prediction": "error",
                "lora_agreement": "0/0",
                "lora_confidence": "unknown",
                "lora_any_truncated": False,
                "lora_correct": False,
                "lora_total_inference_time": 0.0,
                "lora_response": f"PROCESSING ERROR: {repr(e)}",
                "verdict": "Processing error",
            }
        )

        y_true.append(ground_truth)
        y_pred_base.append("unknown")
        y_pred_lora.append("unknown")

        save_results_csv(results, RESULTS_CSV)


# ============================================================
# STEP 8: CONFIRM RESULTS SAVED
# ============================================================
#
# Every image's result was already written to RESULTS_CSV as soon as it
# completed (see the per-image loop above), so nothing left to save here
# -- this is just a final confirmation pass in case anything in memory
# somehow diverged from disk (it shouldn't, but cheap to guarantee).

print()
print()
print("=" * 80)
print("STEP 8: SAVE RESULTS")
print("=" * 80)
print()

if save_results_csv(results, RESULTS_CSV):
    print("Results saved:", RESULTS_CSV)



# ============================================================
# STEP 9: METRICS -- BASE vs LoRA (using majority-vote predictions)
# ============================================================

print()
print()
print("=" * 80)
print("STEP 9: METRICS -- BASE vs BASE+LoRA (majority vote)")
print("=" * 80)
print()

if len(y_true) == 0:
    print("No evaluation results available.")
    raise SystemExit(1)

labels = ["early blight", "late blight", "healthy"]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
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

print(f"{'Metric':22s}{'Base':>10s}{'Base+LoRA':>12s}{'Delta':>10s}")
print("-" * 58)

for display_name, key in metric_rows:
    base_val = base_metrics[key] * 100
    lora_val = lora_metrics[key] * 100
    delta = lora_val - base_val
    delta_str = f"{'+' if delta >= 0 else ''}{delta:.2f}%"
    print(f"{display_name:22s}{base_val:9.2f}%{lora_val:11.2f}%{delta_str:>10s}")

print()


# ============================================================
# STEP 10: STABILITY SUMMARY
# ============================================================
#
# This is the number that actually answers "can I trust this accuracy
# figure": how many images had every repeat agree, versus how many were
# borderline enough that the majority was won 2-1 (or worse).

print("=" * 80)
print("STEP 10: STABILITY (AGREEMENT ACROSS REPEATS)")
print("=" * 80)
print()


def stability_report(results, side_prefix, side_label):
    unanimous = sum(1 for r in results if r[f"{side_prefix}_agreement"] == f"{REPEATS_PER_IMAGE}/{REPEATS_PER_IMAGE}")
    split = len(results) - unanimous

    print(f"{side_label}: {unanimous}/{len(results)} images had unanimous agreement across all "
          f"{REPEATS_PER_IMAGE} repeats; {split} were split votes.")

    if split > 0:
        print("  Split-vote images:")
        for r in results:
            if r[f"{side_prefix}_agreement"] != f"{REPEATS_PER_IMAGE}/{REPEATS_PER_IMAGE}":
                print(f"    - {r['image']}: {r[f'{side_prefix}_repeat_predictions']} "
                      f"(majority {r[f'{side_prefix}_majority_prediction']}, {r[f'{side_prefix}_agreement']})")
    print()


stability_report(results, "base", "BASE MODEL")
stability_report(results, "lora", "BASE + LoRA")


# ============================================================
# STEP 11: CONFUSION MATRICES (BASE and LoRA, side by side)
# ============================================================

print("=" * 80)
print("STEP 11: CONFUSION MATRICES")
print("=" * 80)
print()


def print_confusion_matrix(y_true, y_pred, title):
    matrix_labels = labels + ["other"]
    y_pred_for_matrix = [p if p in labels else "other" for p in y_pred]
    cm = confusion_matrix(y_true, y_pred_for_matrix, labels=matrix_labels)

    print(title)
    header = f"{'Actual / Predicted':20s}{'Early':>10s}{'Late':>10s}{'Healthy':>10s}{'Other':>10s}"
    print(header)
    print("-" * 60)
    for label, row in zip(labels, cm):
        print(f"{label:20s}{row[0]:10d}{row[1]:10d}{row[2]:10d}{row[3]:10d}")
    if any(p not in labels for p in y_pred):
        print(
            "Note: 'Other' covers majority predictions that never resolved "
            "to one of the three known classes."
        )
    print()


print_confusion_matrix(y_true, y_pred_base, "BASE MODEL (majority vote)")
print_confusion_matrix(y_true, y_pred_lora, "BASE + LoRA (majority vote)")


# ============================================================
# STEP 12: PER-CLASS CLASSIFICATION REPORTS
# ============================================================

print("=" * 80)
print("STEP 12: PER-CLASS CLASSIFICATION REPORTS")
print("=" * 80)
print()

print("BASE MODEL")
print(classification_report(y_true, y_pred_base, labels=labels, zero_division=0, digits=4))

print("BASE + LoRA")
print(classification_report(y_true, y_pred_lora, labels=labels, zero_division=0, digits=4))


# ============================================================
# STEP 13: PER-IMAGE COMPARISON TABLE
# ============================================================

print("=" * 80)
print("STEP 13: PER-IMAGE COMPARISON")
print("=" * 80)
print()

print(
    f"{'Image':30s} {'Actual':13s} {'Base (agree)':18s} {'B':4s} "
    f"{'LoRA (agree)':18s} {'L':4s} {'Verdict':26s}"
)
print("-" * 130)

for row in results:
    base_mark = "PASS" if row["base_correct"] else "FAIL"
    lora_mark = "PASS" if row["lora_correct"] else "FAIL"
    base_display = f"{row['base_majority_prediction'][:12]} ({row['base_agreement']})"
    lora_display = f"{row['lora_majority_prediction'][:12]} ({row['lora_agreement']})"
    print(
        f"{row['image'][-30:]:30s} "
        f"{row['ground_truth'][:13]:13s} "
        f"{base_display:18s} "
        f"{base_mark:4s} "
        f"{lora_display:18s} "
        f"{lora_mark:4s} "
        f"{row['verdict']:26s}"
    )

print()

improved = sum(1 for r in results if r["verdict"] == "LoRA IMPROVED this image")
regressed = sum(1 for r in results if r["verdict"] == "LoRA REGRESSED this image")
both_correct = sum(1 for r in results if r["verdict"] == "Both correct")
both_wrong = sum(1 for r in results if r["verdict"] == "Both incorrect")

print(f"Images where LoRA improved on base : {improved}")
print(f"Images where LoRA regressed vs base: {regressed}")
print(f"Images both got right              : {both_correct}")
print(f"Images both got wrong              : {both_wrong}")
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
