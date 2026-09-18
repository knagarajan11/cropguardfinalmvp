# ============================================================
# CropGuard - Complete Multimodal LoRA Evaluation
# ============================================================
#
# Evaluates ALL images in:
#   /workspace/cropguard_eval_images
#
# Reports:
#   - Actual disease
#   - Predicted disease
#   - Model-derived confidence
#   - Correct / Incorrect
#   - Accuracy
#   - Precision
#   - Recall
#   - F1
#   - Macro / Weighted metrics
#   - Confusion matrix
#
# ============================================================

import os
import re
import csv
import time
import traceback

import torch
import numpy as np

from PIL import Image
from transformers import AutoProcessor, AutoModel
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = (
    "/workspace/hf_cache/hub/models--nvidia--"
    "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/"
    "snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"
)

LORA_PATH = (
    "/workspace/outputs/lora/"
    "cropguard_nemotron_lora_full_2gpu/"
    "epoch_1_step_26459/model_remapped"
)

IMAGE_DIR = "/workspace/cropguard_eval_images"

RESULT_CSV = "/workspace/cropguard_evaluation_results.csv"

MAX_NEW_TOKENS = 256

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 80)
print("CROPGUARD - COMPLETE IMAGE EVALUATION")
print("=" * 80)
print()

print("Base model:")
print(BASE_MODEL)
print()

print("LoRA:")
print(LORA_PATH)
print()

print("Image directory:")
print(IMAGE_DIR)
print()

print("Device:")
print(DEVICE)
print()


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(BASE_MODEL):
    raise FileNotFoundError(
        f"Base model not found:\n{BASE_MODEL}"
    )

if not os.path.exists(LORA_PATH):
    raise FileNotFoundError(
        f"LoRA model not found:\n{LORA_PATH}"
    )

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("=" * 80)
print("STEP 1: LOAD PROCESSOR")
print("=" * 80)
print()

start = time.time()

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

print("Processor:", type(processor))
print("Processor loaded in %.2f seconds" % (time.time() - start))
print()


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("=" * 80)
print("STEP 2: LOAD BASE MODEL")
print("=" * 80)
print()

start = time.time()

base_model = AutoModel.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print()
print("BASE MODEL: OK")
print("Loaded in %.2f seconds" % (time.time() - start))
print()


# ============================================================
# LOAD REMAPPED LORA
# ============================================================

print("=" * 80)
print("STEP 3: LOAD REMAPPED LORA")
print("=" * 80)
print()

start = time.time()

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False
)

model.eval()

print("LORA MODEL: OK")
print("Loaded in %.2f seconds" % (time.time() - start))
print()


# ============================================================
# ADAPTER CHECK
# ============================================================

print("=" * 80)
print("STEP 4: ADAPTER CHECK")
print("=" * 80)
print()

try:
    print("Active adapters:", model.active_adapters)
except Exception:
    print("Active adapters: default")

lora_modules = []

for name, module in model.named_modules():

    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        lora_modules.append(name)

print("LoRA modules found:", len(lora_modules))

if len(lora_modules) != 116:
    print(
        "WARNING: Expected 116 LoRA modules, "
        f"found {len(lora_modules)}"
    )
else:
    print("116 LoRA modules: OK")

print()


# ============================================================
# IMAGE FILES
# ============================================================

print("=" * 80)
print("STEP 5: FIND EVALUATION IMAGES")
print("=" * 80)
print()

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_files = []

for filename in sorted(os.listdir(IMAGE_DIR)):

    full_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if (
        os.path.isfile(full_path)
        and filename.lower().endswith(extensions)
    ):
        image_files.append(full_path)


print("Images found:", len(image_files))
print()

for i, path in enumerate(image_files, 1):
    print(
        f"{i:3d}. {os.path.basename(path)}"
    )

print()

if len(image_files) == 0:
    raise RuntimeError(
        "No image files found in "
        + IMAGE_DIR
    )


# ============================================================
# DISEASE NORMALIZATION
# ============================================================

def normalize_label(text):

    if text is None:
        return ""

    text = text.lower()

    # Remove file extension
    text = re.sub(
        r"\.(jpg|jpeg|png|bmp|webp)$",
        "",
        text
    )

    # Common PlantVillage naming cleanup
    text = text.replace("___", "_")
    text = text.replace("__", "_")

    # Replace separators
    text = text.replace("-", " ")
    text = text.replace("_", " ")

    # Remove numbers
    text = re.sub(
        r"\b\d+\b",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # --------------------------------------------------------
    # Known PlantVillage labels
    # --------------------------------------------------------

    replacements = {

        "tomato late blight":
            "Tomato Late Blight",

        "tomato early blight":
            "Tomato Early Blight",

        "tomato bacterial spot":
            "Tomato Bacterial Spot",

        "tomato leaf mold":
            "Tomato Leaf Mold",

        "tomato septoria leaf spot":
            "Tomato Septoria Leaf Spot",

        "tomato spider mites two spotted spider mite":
            "Tomato Spider Mites",

        "tomato target spot":
            "Tomato Target Spot",

        "tomato mosaic virus":
            "Tomato Mosaic Virus",

        "tomato yellow leaf curl virus":
            "Tomato Yellow Leaf Curl Virus",

        "potato early blight":
            "Potato Early Blight",

        "potato late blight":
            "Potato Late Blight",

        "apple scab":
            "Apple Scab",

        "apple black rot":
            "Apple Black Rot",

        "apple cedar apple rust":
            "Apple Cedar Apple Rust",

        "grape black rot":
            "Grape Black Rot",

        "grape esca black measles":
            "Grape Esca",

        "grape leaf blight":
            "Grape Leaf Blight",

        "pepper bell bacterial spot":
            "Pepper Bacterial Spot",

        "pepper bell healthy":
            "Pepper Healthy",

        "potato healthy":
            "Potato Healthy",

        "tomato healthy":
            "Tomato Healthy",

        "apple healthy":
            "Apple Healthy",

        "grape healthy":
            "Grape Healthy",
    }

    if text in replacements:
        return replacements[text]

    return text.title()


# ============================================================
# EXTRACT ACTUAL LABEL FROM FILENAME
# ============================================================

def get_actual_label(image_path):

    filename = os.path.basename(image_path)

    return normalize_label(filename)


# ============================================================
# DISEASE EXTRACTION FROM MODEL OUTPUT
# ============================================================

def extract_disease(response):

    if not response:
        return "Unknown"

    text = response.strip()

    # --------------------------------------------------------
    # Look specifically for:
    #
    # Most likely disease: XXXXX
    # --------------------------------------------------------

    patterns = [

        r"most likely disease\s*:\s*(.+)",

        r"most likely disease\s*-\s*(.+)",

        r"predicted disease\s*:\s*(.+)",

        r"disease\s*:\s*(.+)",

        r"diagnosis\s*:\s*(.+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            # Stop at newline
            value = value.split("\n")[0]

            # Remove markdown
            value = value.replace("*", "")
            value = value.replace("#", "")

            # Remove trailing punctuation
            value = value.strip(
                " .,:;-"
            )

            if len(value) > 1:
                return value


    # --------------------------------------------------------
    # Search known disease names
    # --------------------------------------------------------

    disease_patterns = [

        "tomato late blight",
        "tomato early blight",
        "tomato bacterial spot",
        "tomato leaf mold",
        "tomato septoria leaf spot",
        "tomato target spot",
        "tomato mosaic virus",
        "tomato yellow leaf curl virus",
        "tomato spider mites",

        "potato late blight",
        "potato early blight",

        "apple scab",
        "apple black rot",
        "apple cedar apple rust",

        "grape black rot",
        "grape esca",
        "grape leaf blight",

        "bacterial spot",
        "early blight",
        "late blight",
        "leaf mold",
        "septoria leaf spot",
        "target spot",
        "mosaic virus",
        "yellow leaf curl virus",
        "spider mites",

        "healthy",
    ]

    lower = text.lower()

    for disease in disease_patterns:

        if disease in lower:

            return disease.title()


    return "Unknown"


# ============================================================
# LABEL MATCHING
# ============================================================

def labels_match(actual, predicted):

    a = normalize_label(actual).lower()
    p = normalize_label(predicted).lower()

    if a == p:
        return True

    # --------------------------------------------------------
    # More flexible matching
    # --------------------------------------------------------

    a_words = set(a.split())
    p_words = set(p.split())

    # Exact disease phrase containment
    if a in p or p in a:
        return True

    # Ignore crop prefix where appropriate
    crop_words = {
        "tomato",
        "potato",
        "apple",
        "grape",
        "pepper",
        "bell",
    }

    a_disease = a_words - crop_words
    p_disease = p_words - crop_words

    if (
        len(a_disease) > 0
        and a_disease == p_disease
    ):
        return True

    return False


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(scores, generated_tokens):

    """
    Estimate confidence from generation token probabilities.

    We calculate the probability of each generated token from
    the model's generation scores and use the geometric mean.

    This is NOT the same as a calibrated disease classifier
    probability. It is a generative-model confidence estimate.
    """

    if scores is None:
        return None

    if len(scores) == 0:
        return None

    try:

        token_probs = []

        # scores[i] corresponds to generated token i
        for i, score in enumerate(scores):

            if i >= generated_tokens.shape[1]:
                break

            token_id = generated_tokens[
                0, i
            ]

            probabilities = torch.softmax(
                score[0].float(),
                dim=-1
            )

            probability = probabilities[
                token_id
            ].item()

            # Protect log()
            probability = max(
                min(probability, 1.0),
                1e-12
            )

            token_probs.append(
                probability
            )

        if len(token_probs) == 0:
            return None

        # Geometric mean
        log_mean = np.mean(
            np.log(token_probs)
        )

        confidence = np.exp(
            log_mean
        ) * 100.0

        return float(confidence)

    except Exception as e:

        print(
            "Confidence calculation warning:",
            repr(e)
        )

        return None


# ============================================================
# GENERATION PROMPT
# ============================================================

def build_chat_text():

    conversation = [

        {
            "role": "system",
            "content": ""
        },

        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": """
You are CropGuard, an AI assistant for crop disease detection
and agricultural advisory.

Analyze the provided crop leaf image.

Provide:

1. Crop name
2. Most likely disease
3. Visible symptoms
4. Possible alternative causes
5. Recommended immediate action
6. Preventive measures
7. Whether the farmer should provide another image
8. Confidence level

Give a concise agricultural recommendation.

IMPORTANT:
Clearly state the diagnosis using exactly this format:

Most likely disease: <disease name>
"""
                }
            ]
        }

    ]

    chat_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    return chat_text


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

print("=" * 80)
print("STEP 6: PROCESS IMAGES")
print("=" * 80)
print()


for index, image_path in enumerate(
    image_files,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print()
    print("#" * 80)
    print(
        f"IMAGE {index} / {len(image_files)}"
    )
    print("#" * 80)

    print("File:", filename)

    actual_label = get_actual_label(
        image_path
    )

    print(
        "Actual label:",
        actual_label
    )

    print()

    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        print(
            "Image size:",
            image.size
        )

        # ----------------------------------------------------
        # Chat template
        # ----------------------------------------------------

        chat_text = build_chat_text()

        if "<image>" not in chat_text:

            raise RuntimeError(
                "Image token was NOT found "
                "in chat template."
            )

        print(
            "Image token in chat text: PASSED"
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        inputs = processor(
            text=chat_text,
            images=image,
            return_tensors="pt"
        )

        # ----------------------------------------------------
        # Move inputs
        # ----------------------------------------------------

        for key, value in inputs.items():

            if torch.is_tensor(value):

                if value.dtype.is_floating_point:

                    inputs[key] = value.to(
                        device=DEVICE,
                        dtype=torch.bfloat16
                    )

                else:

                    inputs[key] = value.to(
                        DEVICE
                    )

        print(
            "Processor input: OK"
        )

        print(
            "Input IDs:",
            tuple(
                inputs["input_ids"].shape
            )
        )

        print(
            "Image token ID:",
            getattr(
                processor,
                "image_token_id",
                None
            )
        )

        image_token_id = getattr(
            processor,
            "image_token_id",
            None
        )

        if image_token_id is not None:

            found_image_token = (
                inputs["input_ids"]
                == image_token_id
            ).any().item()

            print(
                "Image token in input_ids:",
                found_image_token
            )

            if not found_image_token:

                raise RuntimeError(
                    "No image token found "
                    "in input_ids."
                )

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        print()
        print(
            "Generating..."
        )

        start = time.time()

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

        inference_time = (
            time.time() - start
        )

        output_ids = output.sequences

        # ----------------------------------------------------
        # Decode only newly generated tokens
        # ----------------------------------------------------

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_tokens = (
            output_ids[
                :,
                input_length:
            ]
        )

        response = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        response = response.strip()

        print(
            "Generation completed in "
            f"{inference_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Extract disease
        # ----------------------------------------------------

        predicted_label = extract_disease(
            response
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = calculate_confidence(
            output.scores,
            generated_tokens
        )

        # ----------------------------------------------------
        # Correctness
        # ----------------------------------------------------

        correct = labels_match(
            actual_label,
            predicted_label
        )

        # ----------------------------------------------------
        # Print response
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("MODEL RESPONSE")
        print("-" * 80)
        print()

        print(response)

        print()
        print("-" * 80)
        print("IMAGE RESULT")
        print("-" * 80)

        print(
            "Actual disease    :",
            actual_label
        )

        print(
            "Predicted disease  :",
            predicted_label
        )

        if confidence is None:

            print(
                "Confidence         : N/A"
            )

        else:

            print(
                "Confidence         : "
                f"{confidence:.2f}%"
            )

        print(
            "Result             :",
            "CORRECT" if correct
            else "INCORRECT"
        )

        print(
            "Inference time     : "
            f"{inference_time:.2f} sec"
        )

        print("-" * 80)

        results.append({

            "image":
                filename,

            "actual":
                actual_label,

            "predicted":
                predicted_label,

            "confidence":
                confidence,

            "correct":
                correct,

            "inference_time_sec":
                inference_time,

            "response":
                response
        })

    except Exception as e:

        print()
        print(
            "IMAGE PROCESSING FAILED"
        )

        print(
            repr(e)
        )

        traceback.print_exc()

        results.append({

            "image":
                filename,

            "actual":
                actual_label,

            "predicted":
                "ERROR",

            "confidence":
                None,

            "correct":
                False,

            "inference_time_sec":
                None,

            "response":
                "ERROR: " + repr(e)
        })

        print()
        print(
            "Continuing with next image..."
        )


# ============================================================
# VALID RESULTS
# ============================================================

valid_results = [

    r for r in results

    if r["predicted"] != "ERROR"
    and r["predicted"] != "Unknown"
]


y_true = [
    r["actual"]
    for r in valid_results
]

y_pred = [
    r["predicted"]
    for r in valid_results
]


# ============================================================
# FINAL EVALUATION
# ============================================================

print()
print()
print("=" * 80)
print("FINAL EVALUATION")
print("=" * 80)
print()

print(
    "Total images processed :",
    len(results)
)

print(
    "Successful predictions :",
    len(valid_results)
)

print(
    "Failed predictions     :",
    len(results) - len(valid_results)
)

print()


# ============================================================
# BASIC COUNTS
# ============================================================

correct_count = sum(
    1
    for r in results
    if r["correct"]
)

incorrect_count = (
    len(results) - correct_count
)

print(
    "Correct predictions    :",
    correct_count
)

print(
    "Incorrect predictions  :",
    incorrect_count
)

print()


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

if len(valid_results) > 0:

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print()

    print(
        f"Accuracy              : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision (Macro)     : "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Recall (Macro)        : "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"F1 Score (Macro)      : "
        f"{f1_macro * 100:.2f}%"
    )

    print()

    print(
        f"Precision (Weighted)  : "
        f"{precision_weighted * 100:.2f}%"
    )

    print(
        f"Recall (Weighted)     : "
        f"{recall_weighted * 100:.2f}%"
    )

    print(
        f"F1 Score (Weighted)   : "
        f"{f1_weighted * 100:.2f}%"
    )

    print()

else:

    print(
        "No valid predictions available "
        "for classification metrics."
    )

    accuracy = 0
    precision_macro = 0
    recall_macro = 0
    f1_macro = 0
    precision_weighted = 0
    recall_weighted = 0
    f1_weighted = 0


# ============================================================
# CONFIDENCE
# ============================================================

confidence_values = [

    r["confidence"]
    for r in results

    if r["confidence"] is not None
]


print("=" * 80)
print("CONFIDENCE")
print("=" * 80)
print()

if confidence_values:

    average_confidence = np.mean(
        confidence_values
    )

    median_confidence = np.median(
        confidence_values
    )

    minimum_confidence = np.min(
        confidence_values
    )

    maximum_confidence = np.max(
        confidence_values
    )

    print(
        f"Average confidence : "
        f"{average_confidence:.2f}%"
    )

    print(
        f"Median confidence  : "
        f"{median_confidence:.2f}%"
    )

    print(
        f"Minimum confidence : "
        f"{minimum_confidence:.2f}%"
    )

    print(
        f"Maximum confidence : "
        f"{maximum_confidence:.2f}%"
    )

else:

    average_confidence = None

    print(
        "Confidence unavailable."
    )


print()


# ============================================================
# PER IMAGE SUMMARY
# ============================================================

print("=" * 80)
print("PER IMAGE SUMMARY")
print("=" * 80)
print()

print(
    f"{'IMAGE':35} "
    f"{'ACTUAL':25} "
    f"{'PREDICTED':25} "
    f"{'CONF':>8} "
    f"{'RESULT'}"
)

print("-" * 110)

for r in results:

    conf = (
        f"{r['confidence']:.2f}%"
        if r["confidence"] is not None
        else "N/A"
    )

    result_text = (
        "CORRECT"
        if r["correct"]
        else "INCORRECT"
    )

    print(
        f"{r['image'][:35]:35} "
        f"{r['actual'][:25]:25} "
        f"{r['predicted'][:25]:25} "
        f"{conf:>8} "
        f"{result_text}"
    )

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

if len(valid_results) > 0:

    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

if len(valid_results) > 0:

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print()

    print(
        "Labels:"
    )

    for i, label in enumerate(labels):

        print(
            f"{i}: {label}"
        )

    print()

    print(cm)

    print()


# ============================================================
# SAVE CSV
# ============================================================

print("=" * 80)
print("SAVE RESULTS")
print("=" * 80)
print()

with open(
    RESULT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "actual",
            "predicted",
            "confidence",
            "correct",
            "inference_time_sec",
            "response"
        ]
    )

    writer.writeheader()

    for r in results:

        writer.writerow(r)


print(
    "Detailed CSV saved:"
)

print(
    RESULT_CSV
)

print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 80)
print("CROPGUARD EVALUATION COMPLETE")
print("=" * 80)
print()

print(
    f"Images processed : {len(results)}"
)

print(
    f"Correct          : {correct_count}"
)

print(
    f"Incorrect        : {incorrect_count}"
)

if len(valid_results) > 0:

    print(
        f"Accuracy         : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision        : "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Recall           : "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"F1 Score         : "
        f"{f1_macro * 100:.2f}%"
    )

if average_confidence is not None:

    print(
        f"Avg Confidence   : "
        f"{average_confidence:.2f}%"
    )

print()

print(
    "CSV:",
    RESULT_CSV
)

print()

print("=" * 80)
print("DONE")
print("=" * 80)
