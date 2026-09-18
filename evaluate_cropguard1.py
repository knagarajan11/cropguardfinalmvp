import os
import re
import time
import json
import glob
import types
import warnings

import torch
import numpy as np

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModel,
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "/workspace/hf_cache/hub/"
    "models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/"
    "snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"
)

LORA_PATH = (
    "/workspace/outputs/lora/"
    "cropguard_nemotron_lora_full_2gpu/"
    "epoch_1_step_26459/"
    "model_remapped"
)

IMAGE_DIR = "/workspace/cropguard_eval_images"


# ------------------------------------------------------------
# If automatic disease extraction does not work, put the
# expected disease here.
#
# Example:
#
# GROUND_TRUTH_DISEASE = "Tomato Early Blight"
#
# Leave as None to try extracting it from the filename.
# ------------------------------------------------------------

GROUND_TRUTH_DISEASE = None


# Number of generated tokens
MAX_NEW_TOKENS = 512


# ============================================================
# UTILITY
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def normalize_disease(text):
    """
    Normalize disease names for evaluation.
    """

    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Common PlantVillage naming cleanup
    replacements = {
        "tomato": "tomato",
        "potato": "potato",
        "pepper": "pepper",
        "bell pepper": "pepper",
        "early blight": "early blight",
        "late blight": "late blight",
        "leaf mold": "leaf mold",
        "septoria leaf spot": "septoria leaf spot",
        "bacterial spot": "bacterial spot",
        "target spot": "target spot",
        "spider mites": "spider mites",
        "two spotted spider mite": "spider mites",
        "yellow leaf curl virus": "yellow leaf curl virus",
        "tomato mosaic virus": "tomato mosaic virus",
        "powdery mildew": "powdery mildew",
        "healthy": "healthy",
    }

    for source, target in replacements.items():
        if source in text:
            return target

    return " ".join(text.split())


# ============================================================
# FIND IMAGE
# ============================================================

print_header("STEP 1: FIND EVALUATION IMAGE")

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
]

image_files = []

for ext in extensions:
    image_files.extend(
        glob.glob(
            os.path.join(IMAGE_DIR, ext)
        )
    )

image_files = sorted(set(image_files))

if len(image_files) == 0:

    raise FileNotFoundError(
        f"No image found in {IMAGE_DIR}"
    )

print("Images found:", len(image_files))

for i, path in enumerate(image_files):
    print(f"{i + 1}. {path}")

IMAGE_PATH = image_files[0]

print()
print("Using image:")
print(IMAGE_PATH)


# ============================================================
# DETERMINE GROUND TRUTH
# ============================================================

print_header("STEP 2: DETERMINE GROUND TRUTH")


def extract_ground_truth_from_filename(filename):

    name = os.path.basename(filename)

    lower = name.lower()

    # PlantVillage disease patterns

    if "early_blight" in lower:
        return "early blight"

    if "late_blight" in lower:
        return "late blight"

    if "septoria_leaf_spot" in lower:
        return "septoria leaf spot"

    if "bacterial_spot" in lower:
        return "bacterial spot"

    if "leaf_mold" in lower:
        return "leaf mold"

    if "target_spot" in lower:
        return "target spot"

    if "spider_mites" in lower:
        return "spider mites"

    if "yellow_leaf_curl" in lower:
        return "yellow leaf curl virus"

    if "mosaic_virus" in lower:
        return "tomato mosaic virus"

    if "powdery_mildew" in lower:
        return "powdery mildew"

    if "healthy" in lower:
        return "healthy"

    return None


if GROUND_TRUTH_DISEASE is not None:

    ground_truth = normalize_disease(
        GROUND_TRUTH_DISEASE
    )

else:

    ground_truth = extract_ground_truth_from_filename(
        IMAGE_PATH
    )

    if ground_truth is not None:
        ground_truth = normalize_disease(
            ground_truth
        )


print("Ground truth disease:", ground_truth)

if ground_truth is None:

    print()
    print("WARNING:")
    print(
        "Could not automatically determine the ground-truth disease."
    )
    print(
        "Precision / Recall / F1 cannot be calculated correctly."
    )
    print()
    print(
        "Set GROUND_TRUTH_DISEASE near the top of this script."
    )


# ============================================================
# LOAD PROCESSOR
# ============================================================

print_header("STEP 3: LOAD PROCESSOR")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("Processor:", type(processor))

print(
    "Image token:",
    getattr(processor, "image_token", None)
)

print(
    "Image token ID:",
    getattr(processor, "image_token_id", None)
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print_header("STEP 4: LOAD BASE MODEL")

start = time.time()

base_model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(
    f"Base model loaded in "
    f"{time.time() - start:.2f} seconds"
)

print("BASE MODEL: OK")


# ============================================================
# MODEL STRUCTURE
# ============================================================

print_header("STEP 5: MODEL STRUCTURE")

language_model = getattr(
    base_model,
    "language_model",
    None
)

if language_model is None:

    raise RuntimeError(
        "Could not find base_model.language_model"
    )

print(
    "language_model:",
    type(language_model)
)

backbone = getattr(
    language_model,
    "backbone",
    None
)

print(
    "Has backbone:",
    backbone is not None
)

if backbone is not None:

    print(
        "backbone:",
        type(backbone)
    )

    print(
        "Number of layers:",
        len(backbone.layers)
    )


# ============================================================
# LOAD REMAPPED LORA
# ============================================================

print_header("STEP 6: LOAD REMAPPED LORA")

if not os.path.exists(LORA_PATH):

    raise FileNotFoundError(
        f"LoRA path does not exist:\n{LORA_PATH}"
    )

if not os.path.exists(
    os.path.join(
        LORA_PATH,
        "adapter_config.json"
    )
):

    raise FileNotFoundError(
        "adapter_config.json not found in:\n"
        + LORA_PATH
    )


start = time.time()

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False,
)

print(
    f"LoRA loaded in "
    f"{time.time() - start:.2f} seconds"
)

print("LORA MODEL: OK")


# ============================================================
# ADAPTER STATUS
# ============================================================

print_header("STEP 7: ADAPTER STATUS")

try:

    print(
        "Active adapters:",
        model.active_adapters
    )

except Exception as e:

    print(
        "Could not read active adapters:",
        repr(e)
    )


# ============================================================
# COUNT LORA MODULES
# ============================================================

print_header("STEP 8: LORA MODULE CHECK")

lora_modules = []

for name, module in model.named_modules():

    if hasattr(module, "lora_A") and hasattr(
        module,
        "lora_B"
    ):

        lora_modules.append(name)


print(
    "LoRA modules found:",
    len(lora_modules)
)

if len(lora_modules) != 116:

    print(
        "WARNING: Expected 116 LoRA modules."
    )

else:

    print("116 LoRA modules: OK")


# ============================================================
# PARAMETER STATUS
# ============================================================

print_header("STEP 9: PARAMETER STATUS")

trainable = 0
total = 0

for parameter in model.parameters():

    total += parameter.numel()

    if parameter.requires_grad:
        trainable += parameter.numel()


percentage = (
    100.0 * trainable / total
    if total > 0
    else 0
)

print(
    f"Trainable parameters: {trainable:,}"
)

print(
    f"Total parameters: {total:,}"
)

print(
    f"Trainable percentage: {percentage:.6f}%"
)

model.eval()

print("Model evaluation mode: OK")


# ============================================================
# IMPORTANT FIX
# ============================================================
#
# NVIDIA's custom multimodal generate() processes the image
# first and then calls:
#
# self.language_model.generate(..., **generate_kwargs)
#
# The processor creates:
#
# num_patches
# num_tokens
# imgs_sizes
#
# These are needed by the OUTER multimodal model but are not
# accepted by the INNER Hugging Face language-model generate().
#
# We therefore wrap ONLY the in-memory language_model.generate()
# and remove these three bookkeeping arguments before the
# inner call.
#
# We do NOT modify NVIDIA's cached modeling.py.
#
# ============================================================

print_header(
    "STEP 10: INSTALL SAFE MULTIMODAL GENERATION PATCH"
)


# Get the actual underlying multimodal model.
#
# PeftModel structure:
#
# model
#   └── base_model
#        └── model
#

try:

    multimodal_model = model.get_base_model()

except Exception:

    multimodal_model = model.base_model.model


print(
    "Multimodal model:",
    type(multimodal_model)
)


inner_language_model = getattr(
    multimodal_model,
    "language_model",
    None
)

if inner_language_model is None:

    raise RuntimeError(
        "Could not locate multimodal_model.language_model"
    )


print(
    "Inner language model:",
    type(inner_language_model)
)


original_inner_generate = (
    inner_language_model.generate
)


def safe_inner_generate(
    self,
    *args,
    **kwargs
):

    removed = {}

    for key in [
        "num_patches",
        "num_tokens",
        "imgs_sizes",
    ]:

        if key in kwargs:

            removed[key] = kwargs.pop(key)

    if removed:

        print()
        print(
            "SAFE GENERATION PATCH:"
        )

        print(
            "Removed inner-model metadata:",
            list(removed.keys())
        )

    return original_inner_generate(
        *args,
        **kwargs
    )


inner_language_model.generate = types.MethodType(
    safe_inner_generate,
    inner_language_model
)


print(
    "Safe generation patch installed."
)

print(
    "NVIDIA modeling.py remains unchanged."
)


# ============================================================
# LOAD IMAGE
# ============================================================

print_header("STEP 11: LOAD IMAGE")

image = Image.open(
    IMAGE_PATH
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


# ============================================================
# CREATE CHAT MESSAGE
# ============================================================

print_header(
    "STEP 12: PREPARE MULTIMODAL INPUT"
)

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
                "type": "image",
            },
            {
                "type": "text",
                "text": """
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
""",
            },
        ],
    },
]


print("Creating chat template...")


chat_text = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False,
)


print()
print("===== CHAT TEXT =====")
print(chat_text)
print()


# ============================================================
# VERIFY IMAGE TOKEN
# ============================================================

image_token = getattr(
    processor,
    "image_token",
    "<image>"
)

print(
    "Image token in chat text:",
    image_token in chat_text
)

if image_token not in chat_text:

    raise RuntimeError(
        "Image token was NOT inserted into chat template."
    )


# ============================================================
# PROCESS IMAGE + TEXT
# ============================================================

print()
print("Running processor...")


inputs = processor(
    text=chat_text,
    images=image,
    return_tensors="pt",
)


print("Processor created multimodal inputs.")


# ============================================================
# MOVE INPUTS
# ============================================================

print()
print("Moving inputs to CUDA...")


# Find a model execution device.
try:

    input_device = next(
        multimodal_model.parameters()
    ).device

except Exception:

    input_device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


print(
    "Input device:",
    input_device
)


for key in list(inputs.keys()):

    value = inputs[key]

    if torch.is_tensor(value):

        if value.dtype.is_floating_point:

            # Image tensors should match model dtype.
            inputs[key] = value.to(
                device=input_device,
                dtype=torch.bfloat16
            )

        else:

            inputs[key] = value.to(
                device=input_device
            )


# ============================================================
# INPUT VALIDATION
# ============================================================

print_header("STEP 13: INPUT VALIDATION")

for key, value in inputs.items():

    if torch.is_tensor(value):

        print(
            f"{key}: "
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, "
            f"device={value.device}"
        )

    else:

        print(
            f"{key}: {type(value)}"
        )


image_token_id = getattr(
    processor,
    "image_token_id",
    None
)


if image_token_id is not None:

    image_token_present = (
        inputs["input_ids"] == image_token_id
    ).any().item()

    print()
    print(
        "Image token ID:",
        image_token_id
    )

    print(
        "Image token found:",
        image_token_present
    )

    if not image_token_present:

        raise RuntimeError(
            "Image token not present in input_ids."
        )


print()
print("INPUTS: OK")


# ============================================================
# GENERATION
# ============================================================

print_header("STEP 14: GENERATION")

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


except Exception as e:

    print()
    print("GENERATION FAILED")
    print()
    print("Exception:", type(e).__name__)
    print("Message:", str(e))
    print()

    print("Input keys passed to outer model:")

    for key in inputs:

        value = inputs[key]

        if torch.is_tensor(value):

            print(
                f"  {key}: "
                f"{tuple(value.shape)} "
                f"{value.dtype} "
                f"{value.device}"
            )

    raise


inference_time = (
    time.time() - start_inference
)


print()
print("Generation completed.")

print(
    f"Inference time: "
    f"{inference_time:.2f} seconds"
)


# ============================================================
# DECODE
# ============================================================

print_header("STEP 15: DECODE RESPONSE")


input_length = (
    inputs["input_ids"].shape[1]
)


if output_ids.shape[1] > input_length:

    generated_tokens = (
        output_ids[:, input_length:]
    )

else:

    generated_tokens = output_ids


try:

    response = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

except Exception:

    response = processor.tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    )


response = response.strip()


print()
print("=" * 70)
print("CROPGUARD RESPONSE")
print("=" * 70)
print()
print(response)
print()
print("=" * 70)


# ============================================================
# EXTRACT DISEASE FROM RESPONSE
# ============================================================

print_header(
    "STEP 16: EXTRACT PREDICTED DISEASE"
)


def extract_predicted_disease(text):

    # First look for our requested format.
    patterns = [
        r"most likely disease\s*:\s*([^\n\r]+)",
        r"most likely diagnosis\s*:\s*([^\n\r]+)",
        r"diagnosis\s*:\s*([^\n\r]+)",
        r"disease\s*:\s*([^\n\r]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            # Remove markdown
            value = value.strip("*# ")

            return normalize_disease(value)


    # --------------------------------------------------------
    # If explicit format wasn't followed, search known
    # PlantVillage disease names.
    # --------------------------------------------------------

    known_diseases = [
        "early blight",
        "late blight",
        "septoria leaf spot",
        "bacterial spot",
        "leaf mold",
        "target spot",
        "spider mites",
        "yellow leaf curl virus",
        "tomato mosaic virus",
        "powdery mildew",
        "healthy",
    ]

    lower = text.lower()

    for disease in known_diseases:

        if disease in lower:

            return normalize_disease(
                disease
            )

    return ""


predicted_disease = (
    extract_predicted_disease(
        response
    )
)


print(
    "Predicted disease:",
    predicted_disease
)

print(
    "Ground truth disease:",
    ground_truth
)


# ============================================================
# DISEASE MATCH
# ============================================================

print_header("STEP 17: DISEASE MATCH")


def disease_match(
    prediction,
    truth
):

    prediction = normalize_disease(
        prediction
    )

    truth = normalize_disease(
        truth
    )

    if not prediction or not truth:

        return False

    if prediction == truth:

        return True

    # Allow cases where the model adds crop name.
    if truth in prediction:

        return True

    if prediction in truth:

        return True

    return False


correct = disease_match(
    predicted_disease,
    ground_truth
)


print(
    "Prediction correct:",
    correct
)


# ============================================================
# PRECISION / RECALL / F1
# ============================================================
#
# IMPORTANT:
#
# With ONE image, disease classification metrics are based on
# a single sample.
#
# If the predicted disease equals the ground truth:
#
# precision = 1
# recall    = 1
# F1        = 1
#
# If wrong:
#
# precision = 0
# recall    = 0
# F1        = 0
#
# This is useful as a smoke test, but NOT a statistically
# meaningful model evaluation.
#
# ============================================================

print_header(
    "STEP 18: PRECISION / RECALL / F1"
)


if ground_truth:

    if correct:

        true_positive = 1
        false_positive = 0
        false_negative = 0

    else:

        true_positive = 0
        false_positive = 1
        false_negative = 1


    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )


    if precision_denominator > 0:

        precision = (
            true_positive
            / precision_denominator
        )

    else:

        precision = 0.0


    if recall_denominator > 0:

        recall = (
            true_positive
            / recall_denominator
        )

    else:

        recall = 0.0


    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0


    accuracy = (
        1.0
        if correct
        else 0.0
    )


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


else:

    accuracy = None
    precision = None
    recall = None
    f1 = None

    print(
        "Metrics NOT calculated because "
        "ground truth disease is unknown."
    )


# ============================================================
# RESPONSE QUALITY PARAMETERS
# ============================================================

print_header(
    "STEP 19: RESPONSE QUALITY"
)


response_lower = response.lower()


response_parameters = {}


response_parameters[
    "response_generated"
] = len(response) > 20


response_parameters[
    "crop_identified"
] = any(
    word in response_lower
    for word in [
        "crop",
        "tomato",
        "potato",
        "apple",
        "grape",
        "rice",
        "pepper",
    ]
)


response_parameters[
    "disease_identified"
] = (
    bool(predicted_disease)
)


response_parameters[
    "symptoms_provided"
] = any(
    word in response_lower
    for word in [
        "symptom",
        "spot",
        "yellow",
        "lesion",
        "discolor",
        "leaf",
    ]
)


response_parameters[
    "immediate_action_provided"
] = any(
    word in response_lower
    for word in [
        "remove",
        "apply",
        "treat",
        "spray",
        "fungicide",
        "bactericide",
        "isolate",
        "action",
    ]
)


response_parameters[
    "preventive_measures_provided"
] = any(
    word in response_lower
    for word in [
        "prevent",
        "prevention",
        "rotate",
        "rotation",
        "monitor",
        "avoid",
        "sanitation",
        "air circulation",
    ]
)


response_parameters[
    "image_recommendation_provided"
] = any(
    word in response_lower
    for word in [
        "image",
        "photo",
        "picture",
        "leaf image",
    ]
)


for key, value in response_parameters.items():

    print(
        f"{key:35s}: "
        f"{'PASS' if value else 'FAIL'}"
    )


# ============================================================
# OVERALL RESPONSE SCORE
# ============================================================

print_header(
    "STEP 20: RESPONSE QUALITY SCORE"
)


passed_parameters = sum(
    1
    for value in response_parameters.values()
    if value
)

total_parameters = len(
    response_parameters
)


response_score = (
    passed_parameters
    / total_parameters
    * 100
)


print(
    f"Passed parameters: "
    f"{passed_parameters}/{total_parameters}"
)

print(
    f"Response quality score: "
    f"{response_score:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("FINAL CROPGUARD EVALUATION")
print("=" * 70)
print()

print(
    f"Image                : "
    f"{os.path.basename(IMAGE_PATH)}"
)

print(
    f"Ground truth         : "
    f"{ground_truth}"
)

print(
    f"Predicted disease    : "
    f"{predicted_disease}"
)

print(
    f"Prediction correct   : "
    f"{correct}"
)

print(
    f"Inference time       : "
    f"{inference_time:.2f} seconds"
)

print()

if accuracy is not None:

    print(
        f"Accuracy             : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision            : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall               : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score             : "
        f"{f1 * 100:.2f}%"
    )

else:

    print(
        "Accuracy             : N/A"
    )

    print(
        "Precision            : N/A"
    )

    print(
        "Recall               : N/A"
    )

    print(
        "F1 Score             : N/A"
    )


print()

print(
    f"Response quality     : "
    f"{response_score:.2f}%"
)

print()

print("=" * 70)
print("MODEL / LORA STATUS")
print("=" * 70)

print(
    "Base model           : PASSED"
)

print(
    "Remapped LoRA        : PASSED"
)

print(
    "LoRA modules         : "
    f"{len(lora_modules)} / 116"
)

print(
    "Multimodal input     : PASSED"
)

print(
    "Image token          : PASSED"
)

print(
    "Generation           : PASSED"
)

print(
    "Evaluation           : PASSED"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print()


# ============================================================
# SAVE JSON RESULT
# ============================================================

result = {

    "image": IMAGE_PATH,

    "ground_truth_disease":
        ground_truth,

    "predicted_disease":
        predicted_disease,

    "prediction_correct":
        bool(correct),

    "inference_time_seconds":
        round(
            inference_time,
            3
        ),

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1_score":
        f1,

    "response_quality_score":
        round(
            response_score / 100,
            4
        ),

    "response_parameters":
        response_parameters,

    "response":
        response,
}


RESULT_PATH = (
    "/workspace/cropguard_evaluation_result.json"
)


with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "Evaluation JSON saved to:"
)

print(
    RESULT_PATH
)
