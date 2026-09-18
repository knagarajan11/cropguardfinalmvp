#!/usr/bin/env python3

import os
import re
import time
import json
import torch

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = (
    "/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/"
    "snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"
)

LORA_MODEL = (
    "/workspace/outputs/lora/"
    "cropguard_nemotron_lora_full_2gpu/"
    "epoch_1_step_26459/"
    "model_remapped"
)

# CHANGE THIS ONLY if your image has a different name
IMAGE_PATH = "/workspace/cropguard_eval_images/train_041363.jpg"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# EXPECTED GROUND TRUTH
# ============================================================
#
# IMPORTANT:
#
# Put the actual disease label of your PlantVillage image here.
#
# Example:
#
#   "Tomato___Early_blight"
#
# or
#
#   "Potato___Late_blight"
#
# or
#
#   "Apple___Apple_scab"
#
# Use None if you only want inference without classification
# metrics.
#
# ============================================================

GROUND_TRUTH = None

# Example:
#
# GROUND_TRUTH = "Tomato___Early_blight"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(text):
    """
    Normalize disease labels for comparison.
    """

    if text is None:
        return None

    text = str(text).lower()

    text = text.replace("___", " ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_disease(response):
    """
    Try to extract the predicted disease from the model response.
    """

    text = response.strip()

    # --------------------------------------------------------
    # Look for explicit disease fields
    # --------------------------------------------------------

    patterns = [
        r"most likely disease\s*:\s*(.+)",
        r"disease\s*:\s*(.+)",
        r"predicted disease\s*:\s*(.+)",
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

            value = value.split("\n")[0]

            value = value.strip(
                " .,:;*-"
            )

            if value:
                return value

    # --------------------------------------------------------
    # Look for common PlantVillage diseases
    # --------------------------------------------------------

    known_diseases = [

        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Bacterial_spot",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",

        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",

        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",

        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",

        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
    ]

    normalized_response = normalize_label(text)

    for disease in known_diseases:

        disease_normalized = normalize_label(
            disease
        )

        if disease_normalized in normalized_response:

            return disease

    return None


def calculate_metrics(
    ground_truth,
    prediction
):
    """
    Calculate binary classification metrics
    for one disease label.

    For a single image:

        TP = prediction == ground truth

        Accuracy = 1 if correct else 0

    Precision / Recall / F1 are calculated for
    the predicted disease class.
    """

    if ground_truth is None:
        return None

    gt = normalize_label(
        ground_truth
    )

    pred = normalize_label(
        prediction
    )

    if pred is None:

        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "correct": False,
        }

    correct = (
        gt == pred
    )

    if correct:

        tp = 1
        fp = 0
        fn = 0
        tn = 0

    else:

        tp = 0
        fp = 1
        fn = 1
        tn = 0

    accuracy = (
        1.0 if correct else 0.0
    )

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
    }


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("CROPGUARD EVALUATION")
print("=" * 70)
print()

print("Base model:")
print(BASE_MODEL)
print()

print("LoRA model:")
print(LORA_MODEL)
print()

print("Image:")
print(IMAGE_PATH)
print()

print("Device:")
print(DEVICE)
print()


# ============================================================
# CHECK FILES
# ============================================================

print("===== STEP 1: CHECK FILES =====")

if not os.path.exists(BASE_MODEL):

    raise FileNotFoundError(
        f"Base model not found:\n{BASE_MODEL}"
    )

print("Base model path: OK")


if not os.path.exists(LORA_MODEL):

    raise FileNotFoundError(
        f"LoRA model not found:\n{LORA_MODEL}"
    )

print("LoRA path: OK")


if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

print("Image path: OK")

print()


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("===== STEP 2: LOAD PROCESSOR =====")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

print(
    "Processor:",
    type(processor)
)

print(
    "Image token:",
    getattr(
        processor,
        "image_token",
        None
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

print("PROCESSOR: OK")
print()


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("===== STEP 3: LOAD BASE MODEL =====")
print()

start_load = time.time()

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

base_load_time = (
    time.time() - start_load
)

print()
print(
    f"Base model loaded in "
    f"{base_load_time:.2f} seconds"
)

print("BASE MODEL: OK")
print()


# ============================================================
# MODEL STRUCTURE
# ============================================================

print("===== STEP 4: MODEL STRUCTURE =====")

try:

    language_model = (
        model.language_model
    )

    print(
        "language_model:",
        type(language_model)
    )

    print(
        "Has backbone:",
        hasattr(
            language_model,
            "backbone"
        )
    )

    if hasattr(
        language_model,
        "backbone"
    ):

        backbone = (
            language_model.backbone
        )

        print(
            "backbone:",
            type(backbone)
        )

        if hasattr(
            backbone,
            "layers"
        ):

            print(
                "Number of layers:",
                len(
                    backbone.layers
                )
            )

except Exception as e:

    print(
        "Could not inspect model structure:"
    )

    print(repr(e))

print()


# ============================================================
# LOAD REMAPPED LORA
# ============================================================

print("===== STEP 5: LOAD REMAPPED LORA =====")
print()

start_lora = time.time()

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL,
    is_trainable=False,
)

lora_load_time = (
    time.time() - start_lora
)

print(
    f"LoRA loaded in "
    f"{lora_load_time:.2f} seconds"
)

print(
    "LoRA MODEL: OK"
)

print()


# ============================================================
# ADAPTER STATUS
# ============================================================

print("===== STEP 6: ADAPTER STATUS =====")

try:

    print(
        "Active adapters:",
        model.active_adapters
    )

except Exception:

    print(
        "Active adapter:",
        "default"
    )

print()


# ============================================================
# COUNT LORA MODULES
# ============================================================

print("===== STEP 7: LORA MODULE CHECK =====")

lora_modules = []

for name, module in model.named_modules():

    if hasattr(
        module,
        "lora_A"
    ) and hasattr(
        module,
        "lora_B"
    ):

        lora_modules.append(
            name
        )

print(
    "LoRA modules found:",
    len(lora_modules)
)

if len(lora_modules) != 116:

    print(
        "WARNING: Expected 116 LoRA modules"
    )

else:

    print(
        "116 LoRA modules: OK"
    )

print()


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

print("===== STEP 8: PARAMETER STATUS =====")

trainable = 0
total = 0

for param in model.parameters():

    total += param.numel()

    if param.requires_grad:

        trainable += param.numel()

print(
    f"Trainable parameters: "
    f"{trainable:,}"
)

print(
    f"Total parameters: "
    f"{total:,}"
)

print(
    f"Trainable percentage: "
    f"{100 * trainable / total:.6f}%"
)

model.eval()

print(
    "Model evaluation mode: OK"
)

print()


# ============================================================
# LOAD IMAGE
# ============================================================

print("===== STEP 9: LOAD IMAGE =====")

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
print()


# ============================================================
# PROMPT
# ============================================================

prompt = """
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
Clearly state the most likely disease using the format:

Most likely disease: <disease name>
"""


# ============================================================
# PREPARE MULTIMODAL CHAT INPUT
# ============================================================

print("===== STEP 10: PREPARE MULTIMODAL INPUT =====")
print()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
]


print("Creating chat template...")

chat_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("Chat template created.")
print()

print("===== CHAT TEXT =====")
print(chat_text)
print()


# ============================================================
# VERIFY IMAGE TOKEN
# ============================================================

if "<image>" not in chat_text:

    raise RuntimeError(
        "ERROR: <image> token is missing "
        "from chat template"
    )

print(
    "Image token in chat text: PASSED"
)


# ============================================================
# PROCESS IMAGE + TEXT
# ============================================================

inputs = processor(
    images=image,
    text=chat_text,
    return_tensors="pt",
)

print(
    "Processor created multimodal inputs."
)


# ============================================================
# VERIFY IMAGE TOKEN ID
# ============================================================

image_token_id = getattr(
    processor,
    "image_token_id",
    None
)

if image_token_id is None:

    raise RuntimeError(
        "Processor does not provide image_token_id"
    )


has_image_token = (
    inputs["input_ids"]
    == image_token_id
).any().item()


print(
    f"Image token ID: {image_token_id}"
)

print(
    "Image token found in input_ids:",
    has_image_token
)

if not has_image_token:

    raise RuntimeError(
        "ERROR: No image token found "
        "in input_ids"
    )


# ============================================================
# MOVE INPUTS TO DEVICE
# ============================================================

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


print()
print("INPUTS: OK")
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

print()


# ============================================================
# GENERATION
# ============================================================

print("===== STEP 11: GENERATION =====")
print()

print("Generating response...")
print()

start_inference = time.time()

with torch.inference_mode():

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )

inference_time = (
    time.time() - start_inference
)

print(
    f"Generation completed."
)

print(
    f"Inference time: "
    f"{inference_time:.2f} seconds"
)

print()


# ============================================================
# DECODE
# ============================================================

print("===== STEP 12: DECODE RESPONSE =====")
print()

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


print("=" * 70)
print("CROPGUARD RESPONSE")
print("=" * 70)
print()
print(response)
print()
print("=" * 70)
print()


# ============================================================
# EXTRACT PREDICTION
# ============================================================

print("===== STEP 13: DISEASE PREDICTION =====")
print()

predicted_disease = extract_disease(
    response
)

if predicted_disease:

    print(
        "Predicted disease:",
        predicted_disease
    )

else:

    print(
        "Could not automatically extract "
        "a disease label."
    )

print()


# ============================================================
# AUTOMATED RESPONSE QUALITY PARAMETERS
# ============================================================

print("===== STEP 14: RESPONSE QUALITY =====")
print()

response_lower = response.lower()

quality_parameters = {

    "response_generated":
        len(response) > 20,

    "crop_identified":
        any(
            word in response_lower
            for word in [
                "crop",
                "tomato",
                "potato",
                "apple",
                "grape",
                "corn",
                "maize",
                "pepper",
                "rice",
            ]
        ),

    "disease_identified":
        predicted_disease is not None,

    "symptoms_provided":
        any(
            word in response_lower
            for word in [
                "symptom",
                "spot",
                "yellow",
                "yellowing",
                "lesion",
                "discolor",
                "blight",
                "mold",
                "rust",
            ]
        ),

    "immediate_action_provided":
        any(
            word in response_lower
            for word in [
                "remove",
                "apply",
                "spray",
                "treat",
                "fungicide",
                "bactericide",
                "isolate",
            ]
        ),

    "preventive_measures_provided":
        any(
            word in response_lower
            for word in [
                "prevent",
                "prevention",
                "rotate",
                "rotation",
                "monitor",
                "avoid",
                "sanitation",
            ]
        ),

    "image_recommendation_provided":
        any(
            word in response_lower
            for word in [
                "image",
                "photo",
                "picture",
                "leaf image",
            ]
        ),

    "confidence_provided":
        any(
            word in response_lower
            for word in [
                "confidence",
                "certain",
                "likely",
                "probable",
                "high confidence",
                "low confidence",
            ]
        ),
}


for name, value in quality_parameters.items():

    print(
        f"{name:35s}: "
        f"{'PASS' if value else 'FAIL'}"
    )

print()


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

print("===== STEP 15: CLASSIFICATION METRICS =====")
print()

if GROUND_TRUTH is None:

    print(
        "GROUND_TRUTH is currently None."
    )

    print()
    print(
        "Therefore classification metrics "
        "cannot be calculated yet."
    )

    print()
    print(
        "Set for example:"
    )

    print(
        'GROUND_TRUTH = "Tomato___Early_blight"'
    )

else:

    print(
        "Ground truth:",
        GROUND_TRUTH
    )

    print(
        "Prediction:",
        predicted_disease
    )

    metrics = calculate_metrics(
        GROUND_TRUTH,
        predicted_disease
    )

    print()

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{metrics['f1']:.4f}"
    )

    print(
        "Correct  :",
        metrics["correct"]
    )

print()


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)
print()

print(
    "Base model loading : PASSED"
)

print(
    "LoRA loading       : PASSED"
)

print(
    f"LoRA modules       : "
    f"{len(lora_modules)} / 116"
)

print(
    "Multimodal input   : PASSED"
)

print(
    "Image token        : PASSED"
)

print(
    "Generation         : PASSED"
)

print(
    f"Inference time     : "
    f"{inference_time:.2f} seconds"
)

if predicted_disease:

    print(
        "Disease prediction : PASSED"
    )

else:

    print(
        "Disease prediction : NOT EXTRACTED"
    )


if GROUND_TRUTH is not None:

    print()
    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score           : "
        f"{metrics['f1']:.4f}"
    )


print()
print("=" * 70)
print("CROPGUARD EVALUATION COMPLETE")
print("=" * 70)
print()
