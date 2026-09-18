import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from peft import PeftModel

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

VALIDATION_JSONL = "/workspace/data/sft/validation.jsonl"

print("=" * 80)
print("CROPGUARD - TOMATO LATE BLIGHT VISUAL VALIDATION")
print("=" * 80)

# ------------------------------------------------------------------
# Load 5 Tomato Late Blight validation samples
# ------------------------------------------------------------------

samples = []

with open(VALIDATION_JSONL, "r") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        metadata = record.get("metadata", {})

        if (
            metadata.get("crop") == "Tomato"
            and metadata.get("disease") == "late_blight"
        ):
            image_path = None

            for msg in record.get("conversation", []):
                for content in msg.get("content", []):
                    if content.get("type") == "image":
                        image_path = content.get("image")
                        break
                if image_path:
                    break

            if image_path and os.path.exists(image_path):
                samples.append(
                    {
                        "image": image_path,
                        "ground_truth": "late_blight",
                        "source": metadata.get("source"),
                    }
                )

        if len(samples) >= 5:
            break

print(f"Validation samples selected: {len(samples)}")

for i, sample in enumerate(samples, 1):
    print(f"{i}. {sample['image']}")
    print(f"   Source: {sample['source']}")

if not samples:
    raise RuntimeError("No Tomato Late Blight validation images found.")

# ------------------------------------------------------------------
# Load processor
# ------------------------------------------------------------------

print("\n===== LOAD PROCESSOR =====")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    local_files_only=True,
)

print("Processor: OK")

# ------------------------------------------------------------------
# Load base multimodal model
# ------------------------------------------------------------------

print("\n===== LOAD BASE MODEL =====")

model = AutoModel.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

print("Base model: OK")

# ------------------------------------------------------------------
# Load validated remapped LoRA
# ------------------------------------------------------------------

print("\n===== LOAD VALIDATED LORA =====")

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL,
    is_trainable=False,
    local_files_only=True,
)

model.eval()

print("LoRA: OK")
print("Active adapters:", model.active_adapters)

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

try:
    device = model.device
except Exception:
    device = next(model.parameters()).device

print("Device:", device)

# ------------------------------------------------------------------
# Test each image
# ------------------------------------------------------------------

results = []

for idx, sample in enumerate(samples, 1):

    print("\n" + "=" * 80)
    print(f"TEST {idx}/{len(samples)}")
    print("=" * 80)
    print("Image:", sample["image"])
    print("Ground truth: late_blight")

    image = Image.open(sample["image"]).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": (
                        "You are CropGuard, an agricultural disease diagnosis assistant. "
                        "Analyze the attached tomato leaf image. "
                        "Identify the most likely disease. "
                        "Return the disease name exactly as one of these labels when applicable: "
                        "early_blight, late_blight, bacterial_spot, "
                        "septoria_leaf_spot, target_spot, leaf_mold, "
                        "yellow_leaf_curl_virus, mosaic_virus, healthy. "
                        "Give the crop and disease clearly."
                    ),
                },
            ],
        }
    ]

    # IMPORTANT:
    # Use the multimodal chat template so the image token is inserted.
    chat_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=chat_text,
        images=image,
        return_tensors="pt",
    )

    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    print("Input tensors:")
    for key, value in inputs.items():
        if torch.is_tensor(value):
            print(f"  {key}: {tuple(value.shape)}")

    # --------------------------------------------------------------
    # IMPORTANT:
    # First inspect the model's accepted generation interface.
    # --------------------------------------------------------------

    with torch.inference_mode():

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )

        except Exception as e:
            print("\nGENERATION FAILED")
            print(type(e).__name__, str(e))
            print(
                "\nThis is an inference-interface problem, "
                "not evidence that the LoRA failed."
            )
            raise

    # --------------------------------------------------------------
    # Decode
    # --------------------------------------------------------------

    try:
        text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]
    except Exception:
        text = processor.decode(
            outputs[0],
            skip_special_tokens=True,
        )

    print("\nMODEL OUTPUT:")
    print(text)

    normalized = text.lower().replace("-", "_").replace(" ", "_")

    prediction = None

    # Prefer late_blight if explicitly present.
    if "late_blight" in normalized:
        prediction = "late_blight"
    elif "early_blight" in normalized:
        prediction = "early_blight"
    elif "bacterial_spot" in normalized:
        prediction = "bacterial_spot"
    elif "septoria_leaf_spot" in normalized:
        prediction = "septoria_leaf_spot"
    elif "target_spot" in normalized:
        prediction = "target_spot"
    elif "leaf_mold" in normalized:
        prediction = "leaf_mold"
    elif "yellow_leaf_curl_virus" in normalized:
        prediction = "yellow_leaf_curl_virus"
    elif "mosaic_virus" in normalized:
        prediction = "mosaic_virus"
    elif "healthy" in normalized:
        prediction = "healthy"

    correct = prediction == "late_blight"

    print("\nPREDICTED:", prediction)
    print("RESULT:", "PASS" if correct else "FAIL")

    results.append(
        {
            "image": sample["image"],
            "prediction": prediction,
            "correct": correct,
        }
    )

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

correct = sum(r["correct"] for r in results)
total = len(results)

print(f"Correct: {correct}/{total}")
print(f"Accuracy: {100.0 * correct / total:.2f}%")

for r in results:
    print(
        f"{'PASS' if r['correct'] else 'FAIL'} | "
        f"{os.path.basename(r['image'])} | "
        f"{r['prediction']}"
    )

print("=" * 80)
