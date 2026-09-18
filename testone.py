import os
import time
import traceback

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"

LORA_PATH = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model_remapped"

IMAGE_PATH = "/workspace/gsh-team03/CropGuard/images180/eval_cosmos3_180/Apple___Apple_scab/Apple___Apple_scab_019.jpg"

DEVICE = "cuda:0"

# Diagnostic only
MAX_NEW_TOKENS = 1024


# ============================================================
# HEADER
# ============================================================

print("=" * 88)
print("       CROPGUARD NEMOTRON ONE-IMAGE DIAGNOSTIC")
print("=" * 88)

print(f"Model : {MODEL_PATH}")
print(f"LoRA  : {LORA_PATH}")
print(f"Image : {IMAGE_PATH}")
print(f"Device: {DEVICE}")
print(f"Tokens: {MAX_NEW_TOKENS}")
print()


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        f"Evaluation image not found:\n{IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")


# ============================================================
# STEP 1: LOAD PROCESSOR
# ============================================================

print()
print("=" * 88)
print("STEP 1: LOAD PROCESSOR")
print("=" * 88)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("Processor loaded.")


# ============================================================
# STEP 2: LOAD BASE MODEL
# ============================================================

print()
print("=" * 88)
print("STEP 2: LOAD BASE MODEL")
print("=" * 88)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Base model loaded.")


# ============================================================
# STEP 3: LOAD LoRA
# ============================================================

print()
print("=" * 88)
print("STEP 3: LOAD LoRA")
print("=" * 88)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False,
)

model.eval()

print("LoRA loaded.")


# ============================================================
# STEP 4: SAFE MULTIMODAL GENERATION PATCH
# ============================================================

print()
print("=" * 88)
print("STEP 4: INSTALL SAFE MULTIMODAL GENERATION PATCH")
print("=" * 88)

multimodal_model = model.get_base_model()
inner_language_model = multimodal_model.language_model

original_inner_generate = inner_language_model.generate

STRIP_KEYS = (
    "num_patches",
    "num_tokens",
    "imgs_sizes",
)


def safe_inner_generate(*args, **kwargs):

    removed = []

    for key in STRIP_KEYS:

        if key in kwargs:

            kwargs.pop(key)

            removed.append(key)

    if removed:

        print(
            "SAFE GENERATION PATCH: removed inner-model metadata:",
            removed
        )

    return original_inner_generate(
        *args,
        **kwargs
    )


inner_language_model.generate = safe_inner_generate

print("Safe patch installed.")


# ============================================================
# STEP 5: PROMPT
# ============================================================

PROMPT = """
Analyze the plant image and identify the disease.

Choose exactly ONE disease from this list:

1. apple scab
2. common rust
3. northern leaf blight
4. black rot
5. septoria leaf spot
6. leaf blast

Provide your reasoning if needed, but make the FINAL ANSWER explicitly one of
the six labels above.
"""


messages = [
    {
        "role": "system",
        "content": (
            "You are an agricultural plant disease diagnosis assistant."
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
                "text": PROMPT
            },
        ],
    },
]


# ============================================================
# STEP 6: PROCESS IMAGE
# ============================================================

print()
print("=" * 88)
print("STEP 6: PROCESS IMAGE")
print("=" * 88)

print("Creating chat template...")

prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("Processing image...")

inputs = processor(
    text=prompt_text,
    images=[image],
    return_tensors="pt",
)

print("INPUTS: OK")

for key, value in inputs.items():

    if torch.is_tensor(value):

        if value.is_floating_point():

            inputs[key] = value.to(
                device=DEVICE,
                dtype=torch.bfloat16
            )

        else:

            inputs[key] = value.to(
                device=DEVICE
            )


# ============================================================
# GENERATION FUNCTION
# ============================================================

def run_pass(
    pass_name,
    lora_enabled,
):

    print()
    print("-" * 88)
    print(f"PASS: {pass_name}")
    print("-" * 88)

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    start = time.time()

    with torch.inference_mode():

        if lora_enabled:

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        else:

            with model.disable_adapter():

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    elapsed = time.time() - start

    input_len = inputs["input_ids"].shape[-1]

    generated_ids = output_ids[:, input_len:]

    raw_response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]

    clean_response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    print()
    print(f"Generation completed in {elapsed:.2f}s")
    print(f"Generated tokens: {generated_ids.shape[-1]}")

    print()
    print("RAW MODEL RESPONSE")
    print("=" * 88)

    print(raw_response)

    print("=" * 88)

    print()
    print("CLEAN MODEL RESPONSE")
    print("=" * 88)

    print(clean_response)

    print("=" * 88)

    return raw_response, clean_response


# ============================================================
# STEP 7: BASE
# ============================================================

try:

    base_raw, base_clean = run_pass(
        "BASE (LoRA disabled)",
        False,
    )


    # ========================================================
    # STEP 8: BASE + LoRA
    # ========================================================

    lora_raw, lora_clean = run_pass(
        "BASE + LoRA (adapter enabled)",
        True,
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 88)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 88)

    print()
    print("Ground truth: apple scab")

    print()
    print("Base response length :",
          len(base_clean))

    print("LoRA response length :",
          len(lora_clean))

    print()
    print("Do NOT run the 180-image evaluator yet.")

    print(
        "The complete responses above will be used to finalize "
        "the disease extraction logic."
    )


except KeyboardInterrupt:

    print()
    print("Interrupted by user.")
    raise


except Exception:

    print()
    print("ERROR DURING DIAGNOSTIC")
    print("=" * 88)

    traceback.print_exc()

    raise

