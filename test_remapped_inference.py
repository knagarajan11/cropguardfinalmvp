import os
import sys
import torch

from transformers import AutoProcessor, AutoModel
from peft import PeftModel


# ============================================================
# PATHS
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

print("=" * 70)
print("CROPGUARD REMAPPED LORA - INFERENCE TEST")
print("=" * 70)

print(f"BASE MODEL : {BASE_MODEL}")
print(f"LORA MODEL : {LORA_MODEL}")
print()


# ============================================================
# SAFETY CHECKS
# ============================================================

print("===== STEP 0: PATH CHECK =====")

if not os.path.exists(BASE_MODEL):
    raise FileNotFoundError(
        f"Base model not found:\n{BASE_MODEL}"
    )

if not os.path.exists(LORA_MODEL):
    raise FileNotFoundError(
        f"Remapped LoRA not found:\n{LORA_MODEL}"
    )

adapter_config = os.path.join(
    LORA_MODEL,
    "adapter_config.json"
)

adapter_weights = os.path.join(
    LORA_MODEL,
    "adapter_model.safetensors"
)

if not os.path.exists(adapter_config):
    raise FileNotFoundError(
        f"adapter_config.json not found:\n{adapter_config}"
    )

if not os.path.exists(adapter_weights):
    raise FileNotFoundError(
        f"adapter_model.safetensors not found:\n{adapter_weights}"
    )

print("Base model : FOUND")
print("LoRA config: FOUND")
print("LoRA weights: FOUND")
print()


# ============================================================
# GPU INFORMATION
# ============================================================

print("===== GPU =====")

if torch.cuda.is_available():
    print("CUDA available : YES")
    print("GPU count      :", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i} : "
            f"{torch.cuda.get_device_name(i)}"
        )

    device = "cuda"
else:
    print("CUDA available : NO")
    device = "cpu"

print()


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("===== STEP 1: LOAD PROCESSOR =====")

try:
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("PROCESSOR: OK")

except Exception as e:
    print("[WARNING] AutoProcessor loading failed:")
    print(repr(e))
    processor = None

print()


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("===== STEP 2: LOAD BASE MODEL =====")

try:

    if device == "cuda":

        base_model = AutoModel.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    else:

        base_model = AutoModel.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            local_files_only=True,
        )

    print("BASE MODEL: OK")

except Exception as e:

    print()
    print("BASE MODEL LOAD FAILED")
    print(repr(e))
    raise

print()


# ============================================================
# MODEL STRUCTURE CHECK
# ============================================================

print("===== STEP 3: MODEL STRUCTURE =====")

try:

    language_model = base_model.language_model

    print(
        "language_model type:",
        type(language_model)
    )

    print(
        "Has backbone:",
        hasattr(language_model, "backbone")
    )

    if hasattr(language_model, "backbone"):

        print(
            "backbone type:",
            type(language_model.backbone)
        )

        print(
            "Number of layers:",
            len(language_model.backbone.layers)
        )

except Exception as e:

    print("[WARNING] Structure check failed:")
    print(repr(e))

print()


# ============================================================
# LOAD REMAPPED LORA
# ============================================================

print("===== STEP 4: LOAD REMAPPED LORA =====")

try:

    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL,
        is_trainable=False,
        local_files_only=True,
    )

    print("REMAPPED LORA: OK")

except Exception as e:

    print()
    print("REMAPPED LORA LOAD FAILED")
    print(repr(e))
    raise

print()


# ============================================================
# ADAPTER STATUS
# ============================================================

print("===== STEP 5: ADAPTER STATUS =====")

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

print()


# ============================================================
# COUNT LORA MODULES
# ============================================================

print("===== STEP 6: LORA MODULE CHECK =====")

lora_modules = []

for name, module in model.named_modules():

    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        lora_modules.append(name)

print(
    "LoRA modules found:",
    len(lora_modules)
)

if len(lora_modules) != 116:

    print()
    print("WARNING:")
    print(
        f"Expected 116 LoRA modules, "
        f"found {len(lora_modules)}"
    )

else:

    print("116 LoRA modules: OK")

print()


# ============================================================
# TRAINABLE PARAMETER CHECK
# ============================================================

print("===== STEP 7: TRAINABLE PARAMETER CHECK =====")

trainable = 0
total = 0

for param in model.parameters():

    total += param.numel()

    if param.requires_grad:
        trainable += param.numel()

print("Trainable parameters:", trainable)
print("Total parameters    :", total)

if trainable == 0:
    print("Inference mode: OK")
else:
    print(
        "WARNING: trainable parameters detected:",
        trainable
    )

print()


# ============================================================
# EVALUATION MODE
# ============================================================

print("===== STEP 8: EVAL MODE =====")

model.eval()

print("Model set to evaluation mode")
print()


# ============================================================
# GENERATION TEST
# ============================================================

print("===== STEP 9: GENERATION TEST =====")

prompt = """
You are CropGuard, an AI assistant for crop disease detection
and agricultural advisory.

A farmer reports that the leaves of a crop have visible spots,
yellowing and discoloration.

Analyze the situation and provide:
1. Possible crop disease or causes
2. Recommended immediate action
3. Preventive measures
4. Whether the farmer should provide a leaf image for further diagnosis

Give a concise agricultural recommendation.
"""

print("PROMPT:")
print(prompt)
print()


# ------------------------------------------------------------
# Find the correct device
# ------------------------------------------------------------

try:

    if hasattr(model, "device"):
        input_device = model.device
    else:
        input_device = next(model.parameters()).device

except Exception:

    input_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

print("Input device:", input_device)
print()


# ------------------------------------------------------------
# Prepare text input
# ------------------------------------------------------------

if processor is None:

    print("PROCESSOR unavailable.")
    print("Skipping generation test.")

else:

    try:

        inputs = processor(
            text=prompt,
            return_tensors="pt",
        )

        # Move tensors to the model's execution device.
        for key in inputs:

            if torch.is_tensor(inputs[key]):

                inputs[key] = inputs[key].to(
                    input_device
                )

        print("Inputs prepared: OK")

    except Exception as e:

        print()
        print("INPUT PREPARATION FAILED")
        print(repr(e))
        raise

    print()


    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    try:

        print("Generating response...")
        print()

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
            )

        print("Generation: SUCCESS")
        print()

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        try:

            generated_text = processor.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]

        except Exception:

            generated_text = processor.decode(
                outputs[0],
                skip_special_tokens=True,
            )

        print("=" * 70)
        print("MODEL OUTPUT")
        print("=" * 70)
        print(generated_text)
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("GENERATION FAILED")
        print("=" * 70)
        print(repr(e))
        print()
        print("The LoRA successfully loaded, but the generation")
        print("interface of this multimodal Nemotron model may")
        print("require its specific processor/chat-template API.")
        print()
        raise


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)
print("Base model loading : PASSED")
print("LoRA loading       : PASSED")
print("LoRA modules       :", len(lora_modules), "/ 116")
print("Model evaluation   : PASSED")
print("=" * 70)
