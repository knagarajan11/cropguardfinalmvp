import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"

LORA = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model"

print("======================================")
print("1. LOAD BASE MODEL")
print("======================================")

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.eval()

print("BASE MODEL: OK")

print("\n======================================")
print("2. LOAD TOKENIZER")
print("======================================")

tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True,
)

print("TOKENIZER: OK")

print("\n======================================")
print("3. LOAD MAPPED LORA")
print("======================================")

# PEFT expects the original adapter paths.
# We first construct the PEFT model and then rename the
# target modules to match Nemotron's backbone structure.

from peft import PeftConfig

config = PeftConfig.from_pretrained(LORA)

print("Original target modules:", len(config.target_modules))

# Convert:
# language_model.model.layers.X...
#
# to:
# language_model.backbone.layers.X...

mapped_targets = []

for target in config.target_modules:
    target = target.replace(
        "language_model.model.layers.",
        "language_model.backbone.layers."
    )
    mapped_targets.append(target)

config.target_modules = mapped_targets

print("Mapped target modules:", len(config.target_modules))

model = PeftModel.from_pretrained(
    model,
    LORA,
    config=config,
    is_trainable=False,
)

model.eval()

print("LORA MODEL: OK")

print("\n======================================")
print("4. CHECK LORA MODULES")
print("======================================")

lora_modules = []

for name, module in model.named_modules():
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        lora_modules.append(name)

print("LoRA modules found:", len(lora_modules))

if len(lora_modules) != 116:
    raise RuntimeError(
        f"Expected 116 LoRA modules but found {len(lora_modules)}"
    )

print("ALL 116 LORA MODULES PRESENT")

print("\n======================================")
print("5. CHECK LORA WEIGHT MAGNITUDES")
print("======================================")

nonzero = 0
total = 0

for name, module in model.named_modules():

    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

        total += 1

        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight

        a_norm = A.float().norm().item()
        b_norm = B.float().norm().item()

        if a_norm > 0 and b_norm > 0:
            nonzero += 1

print("Total LoRA modules :", total)
print("Non-zero A/B pairs :", nonzero)

if nonzero == 0:
    raise RuntimeError("LoRA weights appear to be zero!")

print("LoRA WEIGHTS ARE NON-ZERO")

print("\n======================================")
print("6. TEST INFERENCE")
print("======================================")

prompt = """
A farmer observes yellow spots and brown lesions on the leaves of a tomato plant.
The leaves are becoming dry and curling.
Identify the likely plant disease and provide recommended actions.
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
    )

text = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\n======================================")
print("CROPGUARD OUTPUT")
print("======================================")

print(text)

print("\n======================================")
print("FINAL RESULT")
print("======================================")

print("Base model loaded       : PASS")
print("LoRA modules             : PASS")
print("116 modules present      : PASS")
print("LoRA weights non-zero    : PASS")
print("Inference completed      : PASS")

print("\n===== LORA INFERENCE TEST SUCCESS =====")
