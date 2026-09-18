import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE = "/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"

ADAPTER = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model_remapped"

print("=" * 70)
print("TESTING REMAPPED CROPGUARD LORA")
print("=" * 70)

print("\n===== LOAD BASE =====")

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("BASE MODEL: OK")

print("\n===== LOAD REMAPPED LORA =====")

model = PeftModel.from_pretrained(
    model,
    ADAPTER,
    is_trainable=False,
)

print("LORA MODEL: OK")

print("\n===== ADAPTER STATUS =====")

print("Active adapters:", model.active_adapters)

print("\n===== CHECK LORA WEIGHTS =====")

count = 0
nonzero = 0
zero = 0

for name, module in model.named_modules():

    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

        count += 1

        try:
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight

            a_norm = A.float().norm().item()
            b_norm = B.float().norm().item()

            if a_norm > 0 and b_norm > 0:
                nonzero += 1
            else:
                zero += 1

            if count <= 10:
                print(
                    name,
                    "| A:", tuple(A.shape),
                    "| B:", tuple(B.shape),
                    "| A norm:", a_norm,
                    "| B norm:", b_norm
                )

        except Exception as e:
            print("ERROR:", name, e)

print("\n===== RESULT =====")

print("LoRA modules:", count)
print("Non-zero A/B pairs:", nonzero)
print("Zero A/B pairs:", zero)

if count != 116:
    raise RuntimeError(
        f"Expected 116 LoRA modules, found {count}"
    )

if nonzero != 116:
    raise RuntimeError(
        f"Expected 116 non-zero A/B pairs, found {nonzero}"
    )

print("\nSUCCESS")
print("ALL 116 LORA MODULES HAVE TRAINED WEIGHTS")
print("=" * 70)
