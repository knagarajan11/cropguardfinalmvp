import os
import shutil
from safetensors.torch import load_file, save_file

SRC = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model"

DST = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model_remapped"

os.makedirs(DST, exist_ok=True)

src_weights = os.path.join(SRC, "adapter_model.safetensors")
dst_weights = os.path.join(DST, "adapter_model.safetensors")

print("=" * 70)
print("REMAPPING LORA CHECKPOINT")
print("=" * 70)

print("SOURCE:")
print(src_weights)

print("\nDESTINATION:")
print(dst_weights)

# ------------------------------------------------------------
# Load original LoRA tensors
# ------------------------------------------------------------

state = load_file(src_weights)

print("\nOriginal tensors:", len(state))

# ------------------------------------------------------------
# Rename model.layers -> backbone.layers
# ------------------------------------------------------------

new_state = {}

changed = 0
unchanged = 0

for key, tensor in state.items():

    new_key = key.replace(
        "language_model.model.layers.",
        "language_model.backbone.layers."
    )

    if new_key != key:
        changed += 1
    else:
        unchanged += 1

    if new_key in new_state:
        raise RuntimeError(
            f"Duplicate key after remapping:\n{new_key}"
        )

    new_state[new_key] = tensor

print("Changed keys:", changed)
print("Unchanged keys:", unchanged)
print("New tensors:", len(new_state))

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

save_file(
    new_state,
    dst_weights
)

# ------------------------------------------------------------
# Copy adapter config
# ------------------------------------------------------------

src_config = os.path.join(SRC, "adapter_config.json")
dst_config = os.path.join(DST, "adapter_config.json")

shutil.copy2(src_config, dst_config)

print("\nSaved:")
print(dst_weights)

print("\nCopied:")
print(dst_config)

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

verify = load_file(dst_weights)

print("\n===== VERIFICATION =====")

print("Verification tensors:", len(verify))

bad = [
    k for k in verify
    if "language_model.model.layers." in k
]

good = [
    k for k in verify
    if "language_model.backbone.layers." in k
]

print("Old layer keys remaining:", len(bad))
print("Backbone layer keys:", len(good))

if len(verify) != len(state):
    raise RuntimeError("Tensor count changed!")

if len(bad) != 0:
    raise RuntimeError("Old layer names still exist!")

if len(good) == 0:
    raise RuntimeError("No remapped backbone keys found!")

print("\nSUCCESS: CHECKPOINT REMAPPED")
