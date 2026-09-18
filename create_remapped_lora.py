import os
import json
import shutil
from safetensors.torch import load_file, save_file

SRC = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model"

DST = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model_remapped"

print("===== CREATE REMAPPED LORA =====")
print("SOURCE:", SRC)
print("DEST  :", DST)

# Safety check
if os.path.exists(DST):
    raise RuntimeError(
        f"Destination already exists: {DST}\n"
        "Refusing to overwrite it."
    )

os.makedirs(DST)

# ---------------------------------------------------------
# 1. Load original adapter config
# ---------------------------------------------------------
src_config = os.path.join(SRC, "adapter_config.json")

with open(src_config, "r") as f:
    config = json.load(f)

print()
print("Original target modules:", len(config["target_modules"]))

# ---------------------------------------------------------
# 2. Remap target_modules
# ---------------------------------------------------------
old_prefix = "language_model.model.layers."
new_prefix = "language_model.backbone.layers."

new_targets = []

for target in config["target_modules"]:
    if target.startswith(old_prefix):
        target = new_prefix + target[len(old_prefix):]

    new_targets.append(target)

config["target_modules"] = new_targets

print("Remapped target modules:", len(new_targets))

# Verify no old paths remain
bad_targets = [
    x for x in new_targets
    if "language_model.model.layers." in x
]

if bad_targets:
    raise RuntimeError(
        f"Old target paths still present: {bad_targets}"
    )

# ---------------------------------------------------------
# 3. Save remapped config
# ---------------------------------------------------------
dst_config = os.path.join(DST, "adapter_config.json")

with open(dst_config, "w") as f:
    json.dump(config, f, indent=2)

print("Saved:", dst_config)

# ---------------------------------------------------------
# 4. Load LoRA tensors
# ---------------------------------------------------------
src_weights = os.path.join(SRC, "adapter_model.safetensors")

print()
print("Loading LoRA tensors...")

state = load_file(src_weights)

print("Original tensor count:", len(state))

# ---------------------------------------------------------
# 5. Remap tensor names
# ---------------------------------------------------------
new_state = {}

for key, value in state.items():

    new_key = key

    if old_prefix in new_key:
        new_key = new_key.replace(
            old_prefix,
            new_prefix
        )

    if new_key in new_state:
        raise RuntimeError(
            f"Duplicate tensor after remapping:\n"
            f"{new_key}"
        )

    new_state[new_key] = value

print("Remapped tensor count:", len(new_state))

# ---------------------------------------------------------
# 6. Verify tensor counts
# ---------------------------------------------------------
if len(state) != len(new_state):
    raise RuntimeError(
        f"Tensor count changed: "
        f"{len(state)} -> {len(new_state)}"
    )

# ---------------------------------------------------------
# 7. Verify all tensors were actually remapped
# ---------------------------------------------------------
old_count = sum(
    "language_model.model.layers." in k
    for k in state.keys()
)

new_count = sum(
    "language_model.backbone.layers." in k
    for k in new_state.keys()
)

print()
print("Original layer-path tensors :", old_count)
print("Remapped layer-path tensors :", new_count)

if old_count != new_count:
    raise RuntimeError(
        "Not all layer tensors were remapped correctly."
    )

# ---------------------------------------------------------
# 8. Save remapped weights
# ---------------------------------------------------------
dst_weights = os.path.join(
    DST,
    "adapter_model.safetensors"
)

save_file(new_state, dst_weights)

print()
print("Saved:", dst_weights)

# ---------------------------------------------------------
# 9. Final validation
# ---------------------------------------------------------
print()
print("===== FINAL VALIDATION =====")

with open(dst_config, "r") as f:
    check_config = json.load(f)

print(
    "Config target modules:",
    len(check_config["target_modules"])
)

check_state = load_file(dst_weights)

print(
    "Weight tensors:",
    len(check_state)
)

old_remaining_config = [
    x for x in check_config["target_modules"]
    if "language_model.model.layers." in x
]

old_remaining_weights = [
    x for x in check_state.keys()
    if "language_model.model.layers." in x
]

new_targets_count = sum(
    "language_model.backbone.layers." in x
    for x in check_config["target_modules"]
)

new_weights_count = sum(
    "language_model.backbone.layers." in x
    for x in check_state.keys()
)

print("Old config paths remaining:", len(old_remaining_config))
print("Old weight paths remaining:", len(old_remaining_weights))
print("New config paths:", new_targets_count)
print("New weight paths:", new_weights_count)

if old_remaining_config:
    raise RuntimeError("Old paths remain in adapter_config.json")

if old_remaining_weights:
    raise RuntimeError("Old paths remain in adapter_model.safetensors")

if new_targets_count != 116:
    raise RuntimeError(
        f"Expected 116 target modules, got {new_targets_count}"
    )

if new_weights_count != 232:
    raise RuntimeError(
        f"Expected 232 tensors, got {new_weights_count}"
    )

print()
print("======================================")
print("REMAP SUCCESSFUL")
print("======================================")
print("116 target modules : OK")
print("232 LoRA tensors   : OK")
print("Original checkpoint: UNTOUCHED")
print("======================================")
