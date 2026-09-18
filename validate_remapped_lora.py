import os
import torch
from safetensors.torch import load_file

SRC = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model"
DST = "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model_remapped"

SRC_FILE = os.path.join(SRC, "adapter_model.safetensors")
DST_FILE = os.path.join(DST, "adapter_model.safetensors")

print("===== VALIDATE REMAPPED LORA =====")

src = load_file(SRC_FILE)
dst = load_file(DST_FILE)

print(f"Original tensors : {len(src)}")
print(f"Remapped tensors : {len(dst)}")

# --------------------------------------------------
# 1. Check tensor count
# --------------------------------------------------

if len(src) != 232:
    raise RuntimeError(f"Unexpected source tensor count: {len(src)}")

if len(dst) != 232:
    raise RuntimeError(f"Unexpected destination tensor count: {len(dst)}")

print("Tensor count: OK")

# --------------------------------------------------
# 2. Convert old paths -> new paths
# --------------------------------------------------

def remap_key(k):
    return k.replace(
        "base_model.model.language_model.model.layers.",
        "base_model.model.language_model.backbone.layers."
    )

expected_dst = {remap_key(k): v for k, v in src.items()}

# --------------------------------------------------
# 3. Check every tensor exists
# --------------------------------------------------

missing = []
extra = []

for k in expected_dst:
    if k not in dst:
        missing.append(k)

for k in dst:
    if k not in expected_dst:
        extra.append(k)

print(f"Missing tensors : {len(missing)}")
print(f"Extra tensors   : {len(extra)}")

if missing:
    print("\nFIRST MISSING:")
    for k in missing[:10]:
        print(k)
    raise RuntimeError("Missing remapped tensors")

if extra:
    print("\nFIRST EXTRA:")
    for k in extra[:10]:
        print(k)
    raise RuntimeError("Unexpected remapped tensors")

print("Tensor paths: OK")

# --------------------------------------------------
# 4. Check shapes
# --------------------------------------------------

shape_errors = []

for k, original_tensor in expected_dst.items():
    if dst[k].shape != original_tensor.shape:
        shape_errors.append(
            (k, original_tensor.shape, dst[k].shape)
        )

print(f"Shape errors   : {len(shape_errors)}")

if shape_errors:
    for item in shape_errors[:10]:
        print(item)
    raise RuntimeError("Tensor shape mismatch")

print("Tensor shapes: OK")

# --------------------------------------------------
# 5. Check actual tensor values
# --------------------------------------------------

value_errors = []
max_diff = 0.0

for k, original_tensor in expected_dst.items():

    a = original_tensor.float()
    b = dst[k].float()

    diff = torch.max(torch.abs(a - b)).item()
    max_diff = max(max_diff, diff)

    if not torch.equal(original_tensor, dst[k]):
        value_errors.append((k, diff))

print(f"Value mismatches : {len(value_errors)}")
print(f"Maximum diff     : {max_diff}")

if value_errors:
    print("\nFIRST VALUE MISMATCHES:")
    for k, diff in value_errors[:10]:
        print(k, "diff =", diff)

    raise RuntimeError("Remapped tensor values do not match original")

print("Tensor values: EXACT MATCH")

# --------------------------------------------------
# 6. Count zero / non-zero B matrices
# --------------------------------------------------

zero_b = []
nonzero_b = []

for k, tensor in dst.items():

    if not k.endswith(".lora_B.weight"):
        continue

    norm = torch.linalg.vector_norm(tensor.float()).item()

    if norm == 0:
        zero_b.append((k, norm))
    else:
        nonzero_b.append((k, norm))

print()
print("===== B MATRIX STATUS =====")
print(f"Total B matrices : {len(zero_b) + len(nonzero_b)}")
print(f"Non-zero B       : {len(nonzero_b)}")
print(f"Zero B           : {len(zero_b)}")

if zero_b:
    print("\nZero B matrices:")
    for k, norm in zero_b:
        print(k, "| norm:", norm)

# --------------------------------------------------
# FINAL
# --------------------------------------------------

print()
print("======================================")
print("REMAP VALIDATION SUCCESSFUL")
print("======================================")
print("116 LoRA modules : OK")
print("232 tensors      : OK")
print("Shapes           : OK")
print("Values           : EXACT MATCH")
print("Zero B matrices  : ALLOWED")
print("Original adapter : UNTOUCHED")
print("======================================")
