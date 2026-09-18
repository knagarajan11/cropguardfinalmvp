# CropGuard — Codex Agent Instructions

## Project
- Project root: /home/gsh-ndhmy/CropGuard
- Container workspace: /workspace
- `/workspace` is bound to the CropGuard project root.

## Compute Environment
- Cluster: NVIDIA DGX
- Current compute node is obtained through Slurm.
- Slurm partition: hackathon
- GPU: NVIDIA B300 SXM6 AC
- Standard GPU allocation:
  `srun -p hackathon --gres=gpu:b300:1 --cpus-per-task=16 --pty bash`

## Container Environment
- Known-working Apptainer sandbox:
  `/home/gsh-ndhmy/nemo_sandbox`
- Enter the sandbox from the CropGuard root with:

  apptainer shell --nv \
    --bind "$PWD:/workspace" \
    --pwd /workspace \
    "$HOME/nemo_sandbox"

- Work inside `/workspace` when running CropGuard code in the sandbox.

## Python
- Python executable inside the sandbox:
  `/opt/venv/bin/python`
- Python version: 3.12.3
- PyTorch is known to work with CUDA and the NVIDIA B300.

## Known Working GPU Environment
- PyTorch:
  `2.12.0a0+0291f960b6.nv26.04.48445190`
- CUDA is available.
- GPU:
  `NVIDIA B300 SXM6 AC`

## CropGuard Workflow
- Preserve the existing CropGuard implementation and working LoRA evaluation workflow.
- Reuse existing scripts, datasets, checkpoints, models, and configurations before creating replacements.
- Prefer incremental changes over large rewrites.
- Validate changes with small tests before running expensive GPU workloads.

## Important Existing Resources
- Existing NVIDIA/NeMo container:
  `/home/gsh-ndhmy/CropGuard/nemo-automodel.sif`
- Existing sandbox:
  `/home/gsh-ndhmy/nemo_sandbox`
- Hugging Face cache, checkpoints, outputs, and LoRA artifacts may be large and must be preserved.

## Safety Rules
- Do NOT delete or recreate `nemo_sandbox`.
- Do NOT replace or rebuild `nemo-automodel.sif` unless explicitly requested.
- Do NOT delete checkpoints, LoRA artifacts, Hugging Face cache, datasets, outputs, or existing working scripts.
- Do NOT install or replace CUDA, PyTorch, Transformers, or NeMo unnecessarily.
- Reuse the existing working environment and dependencies whenever possible.
- Do NOT expose, print, commit, or hard-code API keys, tokens, passwords, or credentials.
- Never commit secrets to Git.

## GPU Execution
- GPU-intensive training/evaluation should run on a Slurm GPU allocation.
- Do not assume a GPU is available on the login node.
- Before expensive execution, verify:
  `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`

## Development Style
- Inspect existing code before modifying it.
- Keep changes focused on the requested task.
- Explain significant changes.
- Preserve backward compatibility with existing CropGuard workflows where practical.
- Do not remove working functionality merely to simplify the implementation.
