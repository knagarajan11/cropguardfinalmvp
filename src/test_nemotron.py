import os
import time
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModel


# ============================================================
# CropGuard - Nemotron Omni Baseline Test
# ============================================================

MODEL_DIR = (
    "/home/gsh-ndhmy/CropGuard/hf_cache/hub/"
    "models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/"
    "snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"
)

IMAGE_PATH = "/workspace/data/test_images/plant.jpg"


PROMPT = """
Analyze this tomato plant image for possible disease.

Identify:
1. Crop: tomato
2. Most likely disease or plant health condition
3. Visual symptoms observed in the image
4. Confidence level
5. Possible causes
6. Organic treatment recommendations
7. Preventive measures

Focus only on what can reasonably be inferred from the image.
If the image is insufficient to identify a disease confidently,
say so clearly and explain what additional information would help.
"""


def print_gpu_memory(label):
    """Print current GPU memory usage."""

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print(f"{label}")
        print(f"  GPU allocated: {allocated:.2f} GB")
        print(f"  GPU reserved:  {reserved:.2f} GB")


def main():

    total_start = time.time()

    print("=" * 70)
    print("CropGuard - Nemotron Omni Baseline Test")
    print("=" * 70)

    # ========================================================
    # 1. Environment
    # ========================================================

    print("\n[1] ENVIRONMENT")
    print("-" * 70)

    if not torch.cuda.is_available():

        print("ERROR: CUDA is not available.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory

    print("GPU:")
    print(gpu_name)

    print(
        f"GPU Memory: "
        f"{gpu_memory / (1024 ** 3):.1f} GB"
    )

    print("CUDA Available: True")
    print("PyTorch:", torch.__version__)

    print_gpu_memory("\nInitial GPU memory:")

    # ========================================================
    # 2. Verify model directory
    # ========================================================

    print("\n[2] MODEL DIRECTORY")
    print("-" * 70)

    print(MODEL_DIR)

    if not os.path.exists(MODEL_DIR):

        print("ERROR: Model directory does not exist.")
        return

    print("Model directory exists.")

    # ========================================================
    # 3. Verify image
    # ========================================================

    print("\n[3] IMAGE")
    print("-" * 70)

    print("Image:")
    print(IMAGE_PATH)

    if not os.path.exists(IMAGE_PATH):

        print("ERROR: Image does not exist.")
        return

    image = Image.open(IMAGE_PATH).convert("RGB")

    print("Image loaded successfully.")
    print("Image size:", image.size)
    print("Image mode:", image.mode)

    # ========================================================
    # 4. Load processor
    # ========================================================

    print("\n[4] LOADING PROCESSOR")
    print("-" * 70)

    processor_start = time.time()

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    processor_time = time.time() - processor_start

    print("Processor loaded.")
    print(f"Processor load time: {processor_time:.2f} sec")

    # ========================================================
    # 5. Load model
    # ========================================================

    print("\n[5] LOADING NEMOTRON MODEL")
    print("-" * 70)

    print("Model:")
    print("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")

    print("\nLoading model weights...")
    print("This can take several minutes on the first load.")
    print()

    model_start = time.time()

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model_time = time.time() - model_start

    print("\nModel loaded successfully.")

    print(f"Model load time: {model_time:.2f} sec")

    print_gpu_memory("\nGPU memory after model load:")

    # ========================================================
    # 6. Prepare multimodal conversation
    # ========================================================

    print("\n[6] PREPARING INPUT")
    print("-" * 70)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    print("Preparing inputs...")

    input_start = time.time()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    print("\nINPUT KEYS BEFORE CLEANUP:")
    print(list(inputs.keys()))

    # ========================================================
    # IMPORTANT FIX
    #
    # The Nemotron wrapper passes unknown kwargs to the
    # underlying language_model.generate().
    #
    # These metadata fields cause:
    #
    # ValueError:
    # The following model_kwargs are not used by the model:
    # ['num_patches', 'num_tokens', 'imgs_sizes']
    #
    # Remove them before calling model.generate().
    # ========================================================

    for key in [
        "num_patches",
        "num_tokens",
        "imgs_sizes",
    ]:
        inputs.pop(key, None)

    print("\nINPUT KEYS AFTER CLEANUP:")
    print(list(inputs.keys()))

    # ========================================================
    # Move tensors to GPU
    # ========================================================

    for key, value in inputs.items():

        if torch.is_tensor(value):

            inputs[key] = value.to(
                model.device
            )

    input_time = time.time() - input_start

    print("\nInputs prepared.")
    print(f"Input preparation time: {input_time:.2f} sec")

    print_gpu_memory("\nGPU memory before inference:")

    # ========================================================
    # 7. Run inference
    # ========================================================

    print("\n[7] RUNNING INFERENCE")
    print("-" * 70)

    print("Prompt:")
    print(PROMPT.strip())

    print("\nGenerating response...")
    print("Please wait...")
    print()

    inference_start = time.time()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    inference_time = time.time() - inference_start

    print("\nInference completed.")

    print(
        f"Inference time: "
        f"{inference_time:.2f} sec"
    )

    print_gpu_memory("\nGPU memory after inference:")

    # ========================================================
    # 8. Decode response
    # ========================================================

    print("\n[8] DECODING RESPONSE")
    print("-" * 70)

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[
        :,
        input_length:
    ]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # ========================================================
    # 9. Display result
    # ========================================================

    print("\n")
    print("=" * 70)
    print("NEMOTRON RESULT")
    print("=" * 70)

    print(response.strip())

    print("=" * 70)

    # ========================================================
    # 10. Benchmark summary
    # ========================================================

    total_time = time.time() - total_start

    print("\n[9] BENCHMARK SUMMARY")
    print("-" * 70)

    print(
        f"Processor load:     {processor_time:.2f} sec"
    )

    print(
        f"Model load:         {model_time:.2f} sec"
    )

    print(
        f"Input preparation:  {input_time:.2f} sec"
    )

    print(
        f"Inference:          {inference_time:.2f} sec"
    )

    print(
        f"Total execution:    {total_time:.2f} sec"
    )

    print("=" * 70)
    print("CropGuard Nemotron test completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
