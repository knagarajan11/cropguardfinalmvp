from PIL import Image
from nemo_automodel.components.datasets.vlm.collate_fns import (
    nemotron_omni_collate_fn,
)


def cropguard_nemotron_collate_fn(
    examples,
    processor,
    max_length=None,
    max_video_frames=8,
):
    """
    CropGuard collator.

    Converts image paths to RGB PIL images and resizes
    all images to a common 512x512 resolution before
    passing them to NVIDIA's Nemotron Omni collator.
    """

    fixed_examples = []

    for example in examples:
        conversation = []

        for message in example["conversation"]:
            new_message = dict(message)

            content = message.get("content")

            if isinstance(content, list):
                new_content = []

                for item in content:
                    item = dict(item)

                    if item.get("type") == "image":
                        image_path = item.get("image")

                        if isinstance(image_path, str):
                            image = Image.open(image_path).convert("RGB")

                            # IMPORTANT:
                            # Normalize all images to the same size.
                            image = image.resize(
                                (512, 512),
                                Image.Resampling.LANCZOS,
                            )

                            item["image"] = image

                    new_content.append(item)

                new_message["content"] = new_content

            conversation.append(new_message)

        new_example = dict(example)
        new_example["conversation"] = conversation

        fixed_examples.append(new_example)

    return nemotron_omni_collate_fn(
        fixed_examples,
        processor=processor,
        max_length=max_length,
        max_video_frames=max_video_frames,
    )
