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
    Convert image paths in the CropGuard JSONL dataset
    into PIL RGB images before calling the Nemotron collator.
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
