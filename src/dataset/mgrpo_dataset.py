"""
Multi-query GRPO Dataset for M-GRPO Stage 2 training.

Each sample is a *pair* of items from the JSON list that share the same region/image.
The dataset groups consecutive pairs (index 0-1, 2-3, 4-5, ...) so that each __getitem__
returns a dict with two questions, their answers, and the shared image.

Expected JSON format (list of dicts, every two consecutive items form a pair):
[
  {"id": "flickr30k-85-1", "conversations": [...], "images": ["cot/flickr30k/xxx.jpg"]},
  {"id": "flickr30k-85-2", "conversations": [...], "images": ["cot/flickr30k/xxx.jpg"]},
  ...
]
"""

import copy
import os
import random
import re
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import SYSTEM_MESSAGE


def replace_image_tokens(input_string):
    pattern = r'\n?' + re.escape("<image>") + r'\n?'
    return re.sub(pattern, '', input_string)


def get_image_content(image_path, min_pixel, max_pixel, width, height):
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    return content


class MGRPODataset(Dataset):
    """Dataset for Multi-query GRPO training.
    
    Returns one *pair* per index.  The pair contains:
      - prompt: list of message dicts (with images + two text questions)
      - assistant: dict with text1-answer and text2-answer
    """

    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
    ):
        super().__init__()
        raw = json.load(open(data_path, "r"))
        # Group consecutive pairs
        assert len(raw) % 2 == 0, f"Dataset length {len(raw)} is not even — cannot form pairs."
        self.pairs = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]

        # Optional random subsampling
        sample_ratio = getattr(data_args, "dataset_sample_ratio", 1.0)
        if 0.0 < sample_ratio < 1.0:
            rng = random.Random(getattr(data_args, "random_seed", 42) or 42)
            k = max(1, int(len(self.pairs) * sample_ratio))
            self.pairs = rng.sample(self.pairs, k)
            print(f"[MGRPODataset] Sampled {k} pairs ({sample_ratio*100:.1f}%) from {len(raw)//2} total pairs.", flush=True)

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i) -> Dict:
        item1, item2 = self.pairs[i]

        # ---- Build image content (shared between the two questions) ----
        image_files = item1.get("images", [])
        image_folder = self.data_args.image_folder or ""
        contents = []
        for img in (image_files if isinstance(image_files, list) else [image_files]):
            if not os.path.exists(img):
                if not img.startswith("http"):
                    img = os.path.join(image_folder, img)
            contents.append(get_image_content(
                img, self.image_min_pixel, self.image_max_pixel,
                self.image_resized_w, self.image_resized_h,
            ))

        # ---- Extract the two questions and answers ----
        q1 = replace_image_tokens(item1["conversations"][0]["value"])
        a1 = item1["conversations"][1]["value"]
        q2 = replace_image_tokens(item2["conversations"][0]["value"])
        a2 = item2["conversations"][1]["value"]

        # Build a single prompt that contains images + two text entries
        contents.append({"type": "text1", "text1": q1})
        contents.append({"type": "text2", "text2": q2})

        user_prompt = [{"role": "user", "content": contents}]
        if SYSTEM_MESSAGE:
            user_prompt.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})

        data_dict = dict(
            prompt=user_prompt,
            assistant={"text1-answer": a1, "text2-answer": a2},
        )
        return data_dict


def make_mgrpo_data_module(model_id, processor, data_args):
    dataset = MGRPODataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )
    return dict(train_dataset=dataset, eval_dataset=None)
