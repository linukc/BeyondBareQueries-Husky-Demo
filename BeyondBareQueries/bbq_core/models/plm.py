import os
import urllib.request
import requests
from PIL import Image
from io import BytesIO
import base64
import pickle
import tempfile
import torch

from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform
from core.transforms.video_transform import get_video_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer


class PlmChat:
    def __init__(self):
        # пришлось явно указывать путь после скачивания чтобы заработало
        self.ckpt = "/home/docker_user/.cache/huggingface/hub/models--facebook--Perception-LM-3B/snapshots/f027bc1ef384f3dae09b72500a9eda3c5848b74f/original" #"facebook/Perception-LM-3B"
        self.model, self.tokenizer, self.config = load_consolidated_model_and_tokenizer(ckpt)
        number_of_tiles = 4
        self.transform = get_image_transform(
            vision_input_type=(
                "vanilla" if number_of_tiles == 1 else self.config.data.vision_input_type
            ),
            image_res=self.model.vision_model.image_size,
            max_num_tiles=number_of_tiles,
        )

    def preprocess_image(self, images):
        transform_images = []
        for img in images:
            image, _ = self.transform(img)
            transform_images.append(image)
        return transform_images

    def __call__(self, query, image_features, image_sizes):
        question = query
        image = image_features[0]
        prompts = [(question, image)]
        print("Generating...")
        # Create generator
        temperature=0.0
        top_p=None
        top_k=None
        gen_cfg = dataclass_from_dict(
            PackedCausalTransformerGeneratorArgs,
            {"temperature": temperature, "top_p": top_p, "top_k": top_k},
            strict=False,
        )
        generator = PackedCausalTransformerGenerator(gen_cfg, self.model, self.tokenizer)
        # Run generation
        start_time = time.time()
        generation, loglikelihood, greedy = generator.generate(prompts)
        end_time = time.time()
        return generation[0]
