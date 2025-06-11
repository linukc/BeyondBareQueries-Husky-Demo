import os
import torch
from PIL import Image
import time
from base64 import b64encode
import textwrap
import requests
import urllib.request

from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform
from core.transforms.video_transform import get_video_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer

from fastapi import FastAPI, Request
from io import BytesIO
import base64
import pickle
import tempfile
app = FastAPI()

# ckpt = "facebook/Perception-LM-1B"
# ckpt = "facebook/Perception-LM-3B"
ckpt = "/datasets/KM-Models/Perception-LM-3B"
model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt)
number_of_tiles = 4
transform = get_image_transform(
    vision_input_type=(
        "vanilla" if number_of_tiles == 1 else config.data.vision_input_type
    ),
    image_res=model.vision_model.image_size,
    max_num_tiles=number_of_tiles,
)

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def dump_pkl(obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(obj, f)
        pkl_path = f.name
    return pkl_path

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    question = data["question"]
    pkl_path = data["image"]
    image = load_pkl(pkl_path)
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
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)
    # Run generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    return {"response": generation[0]}

@app.post("/preprocess_image")
async def preprocess_image(request: Request):
    data = await request.json()
    pkl_path = data["pkl_path"]
    images = load_pkl(pkl_path)
    transform_images = []
    for img in images:
        image, _ = transform(img)
        transform_images.append(image)
    pkl_path = dump_pkl(transform_images)
    return {"response": pkl_path}