import time
import base64
import pickle

from fastapi import FastAPI, Request
from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer

ckpt = "/home/docker_user/.cache/huggingface/hub/models--facebook--Perception-LM-3B/snapshots/f027bc1ef384f3dae09b72500a9eda3c5848b74f/original"
model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt)
number_of_tiles = 4
transform = get_image_transform(
    vision_input_type=(
        "vanilla" if number_of_tiles == 1 else config.data.vision_input_type
    ),
    image_res=model.vision_model.image_size,
    max_num_tiles=number_of_tiles,
)

app = FastAPI()

def serialize_for_send(obj):
    data_bytes = pickle.dumps(obj)
    return base64.b64encode(data_bytes).decode('utf-8')

def deserialize_received(data_b64):
    data_bytes = base64.b64decode(data_b64)
    return pickle.loads(data_bytes)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    question = data["question"]
    image_b64 = data["image"]
    image = deserialize_received(image_b64)
    prompts = [(question, image)]

    # Здесь ваш генератор
    print("Generating...")
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
    images_b64 = data["data"]
    images = deserialize_received(images_b64)

    # Здесь ваш transform:
    transform_images = []
    for img in images:
        image, _ = transform(img)
        transform_images.append(image)

    response_b64 = serialize_for_send(transform_images)
    return {"response": response_b64}
