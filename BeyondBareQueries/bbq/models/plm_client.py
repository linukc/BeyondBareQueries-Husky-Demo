import os
import urllib.request
import requests
from PIL import Image
from io import BytesIO
import base64
import pickle
import tempfile
import torch


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def dump_pkl(obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(obj, f)
        pkl_path = f.name
    return pkl_path

class PlmChat:
    def __init__(self, api_url="http://localhost:31623"):
        self.api_url = api_url
        self.api_generate = "/generate"
        self.api_preprocess_image = "/preprocess_image"

    def preprocess_image(self, images):
        pkl_path = dump_pkl(images)
        res = requests.post(self.api_url+self.api_preprocess_image, json={"pkl_path": pkl_path})
        pkl_path = res.json()["response"]
        images = load_pkl(pkl_path)
        return images

    def __call__(self, query, image_features, image_sizes):
        pkl_path = dump_pkl(image_features[0])
        res = requests.post(self.api_url+self.api_generate, json={"question": query, "image": pkl_path})
        res = res.json()["response"]
        return res
