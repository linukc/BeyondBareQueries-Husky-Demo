import pickle
import requests
import base64

def serialize_for_send(obj):
    data_bytes = pickle.dumps(obj)
    return base64.b64encode(data_bytes).decode('utf-8')

def deserialize_received(data_b64):
    data_bytes = base64.b64decode(data_b64)
    return pickle.loads(data_bytes)

class PlmChat:
    def __init__(self, api_url="http://localhost:31623"):
        self.api_url = api_url
        self.api_generate = "/generate"
        self.api_preprocess_image = "/preprocess_image"

    def preprocess_image(self, images):
        data_b64 = serialize_for_send(images)
        res = requests.post(self.api_url + self.api_preprocess_image, json={"data": data_b64})
        received_b64 = res.json()["response"]
        images = deserialize_received(received_b64)
        return images

    def __call__(self, query, image_features):
        data_b64 = serialize_for_send(image_features[0])
        res = requests.post(self.api_url + self.api_generate, json={"question": query, "image": data_b64})
        res_json = res.json()
        return res_json["response"]
