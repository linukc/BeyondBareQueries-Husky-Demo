import os
import yaml
import time
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import json
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from http.server import HTTPServer, SimpleHTTPRequestHandler

from BeyondBareQueries.bbq_core.models.plm_client import PlmChat
from BeyondBareQueries.bbq_core.grounding.llm_interface import Llama3
# from BeyondBareQueries.bbq_utils.draw import draw_answer

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

# Путь к файлу, куда клиент отправляет текст
TEXT_FILE = "text.txt"
# COLOR_PATH = "downloaded_images/image.png"
# DEPTH_PATH = "downloaded_images/depth.png"
# POSE_PATH = "pose.txt"
SAVE_PATH = "outputs"
#LLAMA_PATH = "/datasets/Meta-Llama-3-8B-Instruct"
LLAMA_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"

# os.makedirs("crop_images", exist_ok=True)

DEBUG = False

with open("secrets.yaml") as stream:
    secrets = yaml.safe_load(stream)

ANSWERS = None

hash = datetime.now()

if not DEBUG:
    chat = PlmChat()
    logger.info("PLM chat is initialized.")
    llm = Llama3(LLAMA_PATH)
    logger.info("LLama3 chat is initialized.")

class CustomHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        start = time.time()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        print("Received from client:", post_data)
        user_query = post_data

        if self.path == '/first':
            # url = secrets["CAMERA_SERVER"]
            # payload = 'Send me two images'

            # response = requests.post(url, data=payload, verify=False)

            # if response.status_code == 200 and 'application/zip' in response.headers.get('Content-Type', ''):
            #     with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            #         z.extractall('downloaded_images')
            #         print("Images extracted to 'downloaded_images/'")
            # else:
            #     print("Server error:", response.text)
            print("Getting image has not implemented yet !!!!!!!")

            end = time.time()
            print("Elapsed time:", end - start, "seconds")

            global ANSWERS
            ANSWERS = main_first(user_query)
            print("Answers", len(ANSWERS))
            print(json.dumps(ANSWERS[-1]))
            response = {"message": json.dumps(ANSWERS[-1])}
            response_string = json.dumps(response)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_string)))
            self.end_headers()
            self.wfile.write(response_string.encode())
        if self.path == '/second':
            answer = main_second(ANSWERS)
            response = {"message": json.dumps(answer)}
            response_string = json.dumps(response)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_string)))
            self.end_headers()
            self.wfile.write(response_string.encode())

        return post_data

class TqdmLoggingHandler:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message, **kwargs):
        if message.strip() != "":
            tqdm.write(message, end="")

    def flush(self):
        pass

def set_seed(seed: int=18) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")

def main_first(user_query):
    """Функция, выполняемая при получении сообщения от клиента."""
    print("Сообщение получено! Выполняем main()...")

    with open(os.path.join(SAVE_PATH, "user_query.txt"), "w") as f:
        f.write(user_query)

    # result = describe_objects(objects)
    # logger.info('Saving objects.')
    # with open(os.path.join(SAVE_PATH, "objects.json"), "w") as f:
    #     json.dump(result, f, indent=2)
    with open(os.path.join(SAVE_PATH, "objects.json"), "r") as f:
       result = json.load(f)
    
    if not DEBUG:
        llm.set_scene(os.path.join(SAVE_PATH, "objects.json"))
        logger.info("Selecting relevant nodes")
        related_objects, relations, json_answer = llm.select_relevant_nodes(user_query)
        logger.info(related_objects)

        targets = [obj['id'] for obj in related_objects['target_objects']]
        anchors = [obj['id'] for obj in related_objects['anchor_objects']]

    else:
        relations = [ (0, 1, "test relation"), (2, 3, "test relation")]
        json_answer = '{"test_answer": "Test_answer"}'

        targets = []
        anchors = []
        for obj in result:
            #if obj['id'] % 2 == 0:
            #    targets.append(obj['id'])
            #elif obj['id'] % 3 == 0:
            #    anchors.append(obj['id'])
            targets.append(3)
            anchors.append(0)

    #segmentation1 = deepcopy(segmentation)
    #draw_answer(result, targets, anchors, relations, segmentation1, depth, INTRINSICS, pose, user_query, json_answer, "relevant_objects.png")
    print("Draw answers has not implemented yet !!!!!")
    # return (result, targets, anchors, relations, segmentation, depth, INTRINSICS, pose, user_query, related_objects, json_answer)

    return (user_query, related_objects, targets, anchors, relations, json_answer)
    
def main_second(answers):
    #result, targets, anchors, relations, segmentation, depth, INTRINSICS, pose, user_query, related_objects, json_answer = answers
    user_query, related_objects, targets, anchors, relations, json_answer = answers
    logger.info("Selecting reffered object")
    full_answer, final_answer, relations, pretty_answer = llm.select_referred_object(user_query, related_objects)
    logger.info(full_answer)

    targets = [obj['id'] for obj in related_objects['target_objects'] if obj['id'] == final_answer]
    targets += [obj['id'] for obj in related_objects['anchor_objects'] if obj['id'] == final_answer]

    #final_targets = [obj['id'] for obj in related_objects['target_objects'] if obj['id'] == final_answer]
    filtered_relations = []

    for rel in relations:
        targets.append(rel[0])
        anchors.append(rel[1])
        if rel[0] == final_answer:
            filtered_relations.append(rel)

    json_answer = pretty_answer
    # segmentation2 = deepcopy(segmentation)

    #draw_answer(result, targets, anchors, filtered_relations, segmentation2, depth, INTRINSICS, pose, user_query, json_answer, "final_answer.png", final_targets=final_targets)
    print("Draw answer has not implemented yet !!!")
    return json_answer

def run_server():
    httpd = HTTPServer(('0.0.0.0', 4444), CustomHandler)
    # httpd.socket = ssl.wrap_socket(httpd.socket,
    #                             server_side=True,
    #                             keyfile="key.pem",
    #                             certfile="cert.pem")
    # print("Server running on https://localhost:4444")
    print("start server")
    httpd.serve_forever()

if __name__ == '__main__':
    set_seed()
    run_server()
