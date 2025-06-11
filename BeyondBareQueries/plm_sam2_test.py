from bbq.models import PlmChat
from PIL import Image
import torch

chat = PlmChat()

image = Image.open('rgb_sam2.png')
image_features = [image]
image_sizes = [image.size for image in image_features]
image_features = chat.preprocess_image(image_features)
image_tensor = [image.to("cuda", dtype=torch.float16) for image in image_features]

query_tail = """
You are given an image with several objects highlighted by colored borders.

This object is commonly found in indoor environments, especially laboratories.
Provide a concise and specific description of the object appearance, geometry, and material in four or five words.
Avoid background details or context beyond the object itself.
"""

for i in range(1, 40):
    query_base = f"Focus on the object assigned ID {i}. Describe this object in detail."
    query = query_base + "\n" + query_tail

    text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
    print("Query: ", query_base)
    print("Answer: ", text)


query_base = """
You are given an image that contains multiple objects. Each object is marked by a unique **colored border** and assigned an **ID number** from 0 to N.

For each object, describe only the item inside the colored border corresponding to its ID. Do not mention any background or unrelated content.

Each description must be:
- Concise (four to five words),
- Focused on **appearance, geometry, and material**,
- Suitable for objects typically found in **indoor environments**, especially **laboratories**.

Return your output as a numbered list:

0. <description>
1. <description>
2. <description>
...
N. <description>
"""
query = query_base

text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
print("Query: ", query_base)
print("Answer: ", text)
