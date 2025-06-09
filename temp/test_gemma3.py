import ollama
import ollama
import re
import json
import os
from tqdm import tqdm
import requests
from openai import OpenAI

def gemma3_api(query):

    api_key = 'sk-3d8155d300fd4f938ac1f6de46c67432'
    url = 'http://127.0.0.1:11434/api/chat'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    payload = {
        "model": 'gemma3:27b',
        "messages": [
            {"role": "user", 
            "content": [
                {"type": "image", "url": "/home/sjtu/wrx/images/screenshot.png"},
                {"type": "text", "text": "对照组现在一共收集了多少数据？"}
            ]}
        ],
        "stream": False

    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        result_content = result["message"]["content"]
        print(result_content)
        return result_content
    else:
        print("ERROR:", response.status_code, response.text)
        return 0

gemma3_api("你是谁")


# response = ollama.chat(model='gemma3:27b', messages=[
#     {
#         'role': 'user',
#         'content': [
#             {"type": "image", "url": "/home/sjtu/wrx/images/screenshot.png"},
#             {"type": "text", "text": "对照组现在一共收集了多少数据？"}
#         ]
#     },
# ],
# )
# print(response['message']['content'])

# response = ollama.chat(model='gemma3:27b', messages=[
#     {
#         'role': 'user',
#         'content': "对照组现在一共收集了多少数据？",
#     },
# ],
# )
# print(response['message']['content'])

