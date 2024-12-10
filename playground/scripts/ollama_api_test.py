#%%
import requests
import json
import os

#%%
OLLAMA_ADDRES = os.environ['OLLAMA_ADDRES']
print(OLLAMA_ADDRES)

url = f"http://{OLLAMA_ADDRES}/api/chat"
payload = {
    "model": "llama3.1:70b",
    "stream": False,
    "keep_alive":30,
    "messages": [
        { "role": "user", "content": "why is the sky blue?" }
    ]
}
headers = {}

#%%
res = requests.post(url, json=payload, headers=headers)

# %%
res_dict = json.loads(res.content.decode('utf-8'))
print(type(res_dict), res_dict)

#%%
res_dict['message']['content']