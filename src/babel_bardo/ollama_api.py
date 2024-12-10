import os
import json
import requests
from enum import Enum

class OllamaType(Enum):
    NONE = -1
    CHAT = 0

class PromptConfig():
    def __init__(self, setup:str="", start:str="", end:str="",) -> None:
        """
        "setup" will be sent to Ollama teach it the task it needs to perform.
        "start" and "end" will be positioned at the beggining and end of the Ollama answer.
        If OllamaType.NONE, "start" and "end" will be positioned at the beggining and end of the dialog.
        """
        self.setup = setup
        self.start = start
        self.end = end

    def __str__(self):
        return f"prompt start: {self.start}\nprompt setup:{self.setup}\nprompt end: {self.end}\n"

class OllamaChat():
    def __init__(self, seed:int, window_size:int=20):
        """
            window_size: size of the chat history (system role messages are not counted)
        """
        self.seed = seed
        self.window_size = window_size +1 # +1 to accout for the system message (task prompt)
        self.ollama_addres = os.environ['OLLAMA_ADDRES']
        self.chat_state = [] #list of dicts containing the chat history

    def _append_message(self, message):
        self.chat_state.append(message)

        if len(self.chat_state) > self.window_size:
            del self.chat_state[1] # Never delete the prompt message

    def send(self, prompt:str, setup:bool=False) -> str:

        if setup:
            self.chat_state.append({"role": "system", "content": prompt})
        else:
            self._append_message({"role": "user", "content": prompt})

        url = f"http://{self.ollama_addres}/api/chat"
        payload = {
            "model": "llama3.1:70b",
            "stream": False,
            "keep_alive":300,
            "options": {
                "seed": self.seed,
                "temperature": 0.4,
                "top_k": 20
            },
            "messages": self.chat_state
        }
        headers = {}

        # Try calling the API at most 5 times
        count = 0
        fail = True
        while fail == True and count < 5:
            try:
                res = requests.post(url, json=payload, headers=headers)
                fail = False
            except Exception as e:
                print("OLLAMA REQUEST EXCEPTION: ", e)
                fail == True
                count += 1

        if fail == False:
            res_dict = json.loads(res.content.decode('utf-8'))
            res_message = res_dict['message']['content']

            self._append_message({"role": "assistant", "content": res_message})

            return res_message
        else:
            return ""