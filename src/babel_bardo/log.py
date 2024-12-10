import os
import datetime
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from .ollama_api import PromptConfig
from audiocraft.models import MusicGen

def clear_log(log_file:str|Path):
    log_file = str(log_file)

    if os.path.isfile(log_file):
        os.remove(log_file)

def write_log_header(log_file:str|Path, description:str, prompt_cfg:PromptConfig):
    log_file = str(log_file)

    with open(log_file, 'a') as file:
        file.write(f"{description} \n{str(prompt_cfg)} \n")

def write_log_description(model:MusicGen, log_file:str|Path, frases:str, text_prompt:str, idx:int, start_time:int, tqdm_iter:tqdm):
    log_file = str(log_file)

    with open(log_file, 'a') as file:
        time = str(datetime.timedelta(seconds=start_time + idx*model.duration))

        f_dict = deepcopy(tqdm_iter.format_dict)
        f_dict.update(total=False, bar_format=False, desc='')

        to_write = ""
        try:
            tqdm_str = tqdm.format_meter(**f_dict)
            to_write = f"time: {time} \n" + f"dialog: \n {frases} \n" + f"text_prompt:\n {text_prompt}\n" + f"tqdm: {tqdm_str} \n" + '\n\n'
        except:
            print("\n Failed to log with tqdm \n")
            to_write = f"time: {time} \n" + f"dialog: \n {frases} \n" + f"text_prompt:\n {text_prompt}\n" + '\n\n'

        file.write(to_write)