import os
from copy import deepcopy
import random

import argparse

from .transcript_iter import TranscriptIter
from .music_gen_bypass import generate_bypass, generate_continuation_bypass, encodec_tailfade
from .ollama_api import OllamaChat, OllamaType
from .constants import *
from.log import clear_log, write_log_header, write_log_description
from .templates import BardoTemplate

import numpy as np
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from tqdm import tqdm

class Bardo():
    def __init__(self, template:BardoTemplate, model:MusicGen|None=None) -> None:
        self.t = template

        self._create_dir_structure()

        clear_log(self.t.log_file)
        write_log_header(self.t.log_file, self.t.log_header, self.t.prompt_config)
        print(str(self.t.prompt_config))

        # Set seed
        self.seed = random.randint(0, 2**32 - 1) if self.t.seed == None else self.t.seed
        print("Using seed:", self.seed, "for MusicGen & Ollama")
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Setup Ollama
        if self.t.ollama_type != OllamaType.NONE:
            self.ollama_chat = OllamaChat(self.seed)

        if self.t.prompt_config.setup != "":
            # To set a setup prompt Ollama must be in CHAT mode
            assert self.t.ollama_type == OllamaType.CHAT

            self.ollama_chat.send(self.t.prompt_config.setup, setup=True)
            print("Setted Ollama prompt")

        # Setup MusicGen
        if model == None:
            self._set_model()
        else:
            self.model = model

    def _set_model(self):
        self.model = MusicGen.get_pretrained(MODEL)
        self.model.set_generation_params(
            duration=DURATION,
            extend_stride=EXTEND_STRIDE
        )

    def _create_dir_structure(self):
        for dir in self.t.dirs_to_create:
            if not os.path.isdir(dir):
                os.makedirs(dir)

    def _parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-cc", "--ClearCache", help = "Clears the transcripts cache", action='store_true')

        args = parser.parse_args()

        if args.ClearCache:
            TranscriptIter.clear_transcript_cache()

    def play(self, save_every:int=-1):
        self._parser()

        t_iter = TranscriptIter(self.t.video_id, start_time=self.t.start_time, end_time=self.t.end_time, language=self.t.language)
        overlap = self.model.duration - self.model.extend_stride

        previous_tokens = None
        previous_wav = None

        tqdm_iter = tqdm(enumerate(t_iter), total=len(t_iter), desc=f"Generating Songs For Video {t_iter.video_id}")

        for idx, description in tqdm_iter:
            frases, _ = description

            if self.t.ollama_type != OllamaType.NONE:
                text_prompt = self.ollama_chat.send(frases)
                text_prompt = text_prompt.strip('\"')
                text_prompt = self.t.prompt_config.start + text_prompt + self.t.prompt_config.end
            else:
                text_prompt = self.t.prompt_config.start + frases + self.t.prompt_config.end
            
            if text_prompt == "CONTINUE.":
                tqdm.write('Received a "CONTINUE." command, sending text_prompt=None')
                text_prompt=None

            tqdm.write(f"\nGenerating idx {idx} \nText Prompt: {text_prompt}")

            if idx == 0:
                current_tokens = generate_bypass(
                    self.model,
                    descriptions=[text_prompt],
                    progress=True
                )
            else:
                previous_overlap = previous_tokens[:, :, -overlap*self.model.frame_rate:]

                current_tokens = generate_continuation_bypass(
                    self.model,
                    previous_overlap,
                    descriptions=[text_prompt],
                    prompt_sample_rate = self.model.sample_rate,
                    progress=True
                )

            write_log_description(self.model, self.t.log_file, frases, text_prompt, idx, self.t.start_time, tqdm_iter)

            # Concatenate outputs
            if previous_wav == None:
                # We're gonna play 29s and save 1s as crossfade
                previous_wav = self.model.generate_audio(current_tokens)[:, :, :-CROSSFADE_DURATION*self.model.sample_rate]
            else:
                # Decode the current tokens with tailfade, getting context from the previous ones
                current_wav = encodec_tailfade(self.model, CROSSFADE_DURATION, previous_tokens, current_tokens)
                # Join the wav generated until now with the audio from previous_tokens+current_tokens
                previous_wav = torch.cat((previous_wav, current_wav), dim=2)

            if save_every > 0 and (idx+1) % save_every == 0:
                partial_wav = torch.squeeze(previous_wav, 0)
                save_every_path = os.path.join(self.t.generated_audio_file_no_ext+'_partial', str(idx))
                audio_write(save_every_path, partial_wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)

            # Update previous tokens
            previous_tokens = deepcopy(current_tokens)

        # Save concatenated outputs
        previous_wav = torch.squeeze(previous_wav, 0)
        audio_write(self.t.generated_audio_file_no_ext, previous_wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
