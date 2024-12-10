#%%
import pathlib
import os

import torch

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from time import perf_counter

SCRIPTS_PATH = pathlib.Path(__file__).parent.resolve()
SAVE_AUDIOS_PATH = SCRIPTS_PATH.joinpath('audios')
PROMPT = 'Imagine a track that begins with a soft, mystical ambiance, capturing the essence of stepping into an epic fantasy world. Gentle strings and ethereal chimes create an air of anticipation, setting the stage for the grand adventure ahead. As the dungeon master welcomes the players, the music subtly swells, introducing a light, rhythmic pulse that hints at the excitement and challenges to come. This composition should evoke a sense of camaraderie and heroism, with melodic undertones that reflect the promise of epic quests and the bond between the adventurers. Perfect for immersing players into the magical world of Dungeons & Dragons right from the start of their campaign.'

if not os.path.isdir(SAVE_AUDIOS_PATH):
    os.mkdir(SAVE_AUDIOS_PATH)

model = MusicGen.get_pretrained('facebook/musicgen-small')

#%%
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=30,
    extend_stride=10
)

#%%
start = perf_counter()
wav1, tokens1 = model.generate(
    descriptions=[PROMPT],
    return_tokens=True,
    progress=True
)
end = perf_counter()
print("wav1.shape:", wav1.shape, "took:", end-start,"seconds")

#%%
overlap = model.duration - model.extend_stride
cutted_wav1 = wav1[:, :, -overlap*model.sample_rate:]
print("cutted_wav1.shape:", cutted_wav1.shape)


#%%
start = perf_counter()
wav2, tokens2 = model.generate_continuation(
        cutted_wav1,
        prompt_sample_rate = model.sample_rate,
        return_tokens=True,
        progress=True
    )
end = perf_counter()
print("wav2.shape:", wav2.shape, "took:", end-start,"seconds")

#%%
# Basiline
print("Direct concatenation between wav1 and wav2")
wav_out = torch.cat((wav1, wav2), 2)
wav_out = torch.squeeze(wav_out, 0)
audio_write(SAVE_AUDIOS_PATH.joinpath('baseline'), wav_out.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

#%%
# Render wav2 getting part of wav1
print("\nRender wav2 getting part of wav1")
tokens2_render = torch.cat((tokens1[:, :, -50:], tokens2), 2)
print("cat tokens1 -50: w/ tokens2", tokens2_render.shape)

tokens2_render = model.compression_model.decode(tokens2_render)
print("decoded tokens2_render", tokens2_render.shape)

tokens2_render = tokens2_render[:, :, model.sample_rate:]
print("cutted the wav1 extra 1s from tokens2_render", tokens2_render.shape)

wav_out = torch.cat((wav1, tokens2_render), 2)
print("cat wa1 and tokens2_render", wav_out.shape)

wav_out = torch.squeeze(wav_out, 0)
audio_write(SAVE_AUDIOS_PATH.joinpath('wav2_getting_1s_wa1'), wav_out.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

#%%
# wav1/wav2 crossfade
print("\nwav1/wav2 crossfade")
tokens1_render = torch.cat((tokens1, tokens2[:, :, :50]), 2)
print("cat tokens1 w/ tokens2 :50", tokens1_render.shape)

tokens2_render = torch.cat((tokens1[:, :, -50:], tokens2), 2)
print("cat tokens1 -50: w/ tokens2", tokens2_render.shape)

tokens1_render = model.compression_model.decode(tokens1_render)
tokens2_render = model.compression_model.decode(tokens2_render)
print("decoded tokens1_render", tokens1_render.shape, "and tokens2_render", tokens2_render.shape)

tokens1_render = tokens1_render[:, :, :-model.sample_rate]
tokens2_render = tokens2_render[:, :, model.sample_rate:]
print("cutted the extra 1s tokens1_render", tokens1_render.shape, "tokens2_render", tokens2_render.shape)

wav_out = torch.cat((tokens1_render, tokens2_render), 2)
print("cat tokens1_render and tokens2_render", wav_out.shape)

wav_out = torch.squeeze(wav_out, 0)
audio_write(SAVE_AUDIOS_PATH.joinpath('wav1_wav2_corssfade'), wav_out.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)