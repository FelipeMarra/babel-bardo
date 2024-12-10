#%%
import pathlib
import os

import torch

from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write

SCRIPTS_PATH = pathlib.Path(__file__).parent.resolve()
SAVE_AUDIOS_PATH = SCRIPTS_PATH.joinpath('audios')
PROMPT = 'Imagine a track that begins with a soft, mystical ambiance, capturing the essence of stepping into an epic fantasy world. Gentle strings and ethereal chimes create an air of anticipation, setting the stage for the grand adventure ahead. As the dungeon master welcomes the players, the music subtly swells, introducing a light, rhythmic pulse that hints at the excitement and challenges to come. This composition should evoke a sense of camaraderie and heroism, with melodic undertones that reflect the promise of epic quests and the bond between the adventurers. Perfect for immersing players into the magical world of Dungeons & Dragons right from the start of their campaign.'

if not os.path.isdir(SAVE_AUDIOS_PATH):
    os.mkdir(SAVE_AUDIOS_PATH)

#%%

model = MusicGen.get_pretrained('facebook/musicgen-small')
mbd = MultiBandDiffusion.get_mbd_musicgen()

#%%
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=30,
    extend_stride=10
)

wav1, tokens1 = model.generate(
    descriptions=[PROMPT],
    return_tokens=True,
    progress=True
)

wav1 = mbd.tokens_to_wav(tokens1)

overlap = model.duration - model.extend_stride
cutted_out = wav1[:, :, -overlap*model.sample_rate:]
print(cutted_out.shape)

# As an extra example, this one trims the start and end, but the Hyphotesis
# makes more sence
# https://github.com/GrandaddyShmax/audiocraft_plus/blob/c9ff851e7b509d3b6dd735aaefe438823319674e/app.py#L497

#%%
wav2, tokens2 = model.generate_continuation(
        cutted_out,
        prompt_sample_rate = model.sample_rate,
        return_tokens=True,
        progress=True
    )

wav2 = mbd.tokens_to_wav(tokens2)

#%%
# Concatenate output and continuation
concat_outs = torch.cat((wav1, wav2), dim=1)

# Squeeze to eliminate the first dim which indicates how many audios were generated (useless for when only 1 is generated)
output_sqz = torch.squeeze(concat_outs, 0)

audio_write(SAVE_AUDIOS_PATH.joinpath('mdb_audio'), output_sqz.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)