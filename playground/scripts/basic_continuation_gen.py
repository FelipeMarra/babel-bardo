#%%
import pathlib
import os

import torch

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

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

output = model.generate(
    descriptions=[PROMPT],
    progress=True
)

#%%
# Hypothesis:
# To generate the continuation the context passed must 
# be of duration 30s - extend_stride, getting the end 
# of the last generated output

# Although the assert for the continuation prompt being
# prompt.shape[3] < max_gen_len
# Where max_gen_len = self.duration * self.frame_rate = 30 * 50
# And prompt comes from self.compression_model.encode(prompt), which
# for 30s and sample_rate of 32K receives 960.000 values and returns 1500

# The following apps based on MusicGen corroborates the Hyphotesis
# context/overlap size as input to model.generate_continuation 
# https://colab.research.google.com/github/camenduru/MusicGen-colab/blob/main/MusicGen_long_colab.ipynb
# https://github.com/Oncorporation/audiocraft/blob/c39d98965cda19603cafe215fa7b68fd69f41009/app.py#L278C16-L278C84

overlap = model.duration - model.extend_stride
cutted_out = output[:, :, -overlap*model.sample_rate:]
print(cutted_out.shape)

# As an extra example, this one trims the start and end, but the Hyphotesis
# makes more sence
# https://github.com/GrandaddyShmax/audiocraft_plus/blob/c9ff851e7b509d3b6dd735aaefe438823319674e/app.py#L497

#%%
continuation = model.generate_continuation(
        cutted_out,
        prompt_sample_rate = model.sample_rate,
        progress=True
    )
# Cutting 1 second is sufficient already
print(continuation.shape)

#%%
# Concatenate output and continuation
concat_outs = torch.cat((output, continuation), dim=1)

# Squeeze to eliminate the first dim which indicates how many audios were generated (useless for when only 1 is generated)
output_sqz = torch.squeeze(concat_outs, 0)

audio_write(SAVE_AUDIOS_PATH.joinpath('continuated_audio'), output_sqz.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)