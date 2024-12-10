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

# For long generation just set duration for more than 30s. 
# It will keep generating at most 30s and using extend_stride 
# to know how much to extend the audio at each time, keeping 
# the previously generated as context.
# A extend_stride=10 will generate 10s and keep 20 as context
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=60,
    extend_stride=10
)

output = model.generate(
    descriptions=[PROMPT],
    progress=True
)

# Squeeze to eliminate the first dim which indicates how many audios were generated (useless for when only 1 is generated)
output_sqz = torch.squeeze(output, 0)

audio_write(SAVE_AUDIOS_PATH.joinpath('long_audio'), output_sqz.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

# ####################### _generate_tokens #######################
# Started with prompt_length 0
# stride_tokens 500

# While 0 < 3000
# Generated size torch.Size([1, 4, 1500])
# New prompt_length is 1000
# New current_gen_offset is 500
#  ############### ALL TOKENS: ##################
# torch.Size([1, 4, 1500])

# While 1500 < 3000
# Generated size torch.Size([1, 4, 1500])
# New prompt_length is 1000
# New current_gen_offset is 1000
#  ############### ALL TOKENS: ##################
# torch.Size([1, 4, 1500])
# torch.Size([1, 4, 500])

# While 2000 < 3000
# Generated size torch.Size([1, 4, 1500])
# New prompt_length is 1000
# New current_gen_offset is 1500
#  ############### ALL TOKENS: ##################
# torch.Size([1, 4, 1500])
# torch.Size([1, 4, 500])
# torch.Size([1, 4, 500])

# While 2500 < 3000
# Generated size torch.Size([1, 4, 1500])
# New prompt_length is 1000
# New current_gen_offset is 2000
#  ############### ALL TOKENS: ##################
# torch.Size([1, 4, 1500])
# torch.Size([1, 4, 500])
# torch.Size([1, 4, 500])
# torch.Size([1, 4, 500])