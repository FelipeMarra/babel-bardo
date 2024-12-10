import typing as tp

import torch 

from audiocraft.models import MusicGen

def generate_bypass(model:MusicGen, descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
    """Generate tokens [B, K, T] conditioned on text.

    Args:
        descriptions (list of str): A list of strings used as text conditioning.
        progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
    """
    attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
    assert prompt_tokens is None
    tokens = model._generate_tokens(attributes, prompt_tokens, progress)
    return tokens

def generate_continuation_bypass(model:MusicGen, prompt_tokens: torch.Tensor, prompt_sample_rate: int,
                                    descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                                    progress: bool = False) -> torch.Tensor:
    """Generate samples conditioned on audio prompt tokens and an optional text description bypassing encodec decode..

    Args:
        prompt (torch.Tensor): A batch of tokens used for continuation.
            Prompt should be [B, K, T], or [K, T] if only one sample is generated.
        prompt_sample_rate (int): Sampling rate of the given audio waveforms.
        descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
        progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
    """
    if prompt_tokens.dim() == 2:
        prompt_tokens = prompt_tokens[None]
    if prompt_tokens.dim() != 3:
        raise ValueError("prompt should have 3 dimensions: [B, K, T] (K = 4).")

    if descriptions is None:
        descriptions = [None] * len(prompt_tokens)

    attributes, _ = model._prepare_tokens_and_attributes(descriptions, None)
    assert prompt_tokens is not None
    tokens = model._generate_tokens(attributes, prompt_tokens, progress)

    return tokens

def encodec_tailfade(model:MusicGen, fade_duration:int, tokens1, tokens2):
    assert fade_duration * 2 <= model.duration # fade_duration * 2 can't be greater than model duration

    tokens2_fade = fade_duration * 2 * model.frame_rate
    audio_fade = fade_duration * model.sample_rate

    # Tokens 2 gets tokens2_fade tokens from the end of tokens1
    tokens2_render = torch.cat((tokens1[:, :, -tokens2_fade:], tokens2), 2)

    # Generate audio for tokens2
    tokens2_render = model.generate_audio(tokens2_render)

    # Clip audio_fade duration from the beggining and end of tokens2_render
    tokens2_render = tokens2_render[:, :, audio_fade:-audio_fade]

    return tokens2_render