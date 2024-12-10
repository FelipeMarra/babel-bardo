import os
import shutil
from typing import Generator
from pathlib import Path
import math

from tqdm import tqdm

import ffmpeg
import numpy as np
import torch
import torchaudio

from frechet_audio_distance import FrechetAudioDistance

from .passt.passt import get_passt

def _audio_dir_to_mono_sr_wav(audios_dir:str, sr:int=32000, ident:str='', segment:int=0):
    folder_name = f"{ident}_wav_{sr}_mono_{segment}"
    wav_audios_path = os.path.join(audios_dir, folder_name)

    if not os.path.isdir(wav_audios_path):
        os.mkdir(wav_audios_path)
    else:
        return wav_audios_path

    for audio in os.listdir(audios_dir):
        if audio == folder_name:
            continue

        audio_in_path = os.path.join(audios_dir, audio)

        if segment > 0:
            wav_audio = audio.split('.')[0] + '_%03d.wav'
        else:
            wav_audio = audio.split('.')[0] + '.wav'

        audio_out_path = os.path.join(wav_audios_path, wav_audio)

        if os.path.isfile(audio_out_path):
            continue

        if segment > 0:
            stream = ffmpeg.input(audio_in_path)
            stream = ffmpeg.output(stream, audio_out_path, f='segment', segment_time=10, ar=sr, ac=1)
            ffmpeg.run(stream)
        else:
            stream = ffmpeg.input(audio_in_path)
            stream = ffmpeg.output(stream, audio_out_path, ar=sr, ac=1)
            ffmpeg.run(stream)

    return wav_audios_path

def get_fad_vggish(background_path:str, eval_path:str, eval_ep_file:str|None, remove_back:bool=True, overall:bool=True) -> tuple[float, float, str]:
    SAMPLE_RATE = 16000
    wav_background_path = _audio_dir_to_mono_sr_wav(background_path, sr=SAMPLE_RATE, ident='back', segment=30)

    wav_eval_path = ''
    if overall:
        wav_eval_path = _audio_dir_to_mono_sr_wav(eval_path, sr=SAMPLE_RATE, ident='eval', segment=30)
    else:
        folder_name = f"eval_wav_{SAMPLE_RATE}_mono_30"
        wav_eval_path = os.path.join(eval_path, folder_name)
        if not os.path.isdir(wav_eval_path):
            os.mkdir(wav_eval_path)

    frechet = FrechetAudioDistance(
        ckpt_dir="../checkpoints/vggish",
        model_name="vggish",
        sample_rate=SAMPLE_RATE,
        use_pca=False,
        use_activation=False,
        verbose=True,
        audio_load_worker=8,
    )

    overall_fad = -1
    if overall:
        overall_fad = frechet.score(
            wav_background_path,
            wav_eval_path,
            dtype="float32"
        )

    ep_fad = -1
    if eval_ep_file:
        tmp_eval_path = os.path.join(wav_eval_path, 'tmp_eval_ep_fad')
        tmp_eval_file = os.path.join(tmp_eval_path, eval_ep_file.split('/')[-1])
        os.mkdir(tmp_eval_path)

        eval_ep_file = os.path.join(eval_path, eval_ep_file.split('/')[-1])
        os.rename(eval_ep_file, tmp_eval_file)

        tmp_eval_converted_path = _audio_dir_to_mono_sr_wav(tmp_eval_path, sr=SAMPLE_RATE, ident='tmp_eval', segment=30)

        ep_fad = frechet.score(
            wav_background_path,
            tmp_eval_converted_path,
            dtype="float32"
        )

        # Clear temp
        os.rename(tmp_eval_file, eval_ep_file)
        shutil.rmtree(tmp_eval_path)

    # Clear temp
    if remove_back:
        shutil.rmtree(wav_background_path)

    if overall:
        shutil.rmtree(wav_eval_path)

    return overall_fad, ep_fad, wav_background_path

def _audio_iter(audio_path:str|Path, sr:int=32000, seconds:int=10) -> Generator[torch.Tensor, None, None]:
    """
        sr: sample rate
    """
    # Get audio tensor
    streamer = torchaudio.io.StreamReader(str(audio_path))

    streamer.add_basic_audio_stream(
        frames_per_chunk=sr*seconds,
    )

    for chunk in streamer.stream():
        # audio_wave needs shape of [batch, seconds*sample_rate]
        # since we have only one channel and one batch, the channel
        # dim is beeing being repurposed as the batch one
        audio = chunk[0].transpose(1,0).cuda()

        yield audio

def _transition_audio_iter(audio_file:str|Path, save_path:str|Path|None, sr:int=32000) -> Generator[torch.Tensor, None, None]:
    """
        sr: sample rate
    """
    # Get audio tensor
    streamer = torchaudio.io.StreamReader(str(audio_file))

    streamer.add_basic_audio_stream(
        frames_per_chunk=sr*10,
    )

    current_time = 25
    streamer.seek(current_time)

    for chunk in streamer.stream():
        audio = chunk[0].transpose(1,0)

        if save_path != None:
            transition_file = os.path.join(str(save_path), audio_file.split('/')[-1][:-4] + f"_{current_time}_{current_time+10}.wav")
            torchaudio.save(transition_file, audio, sample_rate=sr)

        # audio_wave needs shape of [batch, seconds*sample_rate]
        # since we have only one channel and one batch, the channel
        # dim is beeing being repurposed as the batch one
        yield audio.cuda()

        current_time+=30
        streamer.seek(current_time)

def _transition_segment_iter(audio_file:str|Path, sr:int=32000) -> Generator[torch.Tensor, None, None]:
    """
        sr: sample rate
    """
    # Get audio tensor
    streamer = torchaudio.io.StreamReader(str(audio_file))

    streamer.add_basic_audio_stream(
        frames_per_chunk=sr*10
    )

    current_time = 20
    streamer.seek(current_time)

    for chunk in streamer.stream():
        segment_1 = chunk[0].transpose(1,0)

        try:
            nxt_chunck = next(streamer.stream())
            segment_2 = nxt_chunck[0].transpose(1,0)
        except:
            continue

        # audio_wave needs shape of [batch, seconds*sample_rate]
        # since we have only one channel and one batch, the channel
        # dim is beeing being repurposed as the batch one
        yield segment_1.cuda(), segment_2.cuda()

        current_time+=30
        streamer.seek(current_time)

def _calculate_kld(background:torch.Tensor, eval:torch.Tensor) -> float:
    """
        Calculates KL-divergence as in https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    """
    return torch.sum(background * (background.log() - eval.log())).item()

def get_kld_for_segments_transitions(eval_audio:str|Path) -> dict:
    """
        KLD for 10s windows centered in 30s intervals, when the transitions happen, comparing
        the generated segment before the transition with the one after the transition.
        Only works for audios in mono at 32KHz.

        sr: sample rate
    """
    duration = float(ffmpeg.probe(eval_audio)['streams'][0]['duration'])

    total = math.ceil(duration/30)
    kld_list = [] # list with mean class klds for each 10s segment

    with torch.no_grad():
        # passt
        model = get_passt()
        model.eval()
        model = model.cuda()

        # For each 10s segment
        for first_segment, second_segment in tqdm(_transition_segment_iter(eval_audio, sr=32000), total=total, desc='Segment Transitions KLD'):
            first__sig = model(first_segment).squeeze()
            second_sig = model(second_segment).squeeze()

            # KLD for each label
            class_klds = np.zeros(183)
            count = 0

            for first_prob, second_prob in zip(first__sig, second_sig):
                first_distrib = torch.tensor([first_prob, 1 - first_prob])
                second_distrib = torch.tensor([second_prob, 1 - second_prob])

                class_kld = _calculate_kld(first_distrib, second_distrib)

                class_klds[count] = class_kld
                count += 1

            kld_list.append(class_klds.sum())

    kld_array = np.array(kld_list)

    return {
        'list': kld_list,
        'sum': kld_array.sum(),
        'mean': kld_array.mean(),
        'std': kld_array.std(),
        'min': kld_array.min(),
        'max': kld_array.max()
    }

def get_kld_for_transitions(background_path:str, background_audio:str|Path, eval_audio:str|Path, save_path:str|Path) -> dict:
    """
        KLD for 10s windows centered in 30s intervals, when the transitions happen, of
        generated vs original audios.
        Only works for audios in mono at 32KHz.

        sr: sample rate
    """
    wav_background_path = _audio_dir_to_mono_sr_wav(background_path)
    background_audio = os.path.join(wav_background_path, background_audio.split('/')[-1])

    duration = float(ffmpeg.probe(background_audio)['streams'][0]['duration'])

    total = math.ceil(duration/30)
    kld_list = [] # list with mean class klds for each 10s segment

    with torch.no_grad():
        # passt
        model = get_passt()
        model.eval()
        model = model.cuda()

        # For each 10s segment
        for back_segment, eval_segment in tqdm(zip(_transition_audio_iter(background_audio, None, sr=32000), _transition_audio_iter(eval_audio, save_path, sr=32000)), total=total, desc='Transitions KLD'):
            background_sig = model(back_segment).squeeze()
            eval_sig = model(eval_segment).squeeze()

            # KLD for each label
            class_klds = np.zeros(183)
            count = 0

            for back_prob, eval_prob in zip(background_sig, eval_sig):
                back_distrib = torch.tensor([back_prob, 1 - back_prob])
                eval_distrib = torch.tensor([eval_prob, 1 - eval_prob])

                class_kld = _calculate_kld(back_distrib, eval_distrib)

                class_klds[count] = class_kld
                count += 1

            kld_list.append(class_klds.sum())

    kld_array = np.array(kld_list)

    shutil.rmtree(wav_background_path)

    return {
        'list': kld_list,
        'sum': kld_array.sum(),
        'mean': kld_array.mean(),
        'std': kld_array.std(),
        'min': kld_array.min(),
        'max': kld_array.max()
    }

def get_kld(background_path:str|Path, background_audio:str|Path, eval_audio:str|Path) -> dict:
    """
        KLD for 10s windows of generated vs original audios. 
        Only works for audios in mono at 32KHz.

        sr: sample rate
    """
    wav_background_path = _audio_dir_to_mono_sr_wav(str(background_path))
    background_audio = os.path.join(wav_background_path, background_audio.split('/')[-1])

    duration = float(ffmpeg.probe(background_audio)['streams'][0]['duration'])

    total = math.ceil(duration/10)
    kld_list = [] # list with mean class klds for each 10s segment

    with torch.no_grad():
        # passt
        model = get_passt()
        model.eval()
        model = model.cuda()

        # For each 10s segment
        for back_segment, eval_segment in tqdm(zip(_audio_iter(background_audio, sr=32000), _audio_iter(eval_audio, sr=32000)), total=total, desc='KLD'):
            background_sig = model(back_segment).squeeze()
            eval_sig = model(eval_segment).squeeze()

            # KLD for each label
            class_klds = np.zeros(183)
            count = 0

            for back_prob, eval_prob in zip(background_sig, eval_sig):
                back_distrib = torch.tensor([back_prob, 1 - back_prob])
                eval_distrib = torch.tensor([eval_prob, 1 - eval_prob])

                class_kld = _calculate_kld(back_distrib, eval_distrib)

                class_klds[count] = class_kld
                count += 1

            kld_list.append(class_klds.sum())

    kld_array = np.array(kld_list)

    shutil.rmtree(wav_background_path)

    return {
        'list': kld_list,
        'sum': kld_array.sum(),
        'mean': kld_array.mean(),
        'std': kld_array.std(),
        'min': kld_array.min(),
        'max': kld_array.max()
    }