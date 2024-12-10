#! python3 -m pip uninstall pt_brdo -y && python3 -m pip install --no-dependencies -e .
import os
import json
import pathlib
import shutil
from pytubefix import Playlist

from babel_bardo.templates import *
from babel_bardo import Bardo, fit_audio_in_video
from babel_bardo.eval_metrics import get_kld, get_kld_for_transitions, get_fad_vggish, get_kld_for_segments_transitions


RPGNAME = 'Call Of The Wild'
PLAYLIST = 'https://www.youtube.com/playlist?list=PLMZlu4rxEyKKl-Ecgca3bbVDMZ89SaQHN'
EXCERPT_LENGTH = 60 * 30
BARDO_ROOT_PATH = pathlib.Path(__file__).parent.joinpath('results').resolve()
SOUNDTRACK_PATH = None # path for FAD background statistics 

playlist = Playlist(PLAYLIST)
playlist = [(video.video_id, video.length) for video in playlist.videos]

no_sub_vids = ['d4WEJ2thu4E']

def write_eps_start(template:BardoTemplate, eps_start:dict):
    if not os.path.isdir(template.root_path):
        os.makedirs(template.root_path)

    with open(template.eps_start_file, "w") as json_file: 
        json.dump(eps_start, json_file, indent=4)

def load_eps_start(template:BardoTemplate, playlist) -> dict:
    eps_start = None

    if os.path.isfile(template.eps_start_file):
        with open(template.eps_start_file, 'r') as json_file:
            eps_start = json.load(json_file)
    else:
        eps_start = {video_id:None for video_id, _ in playlist}

    return eps_start

def write_metrics(template:BardoTemplate, metrics:dict, type:str):
    file = '' 
    if type == 'kld':
        file = template.kld_file
    elif type == 'fad':
        file = template.fad_file
    elif type == 't_kld':
        file = template.transitions_kld_metrics_file
    elif type == 's_t_kld':
        file = template.segments_transitions_kld_metrics_file

    with open(file, "w") as json_file: 
        json.dump(metrics, json_file, indent=4)

def load_metrics(path_to_file:str) -> dict:
    metrics = None

    if os.path.isfile(path_to_file):
        with open(path_to_file, 'r') as json_file:
            metrics = json.load(json_file)
    else:
        metrics = {}

    return metrics

wav_background_path = ''

for idx_v, video in enumerate(playlist):
    video_id, video_length = video

    bardo_templates:list[BardoTemplate] = [
        Bardo1(RPGNAME, BARDO_ROOT_PATH, video_id, translate=False), #Bardo1 commes 1st because it don't use Ollama
        Bardo0(RPGNAME, BARDO_ROOT_PATH, video_id),
        Bardo2(RPGNAME, BARDO_ROOT_PATH, video_id),
        Bardo3(RPGNAME, BARDO_ROOT_PATH, video_id)
    ]

    if video_id in no_sub_vids:
        continue

    for idx_t, template in enumerate(bardo_templates):
        is_last_template = idx_t == len(bardo_templates) -1

        # Set the same start for the same episode in set_random_excerpt
        eps_start = load_eps_start(template, playlist)

        if eps_start[video_id] != None:
            print(f"\n LOADED eps_start {eps_start[video_id]} for {video_id} \n")

        eps_start[video_id] = template.set_random_excerpt(EXCERPT_LENGTH, video_length, eps_start[video_id])

        print(f"\n Video {video_id} is starting at {eps_start[video_id]}")
        write_eps_start(template, eps_start)

        print(template.log_header)

        # Bardo Play
        if not os.path.isfile(template.generated_audio_file):
            bardo = Bardo(template)
            bardo.play()
        else:
            print("Skipping generating", template.generated_audio_file)

        fit_audio_in_video(template, video_id)

        # Get Metrics
        # KLD
        kld_metrics = load_metrics(template.kld_file)

        if kld_metrics.get(template.bardo_name) == None:
            kld_data = get_kld(template.original_audios_path, template.original_audio_file, template.generated_audio_file)

            print(f"\n MEAN KLD for {template.bardo_name}, video {video_id} = {kld_data['mean']} \n")

            kld_metrics[template.bardo_name] = {k:str(v) for k,v in kld_data.items()}
            write_metrics(template, kld_metrics, 'kld')

        # KLD for segments transitions
        s_t_kld_metrics = load_metrics(template.segments_transitions_kld_metrics_file)

        if s_t_kld_metrics.get(template.bardo_name) == None:
            s_t_kld_data = get_kld_for_segments_transitions(template.generated_audio_file)

            print(f"\n MEAN SEGMENT TRANSITION KLD for {template.bardo_name}, video {video_id} = {s_t_kld_data['mean']} \n")

            s_t_kld_metrics[template.bardo_name] = {k:str(v) for k,v in s_t_kld_data.items()}
            write_metrics(template, s_t_kld_metrics, 's_t_kld')

        # KLD for transitions
        t_kld_metrics = load_metrics(template.transitions_kld_metrics_file)

        if t_kld_metrics.get(template.bardo_name) == None:
            t_kld_data = get_kld_for_transitions(template.original_audios_path, template.original_audio_file, template.generated_audio_file, template.transitions_eval_audios_path)

            print(f"\n MEAN TRANSITION KLD for {template.bardo_name}, video {video_id} = {t_kld_data['mean']} \n")

            t_kld_metrics[template.bardo_name] = {k:str(v) for k,v in t_kld_data.items()}
            write_metrics(template, t_kld_metrics, 't_kld')

        #FAD
        if SOUNDTRACK_PATH != None:
            fad_metrics = load_metrics(template.fad_file)
            if fad_metrics.get(template.bardo_name) == None:
                overall_fad, ep_fad, wav_background_path = get_fad_vggish(SOUNDTRACK_PATH, template.generated_audios_path, template.generated_audio_file, False)

                fad_metrics[template.bardo_name] = {}
                fad_metrics[template.bardo_name]['ep'] = str(ep_fad)
                fad_metrics[template.bardo_name]['overall'] = str(overall_fad)

                print(f"\n FAD for {template.bardo_name} = ep: {ep_fad}, overall: {overall_fad} \n")
                write_metrics(template, fad_metrics, 'fad')

shutil.rmtree(wav_background_path)