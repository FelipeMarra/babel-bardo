import os
import shutil
import pathlib

from slugify import slugify
import ffmpeg

from pytubefix import YouTube, Playlist
from pytubefix.cli import on_progress

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, VideoClip, AudioClip
from moviepy.audio.fx.volumex import volumex
import demucs.separate

from babel_bardo.templates import BardoTemplate

def fit_audio_in_video(template:BardoTemplate, volumex_value:float=0.4):
    if os.path.isfile(template.generated_video_file):
        print("Skipping fit_audio_in_video", template.generated_video_file)
        return

    URL = f"https://www.youtube.com/watch?v={template.video_id}"

    # Download video
    if not os.path.isfile(template.original_video_file):
        yt = YouTube(URL, on_progress_callback = on_progress)
        ys = yt.streams.get_highest_resolution()
        ys.download(template.original_videos_path, filename=template.original_video_file_name)
    else:
        print("Skipping download video", URL)

    # Load the video clip and get its audios
    video_clip = VideoFileClip(template.original_video_file)
    video_clip:VideoClip = video_clip.subclip(template.start_time, template.end_time)

    audio_clip:AudioClip = video_clip.audio

    temp_original_audio_file = 'temp_original_audio'
    demucs_path = os.path.join(template.original_vocals_path, 'htdemucs')
    demucs_vocals = os.path.join(demucs_path, temp_original_audio_file, 'vocals.wav')
    demucs_no_vocals = os.path.join(demucs_path, temp_original_audio_file, 'no_vocals.wav')

    if not os.path.isfile(template.original_vocals_file):
        # Write the audio to vocals path to get it separated by demucs
        temp_original_audio_file = os.path.join(template.original_vocals_path, temp_original_audio_file+'.wav')
        audio_clip.write_audiofile(temp_original_audio_file)

        # Separete the vocals from the audio
        demucs.separate.main(["--two-stems", "vocals", temp_original_audio_file, "--out", template.original_vocals_path])

        # Move demucs out file to original_vocals_file
        os.rename(demucs_vocals, template.original_vocals_file)
        os.rename(demucs_no_vocals, template.original_audio_file)

        # Clear
        os.remove(temp_original_audio_file)
        shutil.rmtree(demucs_path)
    else:
        print("Skipping demucs", template.original_audio_file)

    # Load the vocals audio
    vocals = AudioFileClip(template.original_vocals_file)

    try:
        # Load the background music and ajust its volume
        background_music = AudioFileClip(template.generated_audio_file)
        background_music.fx(volumex, volumex_value)

        # Put background music and vocals together
        combined_audio = CompositeAudioClip([background_music, vocals])

        # Put the new combined audio into the video
        final_video:VideoClip = video_clip.set_audio(combined_audio)

        final_video.write_videofile(template.generated_video_file)

    except Exception as e:
        print(f"\n Retrying volume ajustment: {e}\n")

        background_music = AudioFileClip(template.generated_audio_file).volumex(volumex_value)

        combined_audio = CompositeAudioClip([background_music, vocals])

        final_video:VideoClip = video_clip.set_audio(combined_audio)

        final_video.write_videofile(template.generated_video_file)

    finally:
        print("\n Ignoring volume ajustment: \n")

        background_music = AudioFileClip(template.generated_audio_file)
        combined_audio = CompositeAudioClip([background_music, vocals])
        final_video:VideoClip = video_clip.set_audio(combined_audio)

        final_video.write_videofile(template.generated_video_file)

    # Close videos
    video_clip.close()
    audio_clip.close()
    combined_audio.close()
    final_video.close()

def get_audio_playlist_for_fad(download_path:pathlib.Path, playlist_link:str):
    if isinstance(download_path, str):
        download_path = pathlib.Path(download_path)

    if not os.path.isdir(download_path):
        os.makedirs(download_path)

    playlist = Playlist(playlist_link)

    count = 0
    for video in playlist.videos:
        audio = video.streams.get_audio_only()

        if audio == None:
            continue

        audio.download(str(download_path))

        audio_name = audio.default_filename.replace('/','')
        out_name = slugify(audio_name.split('.')[0], separator='_')+'.wav'

        if out_name in os.listdir(download_path):
            out_name = out_name[:-4] + f'_{count}' + out_name[-4:]
            count += 1

        print('AUDIO NAME:', audio_name, out_name)

        stream = ffmpeg.input(download_path.joinpath(audio_name))
        stream = ffmpeg.output(stream, str(download_path.joinpath(out_name)), ac=1)
        ffmpeg.run(stream)

        os.remove(download_path.joinpath(audio_name))