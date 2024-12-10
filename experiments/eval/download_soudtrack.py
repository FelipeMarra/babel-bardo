from babel_bardo.video_manager import get_audio_playlist_for_fad

DOWNLOAD_PATH = '/home/felipe/Desktop/movies_soundtracks/inter'
PLAYLIST = 'https://www.youtube.com/playlist?list=PLco_u-O9FeQ_cV5gc3VdUHoQYBI73MYkU'

print(DOWNLOAD_PATH)

get_audio_playlist_for_fad(DOWNLOAD_PATH, PLAYLIST)