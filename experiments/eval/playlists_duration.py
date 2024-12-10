import datetime
from pytubefix import Playlist
import numpy as np

PLAYLIST = 'https://www.youtube.com/playlist?list=PLJ3A9Ntb1tg69P94-iQo0xCdgaBVX-ecu'
playlist = Playlist(PLAYLIST)
playlist = [(video.video_id, video.length) for video in playlist.videos]

no_sub_vids = ['wiGlOf3mCVM', '6yHpP3dsaws', '-ucsTx0u4Lo']
video_lengths = []

for idx_v, video in enumerate(playlist):
    video_id, video_length = video

    if video_id in no_sub_vids:
        continue

    print(video, video_length)
    video_lengths.append(video_length)

video_lengths = np.array(video_lengths)

total = str(datetime.timedelta(seconds=int(video_lengths.sum())))
mean = str(datetime.timedelta(seconds=int(video_lengths.mean())))
std = str(datetime.timedelta(seconds=int(video_lengths.std())))

print("Total:", total, "Mean:", mean, "Std:", std)