##############################
# Code for the README example
##############################

from babel_bardo.templates import Bardo3
from babel_bardo import Bardo, fit_audio_in_video

VIDEO_ID = "tZWU5iPjQpI"
template = Bardo3('Call Of The Wild', "YOUR ROOT PATH", VIDEO_ID)

# Generate from 5 to 10 minutes in the video
template.start_time = 60 * 5
template.end_time = 60 * 10

bardo = Bardo(template)
bardo.play()

fit_audio_in_video(template, VIDEO_ID)