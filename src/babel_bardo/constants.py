import pathlib

TRANSCRIPTS_CACHE = pathlib.Path(__file__).parent.joinpath("cache", "transcripts").resolve()

MODEL = 'facebook/musicgen-small'
EXTEND_STRIDE = 10
DURATION = 30
CROSSFADE_DURATION = 1

SEED = 2147483647