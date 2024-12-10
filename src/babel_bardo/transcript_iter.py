from typing import Tuple, List
import math
import json
import os
import shutil

from .constants import TRANSCRIPTS_CACHE

from youtube_transcript_api import YouTubeTranscriptApi as ytta

class TranscriptIter():

    def __init__(self, video_id:str, language:str='en', tgt_duration:int=30, join:bool=True, start_time:int=0, end_time:int|None=None) -> None:
        """An iterator that goes over the video transcription returning the frases 
        that fit in the interval specified by the tgt_duration parameter 
        e.g., if tgt_duration = 30, the first iteration will return frases between 0 and 30s.
        The second iteration will return frases between 30 and 60s, and so on.

        Parameters:
            video_id (str): YouTube video ID in the form tZWU5iPjQpI.
            tgt_duration (int): Target duration in seconds.
            join (bool): If false return a list of frases, returns a single string otherwise.
            start (int): From which second the iterator should start
            end (int or none): Until which second the iterator should go. If none it will go until the end.

        Yields:
            frases (Tuple[str, float]): A tuple containing a the frases and the duration from the start of the first frase to the end of the last one.
        """
        self.video_id = video_id
        self.language = language
        self.tgt_duration = tgt_duration
        self.join = join
        self.start_time = start_time
        self.end_time = end_time

        self.iter_len = -1 # init as an invalid value, gets setted at __iter__

    def _get_num_of_iters(self) -> int:
        "Calculates the number of iterations according to the transcription and the tgt_duration"

        if self.end_time == None:
            last_frase = self.transcription[-1]
            last_frase_end = last_frase['start'] + last_frase['duration']
        else:
            last_frase_end = self.end_time

        first_frase_start = self.transcription[self.frase_pointer]['start']
        return math.ceil((last_frase_end - first_frase_start) / self.tgt_duration)

    def __len__(self):
        return self.iter_len

    def _load_transcript(self) -> None:
        "Gets transcript from cache if it exists, else pulls it from the YouTube API. Transcription is stored in self.transcription"

        if not os.path.isdir(TRANSCRIPTS_CACHE):
            os.makedirs(TRANSCRIPTS_CACHE)

        cached_transcripts = os.listdir(TRANSCRIPTS_CACHE)
        transcript_file = self.video_id + ".json"
        transcript_file_path = TRANSCRIPTS_CACHE.joinpath(transcript_file)

        if(transcript_file in cached_transcripts):
            with open(transcript_file_path, 'r') as json_file:
                self.transcription = json.load(json_file)
            print("Using cached transcript located at", transcript_file_path)
        else:
            self.transcription = ytta.get_transcript(self.video_id, languages=[self.language])
            with open(transcript_file_path, "w") as json_file: 
                json.dump(self.transcription, json_file)

    def _jump_to_start(self):
        first_frase = self.transcription[self.frase_pointer]
        current_start = first_frase['start']

        while current_start < self.start_time:
            self.frase_pointer += 1
            current_frase = self.transcription[self.frase_pointer]
            current_start = current_frase['start']

    @staticmethod
    def clear_transcript_cache() -> None:
        print("Cleared transcript cache")
        shutil.rmtree(TRANSCRIPTS_CACHE)

    def __iter__(self) -> Tuple[List[str], float]:
        self._load_transcript()

        self.frase_pointer = 0 # Pointer to the next transcription frase
        self._jump_to_start()

        current_start = self.transcription[self.frase_pointer]['start']
        self.current_tgt = current_start  + self.tgt_duration # gets added by tgt_duration for each next()

        self.iter_len = self._get_num_of_iters()

        return self

    def __next__(self):
        if self.frase_pointer == len(self.transcription):
            raise StopIteration

        frases = []
        actual_duration = 0
        first_frase = self.transcription[self.frase_pointer]
        next_end = -1

        # while start + duration fits in the current_tgt interval
        while next_end <= self.current_tgt:

            frase = self.transcription[self.frase_pointer]
            text, start, duration = frase.values()
            current_end = start + duration

            if self.end_time != None and current_end > self.end_time:
                self.frase_pointer = len(self.transcription)
                break

            frases.append(text)
            self.frase_pointer += 1
            actual_duration = current_end - first_frase['start']

            if self.frase_pointer == len(self.transcription):
                break

            next_frase = self.transcription[self.frase_pointer]
            next_end = next_frase['start'] + next_frase['duration']

        self.current_tgt += self.tgt_duration

        if len(frases) == 0:
            raise StopIteration

        if self.join:
            return ' '.join(frases), actual_duration
        else:
            return frases, actual_duration