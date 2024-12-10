import os
from pathlib import Path
from abc import ABC, abstractmethod
import random

from babel_bardo.ollama_api import PromptConfig, OllamaType
from babel_bardo.constants import SEED

# Bardo and fit_audio_in_video will follow the following directory structure:
# |_original
# .  |_videos
# .  .  |_VID-N.mp4
# .  |_audios
# .  .  |_vocals
# .  .    |_vocals_VID-N.wav
# .  .  |_soundtrack
# .  .    |_VID-N.wav
# bardoX
# |_logs
# .  |_bardoX_VID-N.txt
# |_audios
# .  |_generated
# .  .  |_bardoX_VID-N.wav
# |_videos
# .  |_generated
# .  .  |_bardoX_VID-N.mp4
# BardoTemplate will already contain variables pointing to this structure to avoid file path conflicts

class BardoTemplate(ABC):
    """
        Abstract class to define a template for the Bardo properties.
        It uses abstract methods as properties to force the implementation
        of those properties when creating a new Bardo based on this template. 
    """
    def __init__(self, rpg_name:str, root_path:str|Path, video_id:str, seed:int|None=SEED, language='en') -> None:
        self.rpg_name = rpg_name
        self.video_id = video_id
        self.seed = seed
        self.language = language

        # Start and end time cant be setted as functions becaus of randon_excerpt
        self.start_time = 0
        self.end_time = None

        # PATHS
        self.root_path = root_path

        # metrics paths
        self.metrics_path = os.path.join(self.root_path, 'metrics')
        self.kld_file = os.path.join(self.metrics_path, f"kld_{self.video_id}.json")
        self.fad_file = os.path.join(self.metrics_path, f"fad_{self.video_id}.json")

        self.transitions_eval_path = os.path.join(self.root_path, 'transitions_eval')
        self.transitions_eval_audios_path = os.path.join(self.transitions_eval_path, 'audios', self.bardo_name, self.video_id)
        self.transitions_eval_metrics_path = os.path.join(self.transitions_eval_path, 'metrics')
        self.transitions_kld_metrics_file = os.path.join(self.transitions_eval_metrics_path, f"t_kld_{self.video_id}.json")

        self.segments_transitions_eval_path = os.path.join(self.root_path, 'segments_transitions_eval')
        self.segments_transitions_kld_metrics_file = os.path.join(self.segments_transitions_eval_path, f"s_t_kld_{self.video_id}.json")

        # original_path
        self.original_path = os.path.join(self.root_path, 'original')
        self.original_videos_path = os.path.join(self.original_path, 'videos')
        self.original_audios_path = os.path.join(self.original_path, 'audios')
        self.original_vocals_path = os.path.join(self.original_path, 'vocals')

        # bardo_path
        self.bardo_path = os.path.join(root_path, self.bardo_name)

        self.log_path = os.path.join(self.bardo_path, 'logs')

        self.bardo_audios_path = os.path.join(self.bardo_path, 'audios')
        self.generated_audios_path = os.path.join(self.bardo_audios_path, 'generated')

        self.bardo_videos_path = os.path.join(self.bardo_path, 'videos')
        self.generated_videos_path = os.path.join(self.bardo_videos_path, 'generated')

    # Paths as functions to allow changes in the video_id
    @property
    def dirs_to_create(self) -> list[str]:
        return [
            self.generated_audios_path,
            self.original_audios_path,
            self.original_vocals_path,
            self.original_videos_path,
            self.generated_videos_path,
            self.log_path,
            self.metrics_path,
            self.transitions_eval_audios_path,
            self.transitions_eval_metrics_path,
            self.segments_transitions_eval_path,
        ]

    @property
    def eps_start_file(self) -> str:
        return os.path.join(self.root_path, 'eps_start.json')

    @property
    def log_file(self) -> str:
        return os.path.join(self.log_path, f"{self.bardo_name}_{self.video_id}.txt")

    @property
    def original_vocals_file(self) -> str:
        return os.path.join(self.original_vocals_path, f"vocals_{self.video_id}.wav")

    @property
    def original_audio_file(self) -> str:
        return os.path.join(self.original_audios_path, f"{self.video_id}.wav")

    @property
    def generated_audio_file(self) -> str:
        return os.path.join(self.generated_audios_path, f"{self.bardo_name}_{self.video_id}.wav")

    @property
    def generated_audio_file_no_ext(self) -> str:
        return os.path.join(self.generated_audios_path, f"{self.bardo_name}_{self.video_id}")

    @property
    def original_video_file_name(self) -> str:
        return f"{self.video_id}.mp4"

    @property
    def original_video_file(self) -> str:
        return os.path.join(self.original_videos_path, f"{self.video_id}.mp4")

    @property
    def generated_video_file(self) -> str:
        return os.path.join(self.generated_videos_path, f"{self.bardo_name}_{self.video_id}.mp4")

    @property
    @abstractmethod
    def ollama_type(self) -> OllamaType:
        pass

    @property
    def common_setup(self) -> str:
        return f"You are going to receive a series of Role-playing Game (RPG) video transcript excerpts from players dialogs playing a campaing called {self.rpg_name}. "

    @property
    @abstractmethod
    def prompt_config(self) -> PromptConfig:
        pass

    @property
    @abstractmethod
    def log_header(self) -> str:
        pass

    @property
    @abstractmethod
    def bardo_name(self) -> str:
        pass

    def set_random_excerpt(self, excerpt_length:int, video_length:int, ep_start:int|None) -> int:
        """
            Sets random start_time and defines end_time based on excerpt_length (in seconds).
        """
        if excerpt_length > video_length:
            #Keep the default start at 0 and end at None
            return 0

        if ep_start == None:
            range_end = video_length - excerpt_length
            ep_start = random.randint(0, range_end)

        self.start_time = ep_start
        self.end_time = self.start_time + excerpt_length

        return ep_start

class Bardo0(BardoTemplate):
    def __init__(self, rpg_name:str, root_path:str|Path, video_id:str, seed:int=SEED, language='en') -> None:
        super().__init__(rpg_name, root_path, video_id, seed, language)

    @property
    def ollama_type(self) -> OllamaType:
        return OllamaType.CHAT

    @property
    def prompt_config(self) -> PromptConfig:
        start = "Background music for a Role-playing Game (RPG) dialog, with the following emotion: "
        task_setup = "You will classify each dialog into one of the following emotions: Happy, Calm, Agitated, or Suspenseful. Your answer will be just one word, that is, one of those emotions."
        prompt_setup = self.common_setup + task_setup

        return PromptConfig(start=start, setup=prompt_setup)

    @property
    def log_header(self) -> str:
        return f"""
            ################################################################################################
            # Bardo 0
            # This version imitates the behaviour of the original Bardo 
            # Conditioning MusicGen in one of the following emotions: Happy, Calm, Agitated, and Suspenseful
            #
            # Video: {self.video_id}
            # From {self.start_time} to {self.end_time}
            ################################################################################################
            """

    @property
    def bardo_name(self) -> str:
        return "bardo_0"

class Bardo1Test(BardoTemplate):
    def __init__(self, rpg_name:str, root_path:str|Path, video_id:str,  translate:bool, seed:int=SEED, language='en') -> None:
        super().__init__(rpg_name, root_path, video_id, seed, language)
        self.translate = translate
        self.start_time = 0
        self.end_time = 60

    @property
    def ollama_type(self) -> OllamaType:
        if self.translate:
            return OllamaType.CHAT

        return OllamaType.NONE

    @property
    def prompt_config(self) -> PromptConfig:
        start="Background music for the following Role-playing Game (RPG) dialog: "

        if self.translate:
            task_setup = "For each transcript excerpt you will translate the dialog to english. Your answer will only contain the translation."
            prompt_setup = self.common_setup + task_setup

            return PromptConfig(start=start, setup=prompt_setup)

        return PromptConfig(start=start)

    @property
    def log_header(self) -> str:
        return f"""
            ################################################################################################
            # BARDO 1 TEST
            # This version inputs the dialogs directly into MusicGen
            #
            # Video: {self.video_id}
            # From {self.start_time} to {self.end_time}
            ################################################################################################
            """

    @property
    def bardo_name(self) -> str:
        return "bardo_1_test"

class Bardo1(BardoTemplate):
    def __init__(self, rpg_name:str, root_path:str|Path, video_id:str,  translate:bool, seed:int=SEED, language='en') -> None:
        super().__init__(rpg_name, root_path, video_id, seed, language)
        self.translate = translate

    @property
    def ollama_type(self) -> OllamaType:
        if self.translate:
            return OllamaType.CHAT

        return OllamaType.NONE

    @property
    def prompt_config(self) -> PromptConfig:
        start="Background music for the following Role-playing Game (RPG) dialog: "

        if self.translate:
            task_setup = "For each transcript excerpt you will translate the dialog to english. Your answer will only contain the translation."
            prompt_setup = self.common_setup + task_setup

            return PromptConfig(start=start, setup=prompt_setup)

        return PromptConfig(start=start)

    @property
    def log_header(self) -> str:
        return f"""
            ################################################################################################
            # BARDO 1
            # This version inputs the dialogs directly into MusicGen
            #
            # Video: {self.video_id}
            # From {self.start_time} to {self.end_time}
            ################################################################################################
            """

    @property
    def bardo_name(self) -> str:
        return "bardo_1"

class Bardo2(BardoTemplate):
    def __init__(self, rpg_name:str, root_path:str|Path, video_id:str, seed:int=SEED, language='en') -> None:
        super().__init__(rpg_name, root_path, video_id, seed, language)

    @property
    def ollama_type(self) -> OllamaType:
        return OllamaType.CHAT

    @property
    def prompt_config(self) -> PromptConfig:
        task_setup = "For each transcript excerpt you will, in english, describe a piece of background music that matches that excerpt. Your answer will be fed to a text to music model and should only contain a music description. The description should be concise. Do not mention any dialog excerpt details. Here is a prompt that worked well with the music model: A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle."
        prompt_setup = self.common_setup + task_setup

        return PromptConfig(setup=prompt_setup)

    @property
    def log_header(self) -> str:
        return f"""
            ################################################################################################
            # Bardo 2
            # This version setups Ollama to describe songs for MusicGen given the dialogs
            #
            # Video: {self.video_id}
            # From {self.start_time} to {self.end_time}
            ################################################################################################
            """

    @property
    def bardo_name(self) -> str:
        return "bardo_2"

class Bardo3(BardoTemplate):
    def __init__(self, rpg_name:str, root_path:str|Path, video_id:str, seed:int=SEED, language='en') -> None:
        super().__init__(rpg_name, root_path, video_id, seed, language)

    @property
    def ollama_type(self) -> OllamaType:
        return OllamaType.CHAT

    @property
    def prompt_config(self) -> PromptConfig:
        task_setup = "Your task is to determine whether each excerpt is from the same campaign chapter as the previous one, and based on this determination, either return the word 'CONTINUE.' or generate a music description in english. If no previous transcript has been provided, consider that the current excerpt is the beginning of a new chapter. For each transcript excerpt you will describe a piece of background music that matches that excerpt. If the excerpt is part of the same story chapter as the previous excerpt, the given answer should only contain the word 'CONTINUE.' Your description will be fed to a text to music model. The description should be concise. Do not mention anything about the dialog excerpt. Here is a prompt that worked well with the music model: A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle."
        prompt_setup = self.common_setup + task_setup

        return PromptConfig(setup=prompt_setup)

    @property
    def log_header(self) -> str:
        return f"""
            ################################################################################################
            # BARDO 3
            # This version also setups Ollama to describe songs for MusicGen given the dialogs, but only if
            # needed. If the model finds that MusicGen can just continue from the previously generated 
            # audio tokens, without a text description, than that is what we're going to do. By doing so,
            # we expect to avoid transition problems.
            #
            # Video: {self.video_id}
            # From {self.start_time} to {self.end_time}
            ################################################################################################
            """

    @property
    def bardo_name(self) -> str:
        return "bardo_3"
