from typing import List
import os
import pathlib
from time import perf_counter

import torch
import random
import numpy as np

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

SEED = 2147483647
SCRIPTS_PATH = pathlib.Path(__file__).parent.resolve()
SAVE_AUDIOS_PATH = SCRIPTS_PATH.joinpath('continuation_test_audios')
DESCRIPTION = ['Imagine a track that begins with a soft, mystical ambiance, capturing the essence of stepping into an epic fantasy world. Gentle strings and ethereal chimes create an air of anticipation, setting the stage for the grand adventure ahead. As the dungeon master welcomes the players, the music subtly swells, introducing a light, rhythmic pulse that hints at the excitement and challenges to come. This composition should evoke a sense of camaraderie and heroism, with melodic undertones that reflect the promise of epic quests and the bond between the adventurers. Perfect for immersing players into the magical world of Dungeons & Dragons right from the start of their campaign.']

print('cuda.is_available:', torch.cuda.is_available())

class TestParams():
    def __init__(self, use_sampling: bool = True, top_k: int = 250,
                    top_p: float = 0.0, temperature: float = 1.0,
                    duration: float = 30.0, cfg_coef: float = 3.0,
                    two_step_cfg: bool = False, extend_stride: float = 18,
                    cut_prompt:bool=False, description=None):
        """
            cut_prompt = If true prompt will be the overlap size (30-extend_strid).
                        Otherwise the prompt will be cutted off to 29s (the maximum allowed) if needed.
        """
        self.use_sampling = use_sampling
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.duration = duration
        self.cfg_coef = cfg_coef
        self.two_step_cfg = two_step_cfg
        self.extend_stride = extend_stride
        self.cut_prompt = cut_prompt
        self.description = description

    def __repr__(self):
        return f"use_sampling={self.use_sampling}, \ntop_k={self.top_k}, \ntop_p={self.top_p}, \ntemperature={self.temperature}, \nduration={self.duration}, \ncfg_coef={self.cfg_coef}, \nextend_stride={self.extend_stride}, \ncut_prompt={self.cut_prompt}, \ndescription={self.description} \n"

class ContinuationTest():
    def __init__(self, test_params_list: List[TestParams]) -> None:
        self.test_params_list = test_params_list

    def log(self, path):
        lines = []

        for test_param in self.test_params_list:
            lines.append(str(test_param))

        with open(path, 'a') as file:
            for line in lines:
                file.write(line+'\n')

    def run(self, path:pathlib.Path, idx:int):

        model = MusicGen.get_pretrained('facebook/musicgen-small')
        previous_out = None
        previous_tokens = None

        test_output = None
        test_tokens = None

        print("################ STARTING TEST:", idx)
        for test_param in self.test_params_list:
            start_test = perf_counter()

            model.set_generation_params(
                use_sampling = test_param.use_sampling,
                top_k = test_param.top_k,
                top_p = test_param.top_p,
                temperature = test_param.temperature,
                duration = test_param.duration,
                cfg_coef = test_param.cfg_coef,
                two_step_cfg = test_param.two_step_cfg,
                extend_stride = test_param.extend_stride
            )

            if previous_out == None:
                print("Previous was none so generating")
                previous_out, previous_tokens = model.generate(
                    descriptions=test_param.description,
                    return_tokens=True,
                    progress=True
                )
            else:
                print("Previous had shape", previous_out.shape, previous_tokens.shape)

                if test_param.cut_prompt:
                    overlap = model.duration - model.extend_stride
                    previous_out = previous_out[:, :, -overlap*model.sample_rate:]
                    previous_tokens = previous_tokens[:, :, -overlap*model.frame_rate:]
                    print("Cutted to overlap size", previous_out.shape, previous_tokens.shape)

                if previous_out.shape[-1] >= model.sample_rate * model.duration:
                    # Cut 1 second
                    previous_out = previous_out[:, :, -29*model.sample_rate:]
                    previous_tokens = previous_tokens[:, :, -29*model.frame_rate:]
                    print("Cutted 1 sec", previous_out.shape, previous_tokens.shape)

                previous_tokens = previous_tokens.to(model.device)
                previous_out = model.generate_continuation(
                    previous_out,
                    my_prompt_tokens= previous_tokens,
                    prompt_sample_rate = model.sample_rate,
                    descriptions=[test_param.description],
                    progress=True
                )

                print("Generated continuation with shape", previous_out.shape)

            end_test = perf_counter()
            print(f"It took {(end_test-start_test):>2f}seconds")

            if test_output == None:
                test_output = previous_out
                test_tokens = previous_tokens
            else:
                test_output = torch.cat((test_output, previous_out), dim=2)
                test_tokens = torch.cat((test_tokens, previous_tokens), dim=2)
            print("Test output currently has shape", test_output.shape)
            print()

        test_output = torch.squeeze(test_output, 0)
        audio_write(path, test_output.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        test_tokens = model.generate_audio(test_tokens)
        test_tokens = torch.squeeze(test_tokens, 0)
        audio_write(str(path)+'_from_tokens', test_tokens.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

# top_k == 100 looks fine.
# Lowering the temperature is not a good idea.
# Lowering extend_stride doesn't seem to have much effect.
def set_tests() -> List[ContinuationTest]:
    continuation_tests = []

    ### Test 1
    test1 = ContinuationTest(
        [
            TestParams(duration=30, description=DESCRIPTION, cut_prompt=True),
            TestParams(duration=30, extend_stride=10, cut_prompt=True),
        ]
    )
    continuation_tests.append(test1)
    # Previous was none so generating

    # ####################### _generate_tokens #######################
    # Started with prompt_length 0
    # frame_rate 50 extend_stride 18

    # While 0 < 2000
    # Generated size torch.Size([1, 4, 1500])
    # New prompt_length is 600
    # New current_gen_offset is 900
    # ############### ALL TOKENS: ##################
    # torch.Size([1, 4, 1500])

    # While 1500 < 2000
    # Generated size torch.Size([1, 4, 1100])
    # New prompt_length is 200
    # New current_gen_offset is 1800
    # ############### ALL TOKENS: ##################
    # torch.Size([1, 4, 1500])
    # torch.Size([1, 4, 500])
    # It took 780.3322400280013 seconds
    # Test output currently has shape torch.Size([1, 1, 1280000])

    # Previous had shape torch.Size([1, 1, 1280000])
    # Cutted 1 sec torch.Size([1, 1, 928000])

    # ####################### _generate_tokens #######################
    # Started with prompt_length 1450
    # frame_rate 50 extend_stride 10

    # While 1450 < 2000
    # Generated size torch.Size([1, 4, 1500])
    # New prompt_length is 1000
    # New current_gen_offset is 500
    # ############### ALL TOKENS: ##################
    # torch.Size([1, 4, 1450])
    # torch.Size([1, 4, 50])

    # While 1500 < 2000
    # Generated size torch.Size([1, 4, 1500])
    # New prompt_length is 1000
    # New current_gen_offset is 1000
    # ############### ALL TOKENS: ##################
    # torch.Size([1, 4, 1450])
    # torch.Size([1, 4, 50])
    # torch.Size([1, 4, 500])
    # Generated continuation with shape torch.Size([1, 1, 1280000])
    # It took 286.5574772749969 seconds
    # Test output currently has shape torch.Size([1, 1, 2560000])

    # Previous had shape torch.Size([1, 1, 1280000])
    # Cutted 1 sec torch.Size([1, 1, 928000])

    # ####################### _generate_tokens #######################
    # Started with prompt_length 1450
    # frame_rate 50 extend_stride 10

    # While 1450 < 2000
    # Generated size torch.Size([1, 4, 1500])
    # New prompt_length is 1000
    # New current_gen_offset is 500
    # ############### ALL TOKENS: ##################
    # torch.Size([1, 4, 1450])
    # torch.Size([1, 4, 50])

    # While 1500 < 2000
    # Generated size torch.Size([1, 4, 1500])
    # New prompt_length is 1000
    # New current_gen_offset is 1000
    # ############### ALL TOKENS: ##################
    # torch.Size([1, 4, 1450])
    # torch.Size([1, 4, 50])
    # torch.Size([1, 4, 500])
    # Generated continuation with shape torch.Size([1, 1, 1280000])
    # It took 286.1960641080004 seconds
    # Test output currently has shape torch.Size([1, 1, 3840000])

    ### Test 2
    test2 = ContinuationTest(
        [
            TestParams(duration=15, top_k=100, description=DESCRIPTION),
            TestParams(duration=30, top_k=100, extend_stride=5)
        ]
    )
    #continuation_tests.append(test2)

    ### Test 3
    test3 = ContinuationTest(
        [
            TestParams(duration=15, top_k=100, temperature=0.5, description=DESCRIPTION),
            TestParams(duration=30, top_k=100, temperature=0.5, extend_stride=5)
        ]
    )
    #continuation_tests.append(test3)

    ### Test 4
    test4 = ContinuationTest(
        [
            TestParams(duration=15, top_k=100, temperature=0, description=DESCRIPTION),
            TestParams(duration=30, top_k=100, temperature=0, extend_stride=5)
        ]
    )
    #continuation_tests.append(test4)

    ### Test 5
    test5 = ContinuationTest(
        [
            TestParams(duration=15, top_k=100, temperature=0, cfg_coef=1, description=DESCRIPTION),
            TestParams(duration=30, top_k=100, temperature=0, cfg_coef=1, extend_stride=5)
        ]
    )
    #continuation_tests.append(test5)

    ### Test 6
    test6 = ContinuationTest(
        [
            TestParams(duration=15, top_p=0.1, temperature=0, cfg_coef=1, description=DESCRIPTION),
            TestParams(duration=30, top_p=0.1, temperature=0, cfg_coef=1, extend_stride=5)
        ]
    )
    #continuation_tests.append(test6)

    ### Test 7
    test7 = ContinuationTest(
        [
            TestParams(duration=15, top_p=0.5, temperature=0, cfg_coef=1, description=DESCRIPTION),
            TestParams(duration=30, top_p=0.5, temperature=0, cfg_coef=1, extend_stride=5)
        ]
    )
    #continuation_tests.append(test7)

    ### Test 8
    test8 = ContinuationTest(
        [
            TestParams(duration=15, top_k=50, temperature=0.1, cfg_coef=3, description=DESCRIPTION),
            TestParams(duration=30, top_k=50, temperature=0.2, cfg_coef=3, extend_stride=5),
            TestParams(duration=30, top_k=50, temperature=0.3, cfg_coef=3, extend_stride=5)
        ]
    )
    #continuation_tests.append(test8)

    ### Test 9
    test9 = ContinuationTest(
        [
            TestParams(duration=30, top_k=50, temperature=0.1, cfg_coef=3, description=DESCRIPTION),
            TestParams(duration=30, top_k=50, temperature=0.1, cfg_coef=3, extend_stride=2),
            TestParams(duration=30, top_k=50, temperature=0.1, cfg_coef=3, extend_stride=2)
        ]
    )
    #continuation_tests.append(test9)

    ### Test 10
    test10 = ContinuationTest(
        [
            TestParams(duration=5, top_k=50, temperature=0.1, cfg_coef=3, description=DESCRIPTION),
            TestParams(duration=10, top_k=50, temperature=0.1, cfg_coef=3, extend_stride=1),
            TestParams(duration=15, top_k=50, temperature=0.1, cfg_coef=3, extend_stride=1)
        ]
    )
    #continuation_tests.append(test10)

    return continuation_tests

def run_tests():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    continuaiton_tests = set_tests()

    for idx, test in enumerate(continuaiton_tests):
        idx_1 = idx+1
        test_path = SAVE_AUDIOS_PATH.joinpath(str(idx_1))

        if not os.path.isdir(test_path):
            os.makedirs(test_path)

        test.log(test_path.joinpath(f"{idx_1}_log.txt"))
        test.run(test_path.joinpath(f"{idx_1}_audio"), idx_1)

run_tests()