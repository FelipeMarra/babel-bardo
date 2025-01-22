# Babel Bardo
## Introdution
Babel bardo is a system designed to generate music for Tabletop Role-Playing Games (TRPGs) in real-time. The system works in a 30 seconds window, by executing the following procedure:

```
For each 30s of gamplay:
    Extract the players dialogs transcriptions with a Speech Recognition (SR) system
    Use a LLM to transform the transcription in a music description
    Feed a Text-to-Music model (TTM) with the music description
    Play the generated piece of music (in practice we are concatenating the generated segments and saving on disk after 30min)
```

A visual representation of the system can be seen in the Figure 1. 

![Figure 1. And overview o the Babel Bardo system](/assets/bardo_overview.png)

By prompting the LLM in different ways we obtained different versions of the system. For more details head towards the ***paper [here](https://arxiv.org/abs/2411.03948)***. The following list presents the nomeclature difference between the systems in the paper and the ones presented in this repository:

* Babel Bardo - Baseline (B): Bardo 1
* Babel Bardo - Emotion (E): Bardo 0
* Babel Bardo - Description (D): Bardo 2
* Babel Bardo - Description Continuation (DC): Bardo 3

To listen to some results access the demo through [this link](https://felipemarra.github.io/babel-bardo/)

## Instalation
Clone the repo and cd into it and install as a local package with pip.

``` bash
git clone https://github.com/FelipeMarra/babel-demo.git
cd babel-bardo
# you might want to create a virtual env here
python3 -m pip install -e .
```

Since the LLM used in here is Ollama, the system will be expecting an endpoint to a instanciation of the [Ollama API](https://github.com/ollama/ollama/blob/main/README.md#rest-api) in the enviroment variable `OLLAMA_ADDRES`

``` bash
export OLLAMA_ADDRES = my_ollama_adderes
```

You can setup the [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) configurations for the system in the [consts.py](https://github.com/FelipeMarra/babel-bardo/blob/main/src/babel_bardo/constants.py) file.

To obtain the paper results we used Ollama 3.1, with 70B parameters, and MuscGen large, with 3.3B parameters. 

> One can use MusicGen's small model for testing the system locally if your Ollama API is on a server elsewhere. It runs on a GTX 1050 with 4G of VRAM. In order to do so, set `MODEL = 'facebook/musicgen-small'` in the consts file.

## Creating and running experiments
In the [experiments folder](https://github.com/FelipeMarra/babel-bardo/tree/main/experiments) you'll find the experiments for the RPGs Call of the Wild (COTW) and O Segredo Na Ilha (OSNI), featured in the demo.

Let's take cotw as an example and recreate it's experiment in a concise form.

### Bardo Template
> To jump to ahead to the final code of this tutorial check the [tutorial.py](https://github.com/FelipeMarra/babel-bardo/blob/main/experiments/tutorial.py) file.

The first thing you need to notice is the [Babel Bardo Template](https://github.com/FelipeMarra/babel-bardo/blob/main/src/babel_bardo/templates.py). It defines the youtube video that will be used, the folder structure, sets the configuration for Ollama and the header for the logs. You can create your won Bardo, or use one of the existent presets. 

Let's supose you chose to use Description Continuation (DC), which corresponds to the preset Bardo3. So we'll instanciate this template passing the `rpg_name`, the `root_path` for the folder structure and the `video_id`. The video id for the firt episode of COTW with link https://www.youtube.com/watch?v=tZWU5iPjQpI will be `tZWU5iPjQpI`.

``` python
from babel_bardo.templates import Bardo3
from babel_bardo import Bardo, fit_audio_in_video

template = Bardo3('Call Of The Wild', "my_root_path", "tZWU5iPjQpI")
```

You can set the excerpt of the video you that want the system to generate a song to.

``` python
# Generate from 5 to 10 minutes in the video
template.start_time = 60 * 5
template.end_time = 60 * 10
```

Then all you need to do is to ask Bardo to play you a song

``` pytohn 
bardo = Bardo(template)
bardo.play()
```

And to fit the generated piece into the video you asked it to generate a piece for

``` pytohn 
fit_audio_in_video(template, "tZWU5iPjQpI")
```

The generated audio pieces and videos with the pieces fitted in it will be saved in folders following the template structure. The generated audio will be at `my_root_path/bardo_3/audios/generated/` and the fitted video will be at `my_root_path/bardo3/videos/generated` 

## Technical Details
#TODO

Please email felipeferreiramarra@gmail.com with the title "Babel Bardo Docs" to get notified when this documentation is updated.