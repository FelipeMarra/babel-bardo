# Babel Bardo
## Introdution
Babel bardo is a system designed to generate music for Tabletop Role-Playing Games (TRPGs) in real-time. The system works in a 30 seconds window, by executing the following procedure:

```
    For each 30s of gamplay:
        Extract the players dialogs transcriptions with a Speech Recognition (SR) system
        Use a LLM to transform the transcription in a music description
        Feed a Text-to-music model (TTM) with the music description
        Play the generated piece of music
```

A visual representation of the system can be seen in the Figure 1. 

![Figure 1. And overview o the Babel Bardo system](/assets/bardo_overview.png)

By prompting the LLM in different ways we obtained different versions of the system. For more details head towards the paper here. The following list presents the nomeclature difference between the systems in the paper and the ones presented in this repository:

* Babel Bardo - Baseline (B): Bardo 1
* Babel Bardo - Emotion (E): Bardo 0
* Babel Bardo - Description (D): Bardo 2
* Babel Bardo - Description Continuation (DC): Bardo 3

## Installation
#TODO collab

## Usage
#TODO

## Techinical Details
#TODO