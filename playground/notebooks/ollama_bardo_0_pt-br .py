# %%
from babel_bardo import OllamaChat, TranscriptIter
from itertools import islice

# %%
SEED = 2147483647
ollamaChat = OllamaChat(SEED)

# %%
VIDEO_ID = 'Pf4HzTdA2WE'
START_TIME = 15529
END_TIME = 15829
t_iter = TranscriptIter(VIDEO_ID, start_time=START_TIME, end_time=END_TIME)

# %%
common_setup = "You are going to receive a series of Role-playing Game (RPG) video transcript excerpts from players dialogs playing a campaing called O Segredo na Ilha. "
task_setup = "You will classify each dialog into one of the following emotions: Happy, Calm, Agitated, or Suspenseful. Your answer will be just one word, that is, one of those exact emotions."
prompt_setup = common_setup + task_setup

res = ollamaChat.send(prompt_setup, setup=True)
print(res)

# %%
for description in islice(t_iter, 20):
    frases, _ = description
    res = ollamaChat.send(frases)

    print("Frases:", frases)
    print("Ollama:", res)
    print(ollamaChat.chat_state)
    print(len(ollamaChat.chat_state))
    print()


