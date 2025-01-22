#%% Imports & Constants
import os
import json
import pandas as pd

METRICS_FOLDER = '/home/felipe/Documents/Github/Pt-Brdo/experiments/cotw/results/metrics'
EPS_START_FILE = '/home/felipe/Documents/Github/Pt-Brdo/experiments/cotw/results/eps_start.json'

#METRICS_FOLDER = '/home/felipe/Documents/Github/Pt-Brdo/experiments/osni/results/metrics'
#EPS_START_FILE = '/home/felipe/Documents/Github/Pt-Brdo/experiments/osni/results/eps_start.json'

BARDOS = ['bardo_0', 'bardo_1', 'bardo_2', 'bardo_3']

#%%
def load_dict(path:str) -> dict:
    with open(path, 'r') as json_file:
        my_dict = json.load(json_file)

    return my_dict

#%% 
data_dict = {bardo:[] for bardo in BARDOS}
data_index = []

eps_start = load_dict(EPS_START_FILE)
last_video = list(eps_start.keys())[-1]
videos_ids = list(eps_start.items())

for idx, values in enumerate(videos_ids):
    video_id, start_time = values

    if start_time == None:
        continue

    metrics_file = os.path.join(METRICS_FOLDER, f"metrics_{video_id}.json")

    data_index.append(f"Ep {idx+1} ({video_id}) BCE:")
    #data_index.append(f"Ep {idx+1} KLD:")

    for bardo in BARDOS:
        video_metrics = load_dict(metrics_file)
        mkld = round(float(video_metrics[bardo]['bce']['mean']), 5)
        data_dict[bardo].append(mkld)

        #if video_id == last_video:
        #    fad = round(float(video_metrics[bardo]['fad']), 5)
        #    data_dict[bardo].append(fad)

    #if video_id == last_video:
    #    data_index.append("FAD:")

# %%
df = pd.DataFrame(data_dict, index=data_index)

df
df.style.highlight_min(color = 'blue', axis = 1)
#df.style.text_gradient(cmap="RdYlGn", vmin=2, vmax=15)

# %%
