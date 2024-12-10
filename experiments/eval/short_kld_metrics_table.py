#%% Imports & Constants
import os
import json

import pandas as pd
import numpy as np

RPG = 'cotw'
EPS_START_FILE = f'/home/felipe/Documents/Github/Pt-Brdo/experiments/{RPG}/results/eps_start.json'

METRICS_FOLDER = f'/home/felipe/Documents/Github/Pt-Brdo/experiments/{RPG}/results/metrics'
#METRICS_FOLDER = f'/home/felipe/Documents/Github/Pt-Brdo/experiments/{RPG}/results/transitions_eval/metrics'
#METRICS_FOLDER = f'/home/felipe/Documents/Github/Pt-Brdo/experiments/{RPG}/results/segments_transitions_eval'

BARDOS = ['bardo_1', 'bardo_0', 'bardo_2', 'bardo_3']

#%%
def load_dict(path:str) -> dict:
    with open(path, 'r') as json_file:
        my_dict = json.load(json_file)

    return my_dict

#%% 
data_dict = {bardo:[] for bardo in BARDOS}

eps_start = load_dict(EPS_START_FILE)
videos_ids = list(eps_start.items())
valid_vides_ids = [video_id for video_id, start_time in videos_ids if start_time != None]

latex_table = "\multicolumn{1}{c} & "

for v_idx, video_id in enumerate(valid_vides_ids):
    metrics_file = os.path.join(METRICS_FOLDER, f"kld_{video_id}.json")


    for b_idx, bardo in enumerate(BARDOS):
        video_metrics = load_dict(metrics_file)

        klds = video_metrics[bardo]['list'][1:-2].split(',')
        klds = [float(kld) for kld in klds]

        data_dict[bardo] = data_dict[bardo] + klds

for b_idx, bardo in enumerate(BARDOS):
    klds = np.array(data_dict[bardo]) 

    klds_mean = round(klds.mean(), 2)
    klds_std = round(klds.std(), 2)

    if b_idx == len(BARDOS)-1:
        latex_table += f"{klds_mean}$\pm${klds_std} \\\\ \n"
    else:
        latex_table += f"{klds_mean}$\pm${klds_std} & "

    data_dict[bardo] = [klds_mean]

print(data_dict)


# %%
print(latex_table)
# %%
df = pd.DataFrame(data_dict)

#df
df.style.set_caption("Mean KLD for COTW").highlight_min(color='blue', axis=1)