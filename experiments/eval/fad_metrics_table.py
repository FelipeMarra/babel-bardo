#%% Imports & Constants
import os
import json

import pandas as pd
import matplotlib.pyplot as plt

RPG = 'cotw'
METRICS_FOLDER = f'/home/felipe/Documents/Github/Pt-Brdo/experiments/{RPG}/results/metrics'
EPS_START_FILE = f'/home/felipe/Documents/Github/Pt-Brdo/experiments/{RPG}/results/eps_start.json'

BARDOS = ['bardo_1', 'bardo_0', 'bardo_2', 'bardo_3']

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
valid_vides_ids = [video_id for video_id, start_time in videos_ids if start_time != None]

latex_table = ""
overall_row = ""
added_overall = False

for v_idx, video_id in enumerate(valid_vides_ids):
    metrics_file = os.path.join(METRICS_FOLDER, f"fad_{video_id}.json")

    #data_index.append(f"Ep {idx+1} ({video_id}) KLD Mean|Std:")
    latex_table += "\multicolumn{1}{c}{\\textbf{"+f"{v_idx+1}"+"}} & "
    data_index.append(f"Ep {v_idx+1}")

    for b_idx, bardo in enumerate(BARDOS):
        video_metrics = load_dict(metrics_file)
        fad_ep = round(float(video_metrics[bardo]['ep']), 2)

        if b_idx == len(BARDOS)-1:
            latex_table += f"{fad_ep} \\\\ \n"
        else:
            latex_table += f"{fad_ep} & "

        data_dict[bardo].append(f"{fad_ep}")

        if video_id == last_video:
            fad_overall = round(float(video_metrics[bardo]['overall']), 2)
            data_dict[bardo].append(fad_overall)

            if not added_overall:
                overall_row += "\multicolumn{1}{c}{\\textbf{Overall}} &"
                added_overall = True

            if b_idx == len(BARDOS)-1:
                overall_row += f"{fad_overall} \\\\ \n"
            else:
                overall_row += f"{fad_overall} & "

    if video_id == last_video:
        data_index.append("Overall:")

latex_table += overall_row

# %%
print(latex_table)

# %%
df = pd.DataFrame(data_dict, index=data_index)

#df
df.style.set_caption("Mean KLD for COTW").highlight_min(color='blue', axis=1)