from babel_bardo.eval_metrics import get_fad_vggish

back_path = '/root/soudtrack'
eval_path = '/root/Pt-Brdo/experiments/cotw/results/original/audios'
overall_fad, ep_fad, wav_background_path = get_fad_vggish(back_path, eval_path, None, remove_back = False)

print(overall_fad, ep_fad, wav_background_path)