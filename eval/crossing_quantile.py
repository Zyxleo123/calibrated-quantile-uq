import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import argparse
import pickle
from tqdm import tqdm
from data.fetch_data import get_uci_data

Q_LIST = torch.arange(0.01, 1.0, 0.01)

SAVE_PATH = '/home/scratch/yixiz/eval/'

def compute_min_swaps(p):
    n = len(p)
    visited = [False] * n
    cycles = 0
    for i in range(n):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = p[j]
    return n - cycles

def load_models_and_metrics(models_path):
    with open(models_path, 'rb') as f:
        pkl = pickle.load(f)
        models = pkl['models']
        va_sharp = pkl['va_sharp_list']
        va_ece = pkl['va_ece_list']
    te_metrics_path = models_path.replace('_models', '')
    with open(te_metrics_path, 'rb') as f:
        pkl = pickle.load(f)
        te_sharp = pkl['te_sharp_score_controlled']
        te_ece = pkl['te_ece_controlled']
    return {
        'models': models,
        'va_sharp': va_sharp,
        'va_ece': va_ece,
        'te_sharp': te_sharp,
        'te_ece': te_ece
    }    

def main(args):
    seed, data = parse_models_path(args.models_path)
    args.seed = seed
    args.data = data
    args.data_dir = 'data/UCI_Datasets'

    x_te = get_uci_data(args).x_te.cuda()
    models_and_metrics = load_models_and_metrics(args.models_path)
    models = models_and_metrics['models']
    result_dict = models_and_metrics.copy()
    del result_dict['models']

    avg_swaps_list = []
    for i, model in tqdm(enumerate(models), total=len(models)):
        q_pred = model.predict_q(x_te, Q_LIST) # [n_te, n_q]
        q_pred_np = q_pred.cpu().numpy()
        sorted_indices = q_pred_np.argsort(axis=1)
        tot_swaps = 0
        for row_indices in sorted_indices:
            tot_swaps += compute_min_swaps(row_indices)
        avg_swaps = tot_swaps / q_pred_np.shape[0]
        avg_swaps_list.append(avg_swaps)

    result_dict['avg_swaps'] = avg_swaps_list
    save_path = SAVE_PATH + os.path.basename(args.models_path).replace('_models', '_crossing_quantile')
    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f)
    print(f"Saved crossing quantile results to {save_path}")
    return result_dict

def parse_models_path(models_path):
    parts = os.path.basename(models_path).split('_')
    seed = None
    for part in parts:
        if part.isdigit():
            seed = int(part)
            break
    data = parts[0]
    return seed, data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)