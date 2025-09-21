import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import get_pareto_front
from plots.plot_utils import (
    load_pickle,
    RESULTS_BASE_DIR,
    BASELINE_NAMES,
    SEEDS,
    Y_METRICS,
    TITLE_METRICS,
    TITLE_METHODS,
    COLORS,
    METHOD_COLORS,
    METHOD_MARKERS,
    PERFORMANCE_METRICS,
    HYPER_MARKERS,
    HYPER_COLORS
)
from typing import List, Tuple, Dict

def get_pf_for_hp_dir(hp_dir: Path, key_prefix: str) -> List[Tuple[float, float]]:
    all_pkls = hp_dir.glob('*.pkl')
    all_data = [load_pickle(p) for p in all_pkls]
    all_points = []
    for data in all_data:
        all_points.extend(zip(
            data[f'{key_prefix}te_ece_controlled'],
            data[f'{key_prefix}te_sharp_score_controlled'])
        )
    pf = get_pareto_front(all_points)
    return pf

def plot_pareto_fronts(pf_by_hp: Dict[str, List[Tuple[float, float]]], dataset_name: str, title_suffix: str, filename_suffix: str, save_dir: Path):
    plt.figure(figsize=(10, 8))
    for hp_config_name, pf in sorted(pf_by_hp.items()):
        hp_config_name = 'nl-2_hs-64' if hp_config_name == 'default' else hp_config_name
        if not pf:
            print(f"Warning: No data points found for {dataset_name} / {hp_config_name}. Skipping plot.")
            continue
        pf = sorted(pf)
        x, y = zip(*pf)
        plt.plot(
            x, y, 
            label=hp_config_name,
            color=HYPER_COLORS[hp_config_name],
            markersize=8
        )
    plt.xlabel('Test ECE')
    plt.ylabel('Test Sharpness')
    plt.title(f'Pareto Fronts - {dataset_name}{title_suffix}', fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir / f'pareto_fronts{filename_suffix}.png')
    plt.close()


def main(args):
    """Main function to find all experiment configs and generate plots."""
    if args.recalibrated:
        key_prefix = 'recal_'
        title_suffix = ' (Recalibrated)'
        filename_suffix = '_recalibrated'
    else:
        key_prefix = ''
        title_suffix = ''
        filename_suffix = ''
    
    exp_name = args.exp_name
    exp_dir = Path(RESULTS_BASE_DIR) / exp_name

    print(f"--- Starting analysis for experiment: {exp_name} ---")
    if args.recalibrated:
        print("--- Plotting RECALIBRATED data ---")

    dataset_dirs = [d for d in exp_dir.glob('*') if d.is_dir()]

    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name
        hp_dirs = dataset_dir.glob('*')
        pf_by_hp = {}
        for hp_dir in sorted(hp_dirs):
            if not hp_dir.is_dir():
                continue
            hp_config_name = hp_dir.name
            print(f"\nProcessing: {dataset_name} / {hp_config_name}")
            pf_by_hp[hp_config_name] = get_pf_for_hp_dir(hp_dir, key_prefix)
        plot_pareto_fronts(pf_by_hp, dataset_name, title_suffix, filename_suffix, dataset_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", '-n', type=str, required=True)
    parser.add_argument("--recalibrated", '-r', action='store_true', help="Whether to plot recalibrated results.")
    args = parser.parse_args()
    main(args)