import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pickle
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plots.plot_utils import load_pickle

RESULTS_BASE_DIR = os.path.join(os.environ['SCRATCH'], "results")
BASELINE_NAMES = ['batch_qr', 'batch_cal', 'batch_int']
FALLBACK_MAX_THRESHOLDS = 150 # Fallback if num_thres cannot be read from data

def aggregate_seeds_median(seed_pkls, quantile=0.5):
    """
    Loads data from seed pkls, aggregates them by taking the median
    performer at each threshold (based on validation sharpness), and returns the metrics.
    """
    if not seed_pkls:
        return {}

    datas = [d for d in [load_pickle(pkl) for pkl in seed_pkls] if d is not None]
    if not datas:
        return {}

    num_seeds = len(datas)
    
    num_thres = datas[0]['args'].num_thres
    if num_thres is None:
        print(f"Warning: Could not find 'num_thres' in args. Using fallback value: {FALLBACK_MAX_THRESHOLDS}")
        num_thres = FALLBACK_MAX_THRESHOLDS

    # Initialize arrays for padding
    va_sharp = np.full((num_seeds, num_thres), 1.0)
    te_sharp = np.full((num_seeds, num_thres), 1.0)
    te_ece = np.full((num_seeds, num_thres), 1.0)

    # Populate arrays, aligning to the right
    for i, data in enumerate(datas):
        for key, arr in [('va_sharp_score_thresholded', va_sharp),
                         ('te_sharp_score_thresholded', te_sharp),
                         ('te_ece_thresholded', te_ece)]:
            if key in data and len(data[key]) > 0:
                arr[i, -len(data[key]):] = data[key]

    # Find the indices of the median seed for each threshold based on validation sharpness
    order = np.argsort(va_sharp, axis=0)
    q_idx = int(quantile * (num_seeds - 1))
    median_seed_indices = order[q_idx, :]

    # Retrieve test scores based on these median seed indices
    threshold_to_test_metrics = {}
    threshold_values = np.linspace(0.001, 0.15, num_thres)

    for thres_idx, threshold in enumerate(threshold_values):
        seed_idx = median_seed_indices[thres_idx]
        current_te_ece = te_ece[seed_idx, thres_idx]
        current_te_sharp = te_sharp[seed_idx, thres_idx]
        
        if current_te_ece == 1.0 and current_te_sharp == 1.0:
            continue
            
        threshold_to_test_metrics[float(threshold)] = {
            'te_ece': float(current_te_ece), 
            'te_sharp_score': float(current_te_sharp)
        }
    
    return threshold_to_test_metrics

def plot_ece_exceedance(ax, threshold_to_metrics, baseline_name):
    """Plots Test ECE Exceedance vs. Threshold for a single baseline."""
    if not threshold_to_metrics:
        print(f"  - No data to plot for ECE Exceedance (baseline: {baseline_name})")
        return

    thresholds = np.array(list(threshold_to_metrics.keys()))
    te_ece_values = np.array([d['te_ece'] for d in threshold_to_metrics.values()])
    
    exceedance = np.maximum(te_ece_values - thresholds, 0)
    is_exceeding = te_ece_values > thresholds
    
    below_prop = np.mean(~is_exceeding)
    mean_exceedance = np.mean(exceedance[is_exceeding]) if np.any(is_exceeding) else 0.0

    label = f"{baseline_name}: p={below_prop:.1%} m={mean_exceedance:.3f}"
    ax.scatter(thresholds, exceedance, label=label, alpha=0.7, s=10)

def plot_sharpness_vs_ece(ax, threshold_to_metrics, baseline_name):
    """Plots Test Sharpness vs. Test ECE for a single baseline."""
    if not threshold_to_metrics:
        print(f"  - No data to plot for Sharpness vs. ECE (baseline: {baseline_name})")
        return

    te_ece = [d['te_ece'] for d in threshold_to_metrics.values()]
    te_sharp = [d['te_sharp_score'] for d in threshold_to_metrics.values()]
    
    ax.scatter(te_ece, te_sharp, label=baseline_name, alpha=0.7, s=10)

# --- Main Script Logic ---

def main(args):
    exp_name = args.exp_name
    """Main function to find all experiment configs and generate plots."""
    exp_dir = Path(RESULTS_BASE_DIR) / exp_name
    if not exp_dir.is_dir():
        print(f"Error: Experiment directory not found at '{exp_dir}'")
        return

    print(f"--- Starting analysis for experiment: {exp_name} ---")
    hyperparam_dirs = [d for d in exp_dir.glob('*/*') if d.is_dir()]

    if not hyperparam_dirs:
        print(f"No hyperparameter subdirectories found in {exp_dir}. Check directory structure.")
        return

    for hp_dir in sorted(hyperparam_dirs):
        dataset_name = hp_dir.parent.name
        hp_config_name = hp_dir.name
        print(f"\nProcessing: {dataset_name} / {hp_config_name}")

        # 1. Aggregate data for all baselines first
        all_baselines_data = {}
        for baseline in BASELINE_NAMES:
            pattern = f"*_loss{baseline}_*.pkl"
            seed_pkls = sorted(glob.glob(str(hp_dir / pattern)))

            if not seed_pkls:
                print(f"  - No files found for baseline: {baseline}")
                continue
            
            print(f"  - Found {len(seed_pkls)} seed(s) for baseline: {baseline}")
            all_baselines_data[baseline] = aggregate_seeds_median(seed_pkls, quantile=args.quantile)

        if not all_baselines_data:
            print(f"  - No data found for any baseline in this directory. Skipping plots.")
            continue
            
        # 2. Create and save the ECE Exceedance plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        for name, metrics in all_baselines_data.items():
            plot_ece_exceedance(ax1, metrics, name)
        
        ax1.set_title(f"ECE Exceedance\nDataset: {dataset_name} | Config: {hp_config_name} | Quantile: {args.quantile:.0%}\n"
                      "(p = prop. below threshold; m = mean exceedance of exceeding points)")
        ax1.set_xlabel('ECE Threshold')
        ax1.set_ylabel('max(Test ECE - Threshold, 0)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        output1 = hp_dir / "ece_exceedance_plot.png"
        plt.savefig(output1, bbox_inches='tight')
        plt.close(fig1)

        # 3. Create and save the Sharpness vs. ECE plot
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for name, metrics in all_baselines_data.items():
            plot_sharpness_vs_ece(ax2, metrics, name)

        ax2.set_title(f"Sharpness vs. ECE\nDataset: {dataset_name} | Config: {hp_config_name} | Quantile: {args.quantile:.0%}")
        ax2.set_xlabel('Test ECE')
        ax2.set_ylabel('Test Sharpness Score')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        output2 = hp_dir / "sharpness_vs_ece_plot.png"
        plt.savefig(output2, bbox_inches='tight')
        plt.close(fig2)

        print(f"  -> Plots saved to {hp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate and plot experiment results from a structured directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_name", 
        "-n",
        type=str, 
        help="The name of the experiment to process (e.g., 'run')."
    )
    parser.add_argument(
        "--quantile",
        "-q",
        type=float,
        help="The quantile taken over all seeds"
    )
    args = parser.parse_args()
    
    main(args)