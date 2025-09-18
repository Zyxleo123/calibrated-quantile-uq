
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
from collections import defaultdict
import math
import pandas as pd

RESULTS_BASE_DIR = os.path.join(os.environ['SCRATCH'], "results")
BASELINE_NAMES = ['batch_cal', 'batch_int', 'batch_qr']
FALLBACK_MAX_THRESHOLDS = 150


def lump(all_points, window_frac=0.05):
    """
    Calculates moving median and quantiles on a set of points for visualization.
    Reused from plot_lump.py.
    """
    df = pd.DataFrame(all_points, columns=['x', 'y'])
    if df.empty:
        return [], [], []
        
    if window_frac == 0.0:
        df['median'] = df['y'].expanding(min_periods=1).median()
        df['upper'] = df['y'].expanding(min_periods=1).quantile(0.9)
        df['lower'] = df['y'].expanding(min_periods=1).quantile(0.1)
    else:
        window_size = max(1, int(len(df) * window_frac))
        df['median'] = df['y'].rolling(window=window_size, min_periods=window_size, center=True).median()
        df['upper'] = df['y'].rolling(window=window_size, min_periods=window_size, center=True).quantile(0.9)
        df['lower'] = df['y'].rolling(window=window_size, min_periods=window_size, center=True).quantile(0.1)

    median_line = list(zip(df['x'], df['median']))
    upper_line = list(zip(df['x'], df['upper']))
    lower_line = list(zip(df['x'], df['lower']))

    return median_line, upper_line, lower_line


def aggregate_seeds_median(seed_pkls, quantile=0.5, prefix=''):
    """
    Loads data from seed pkls, aggregates them by lumping all points and
    calculating an expanding median curve. This approach is adapted from plot_lump.py.

    Args:
        seed_pkls (list): List of paths to pickle files for different seeds.
        quantile (float): This argument is unused in the new implementation but kept for compatibility.
        prefix (str): A prefix (e.g., 'recal_') to prepend to dictionary keys.
    """
    if not seed_pkls:
        return {}

    # Define keys with the dynamic prefix
    te_sharp_key = f'{prefix}te_sharp_score_controlled'
    te_ece_key = f'{prefix}te_ece_controlled'

    # 1. Load and extract all (ECE, Sharpness) points from all seeds
    all_points = []
    for pkl_path in seed_pkls:
        data = load_pickle(pkl_path)
        if data and te_ece_key in data and te_sharp_key in data:
            xs = data[te_ece_key]
            ys = data[te_sharp_key]
            # Filter out placeholder/default values before pooling
            valid_indices = [i for i, (x, y) in enumerate(zip(xs, ys)) if not (x == 1.0 and y == 1.0)]
            if valid_indices:
                valid_xs = [xs[i] for i in valid_indices]
                valid_ys = [ys[i] for i in valid_indices]
                all_points.extend(zip(valid_xs, valid_ys))
    
    if not all_points:
        return {}

    # 2. Sort points by ECE (the x-axis)
    all_points.sort(key=lambda p: p[0])

    # 3. Lump the points to get the median curve using an expanding window
    median_line, _, _ = lump(all_points)

    # 4. Convert the median line into the dictionary format expected by downstream functions
    lumped_metrics = {}
    for i, (ece, sharp) in enumerate(median_line):
        # Using index as the key since 'threshold' is no longer directly available or relevant
        lumped_metrics[i] = {
            f'{prefix}te_ece': float(ece),
            f'{prefix}te_sharp_score': float(sharp)
        }

    return lumped_metrics

def plot_sharpness_vs_ece(ax, threshold_to_metrics, baseline_name, prefix=''):
    """Plots Test Sharpness vs. Test ECE for a single baseline."""
    if not threshold_to_metrics:
        print(f"  - No data to plot for Sharpness vs. ECE (baseline: {baseline_name})")
        return

    colors = plt.get_cmap('viridis', len(BASELINE_NAMES))
    markers = ['o', 's', '^', 'D', 'v', '<', '>'] 
    te_ece = [d[f'{prefix}te_ece'] for d in threshold_to_metrics.values()]
    baseline_idx = BASELINE_NAMES.index(baseline_name)
    te_sharp = [d[f'{prefix}te_sharp_score'] for d in threshold_to_metrics.values()]

    ax.scatter(te_ece, te_sharp, label=baseline_name, alpha=0.7, s=50, color=colors(baseline_idx), marker=markers[baseline_idx % len(markers)])
    # The original implementation had threshold annotations, which are no longer applicable
    # with the new lumping method. The scatter plot remains the primary visualization.

def main(args):
    """Main function to find all experiment configs and generate plots."""
    # Determine prefix and suffixes based on the recalibrated flag
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
    if not exp_dir.is_dir():
        print(f"Error: Experiment directory not found at '{exp_dir}'")
        return

    print(f"--- Starting analysis for experiment: {exp_name} ---")
    if args.recalibrated:
        print("--- Plotting RECALIBRATED data ---")

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
            all_baselines_data[baseline] = aggregate_seeds_median(
                seed_pkls, quantile=args.quantile, prefix=key_prefix
            )

        if not all_baselines_data:
            print(f"  - No data found for any baseline in this directory. Skipping plots.")
            continue
            
        # 2. Create and save the ECE Exceedance plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        ax1.set_title(
            f"ECE Exceedance{title_suffix}\nDataset: {dataset_name} | Config: {hp_config_name} | Quantile: {args.quantile:.0%}\n"
            "(p = prop. below threshold; m = mean exceedance of exceeding points)"
        )
        ax1.set_xlabel('ECE Threshold')
        ax1.set_ylabel('max(Test ECE - Threshold, 0)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        output1 = hp_dir / f"ece_exceedance_plot{filename_suffix}_{args.quantile:.0%}.png"
        plt.savefig(output1, bbox_inches='tight')
        plt.close(fig1)

        # 3. Create and save the Sharpness vs. ECE plot
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for name, metrics in all_baselines_data.items():
            plot_sharpness_vs_ece(ax2, metrics, name, prefix=key_prefix)

        ax2.set_title(f"Sharpness vs. ECE{title_suffix}\nDataset: {dataset_name} | Config: {hp_config_name} | Quantile: {args.quantile:.0%}")
        ax2.set_xlabel('Test ECE')
        ax2.set_ylabel('Test Sharpness Score')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        output2 = hp_dir / f"sharpness_vs_ece_plot{filename_suffix}_{args.quantile:.0%}.png"
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
        required=True,
        help="The name of the experiment to process (e.g., 'run')."
    )
    parser.add_argument(
        "--quantile",
        "-q",
        type=float,
        default=0.5,
        help="The quantile taken over all seeds (0.5 for median)."
    )
    parser.add_argument(
        "--recalibrated",
        '-r',
        action="store_true",
        help="Plot using recalibrated data keys (e.g., 'recal_te_ece')."
    )
    args = parser.parse_args()

    main(args)