#!/usr/bin/env python3
"""
Lumps and plots experiment results to analyze model performance dominance.

This script processes .pkl files containing model evaluation metrics, performs
moving median/decile smoothing (lumping), and generates plots to show which
model baseline dominates across different ECE (Expected Calibration Error) ranges.

It performs four distinct analyses:
1. Best hyperparameter model comparison: Finds the best hyperparameter set for each 
   baseline (lumped over seeds), then lumps these best models over all datasets.
2. Hyperparameter comparison: Lumps all results for each hyperparameter set 
   (over all seeds and datasets) to compare hyperparameter performance.
3. Per-dataset comparison: For each dataset, lumps all results for each baseline
   (over all seeds and hyperparameters) to see which baseline is best for that dataset.
4. Global comparison: Lumps all results for each baseline over everything (seeds,
   hyperparameters, datasets) for a final, high-level summary.
"""
import os
import glob
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d

import plots_utils

# --- Constants ---
RESULTS_PREFIX = "/home/scratch/yixiz/results" # As requested
BASELINES = ['batch_qr', 'batch_cal', 'batch_int']
SEEDS = list(range(5))

# Metrics to plot. The script will generate one plot per metric.
# The x-axis is always 'te_ece_controlled'.
Y_METRICS = [
    'te_sharp_score_controlled',
    'te_bag_nll_controlled',
    'te_crps_controlled',
    'te_mpiw_controlled',
    'te_interval_controlled',
    'te_check_controlled',
    'te_variance_controlled'
]
# For all metrics, lower is better.

# --- Core Operations ---

def load_and_extract_data(pkl_paths, x_key, y_key):
    """
    Loads data from a list of pkl files and extracts (x, y) pairs.
    
    Args:
        pkl_paths (list): List of paths to .pkl files.
        x_key (str): Dictionary key for the x-axis data (ECE).
        y_key (str): Dictionary key for the y-axis data (e.g., sharpness).

    Returns:
        list: A list of (x, y) tuples, sorted by x-value.
    """
    all_points = []
    for pkl_path in pkl_paths:
        data = plots_utils.load_pickle(pkl_path)
        if data and x_key in data and y_key in data:
            xs = data[x_key]
            ys = data[y_key]
            if len(xs) == len(ys):
                all_points.extend(zip(xs, ys))
    
    # Sort points by x-value (ECE) for rolling operations
    all_points.sort(key=lambda p: p[0])
    return all_points

def lump(all_points, window_frac=None):
    """
    Calculates moving median, upper decile, and lower decile on a set of points.

    Args:
        all_points (list): A list of (x, y) tuples, pre-sorted by x.
        window_frac (float): Fraction of total points to use for the moving window size.

    Returns:
        tuple: Three lists of (x, y) tuples for (median, lower_decile, upper_decile).
               Returns (None, None, None) if not enough data.
    """
    df = pd.DataFrame(all_points, columns=['x', 'y'])

    window_size = int(len(df) * window_frac)
    df['median'] = df['y'].rolling(window=window_size, min_periods=1, center=True).median()
    df['lower'] = df['y'].rolling(window=window_size, min_periods=1, center=True).quantile(0.)
    df['upper'] = df['y'].rolling(window=window_size, min_periods=1, center=True).quantile(0.9)

    median_line = list(zip(df['x'], df['median']))
    lower_line = list(zip(df['x'], df['lower']))
    upper_line = list(zip(df['x'], df['upper']))
    
    return median_line, lower_line, upper_line

def get_interpolated_data(lumped_data, common_x_axis):
    """
    Interpolates and extrapolates lumped data onto a common x-axis.

    Extrapolation rules:
    - Below data range: Assign a "terrible" value (1.0).
    - Above data range: Use the value from the highest-ECE point.

    Args:
        lumped_data (tuple): (median_line, lower_line, upper_line) from lump().
        common_x_axis (np.array): The common ECE axis for interpolation.

    Returns:
        dict: A dictionary with interpolated 'median', 'lower', 'upper' y-values.
    """
    median_line, lower_line, upper_line = lumped_data
    
    interpolated_results = {}
    for name, line_data in [('median', median_line), ('lower', lower_line), ('upper', upper_line)]:
        if not line_data:
            # If no data, fill with terrible values
            interpolated_results[name] = np.full_like(common_x_axis, 1.0)
            continue
            
        x, y = zip(*line_data)
        x, y = np.array(x), np.array(y)
        
        # Create interpolation function
        f = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
        
        # Apply on common axis
        interpolated_y = f(common_x_axis)
        
        # Apply custom extrapolation rules
        min_x, max_x = x.min(), x.max()
        last_y_val = y[np.argmax(x)]
        
        interpolated_y[common_x_axis < min_x] = 1.0  # Terrible value for lower ECE
        interpolated_y[common_x_axis > max_x] = last_y_val # Constant extrapolation for higher ECE
        
        interpolated_results[name] = interpolated_y
        
    return interpolated_results


def plot_dominance(lumped_results_dict, y_metric_name, output_path, title):
    """
    Generates and saves a plot showing dominance regimes for different methods.

    Args:
        lumped_results_dict (dict): Keys are method names, values are lumped data tuples.
        y_metric_name (str): Name of the y-metric for labeling.
        output_path (str): Path to save the PNG file.
        title (str): Title for the plot.
    """
    if not lumped_results_dict:
        print(f"Skipping plot for {title}, no data.")
        return

    # 1. Create a common x-axis
    all_x = []
    for data in lumped_results_dict.values():
        if data[0]: # median line
            all_x.extend([p[0] for p in data[0]])
    if not all_x:
        print(f"Skipping plot for {title}, no valid lumped data.")
        return
    
    min_ece, max_ece = min(all_x), max(all_x)
    common_x_axis = np.linspace(min_ece, max_ece, 300)

    # 2. Interpolate all data onto the common axis
    interpolated_data = {}
    for name, lumped_data in lumped_results_dict.items():
        interpolated_data[name] = get_interpolated_data(lumped_data, common_x_axis)
        
    # 3. Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.get_cmap('viridis', len(lumped_results_dict))
    
    median_lines = []
    for i, (name, data) in enumerate(interpolated_data.items()):
        ax.plot(common_x_axis, data['median'], label=f'{name} (Median)', color=colors(i), lw=2.5)
        ax.fill_between(common_x_axis, data['lower'], data['upper'], color=colors(i), alpha=0.2, label=f'{name} (10-90th percentile)')
        median_lines.append(data['median'])

    # 4. Report dominance
    median_lines = np.array(median_lines)
    best_method_indices = np.argmin(median_lines, axis=0)
    method_names = list(interpolated_data.keys())
    
    dominance = {name: np.mean(best_method_indices == i) * 100 for i, name in enumerate(method_names)}
    
    dominance_str = ", ".join([f"{name}: {perc:.1f}%" for name, perc in dominance.items()])
    
    ax.set_xlabel("Test ECE (Lower is better)")
    ax.set_ylabel(f"{y_metric_name} (Lower is better)")
    ax.set_title(f"{title}\nDominance: {dominance_str}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(min_ece, max_ece)
    
    # Improve y-axis limits, e.g., by clipping high outliers for better visualization
    all_y_medians = np.concatenate([data['median'] for data in interpolated_data.values()])
    y_upper_bound = np.percentile(all_y_medians[all_y_medians < 0.99], 99) # Clip 1.0s
    ax.set_ylim(bottom=0, top=min(y_upper_bound * 1.5, 1.0))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")

# --- Analysis Functions ---

def run_analysis_1(run_name_path, x_key, y_key):
    """
    1. First pick the best model (in terms of dominance) for each baseline by lumping 
       over all seeds of each hyperparameter combination for each dataset. 
    2. Then with the best models and their data, lump over dataset and plot.
    """
    print("\n--- Running Analysis 1: Best Hyperparameter Model Comparison ---")
    
    datasets = [d for d in os.listdir(run_name_path) if os.path.isdir(os.path.join(run_name_path, d))]
    
    best_hyperparam_pkls = defaultdict(list)

    for dataset in tqdm(datasets, desc="Analysis 1/2: Finding best hypers per dataset"):
        dataset_path = os.path.join(run_name_path, dataset)
        
        for baseline in BASELINES:
            hyperparam_dirs = glob.glob(os.path.join(dataset_path, f"*{baseline}*"))
            
            best_auc = float('inf')
            best_pkls_for_baseline = []
            
            if not hyperparam_dirs:
                continue

            for hyper_dir in hyperparam_dirs:
                pkl_paths = glob.glob(os.path.join(hyper_dir, "*.pkl"))
                
                # Lump over seeds
                points = load_and_extract_data(pkl_paths, x_key, y_key)
                median, _, _ = lump(points)
                
                if median:
                    # Use Area Under Curve of the median line as proxy for "best"
                    x, y = zip(*median)
                    auc = np.trapz(y, x)
                    if auc < best_auc:
                        best_auc = auc
                        best_pkls_for_baseline = pkl_paths
            
            if best_pkls_for_baseline:
                best_hyperparam_pkls[baseline].extend(best_pkls_for_baseline)

    # Now lump the "best" pkls over all datasets
    final_lumped_data = {}
    for baseline in BASELINES:
        if not best_hyperparam_pkls[baseline]:
            print(f"Warning (Analysis 1): No valid pkls found for baseline '{baseline}'")
            continue
        all_points = load_and_extract_data(best_hyperparam_pkls[baseline], x_key, y_key)
        final_lumped_data[baseline] = lump(all_points)

    output_path = os.path.join(run_name_path, "analysis_1_best_hyperparams", f"{y_key}_dominance.png")
    plot_dominance(final_lumped_data, y_key, output_path, "Best Hyperparameters Lumped Over Datasets")

def run_analysis_2(run_name_path, x_key, y_key):
    """
    Lump within all seeds and all dataset, for each hyperparameter combination.
    """
    print("\n--- Running Analysis 2: Hyperparameter Comparison ---")
    
    all_hyperparam_dirs = glob.glob(os.path.join(run_name_path, "*", "*"))
    # Extract unique hyperparameter configuration names (the final directory name)
    hyperparam_names = sorted(list(set([os.path.basename(p) for p in all_hyperparam_dirs])))

    lumped_by_hyperparam = {}
    for hyper_name in tqdm(hyperparam_names, desc="Analysis 2: Lumping by hyperparam"):
        # Find all pkls across all datasets that match this hyperparameter name
        pkl_paths = glob.glob(os.path.join(run_name_path, "*", hyper_name, "*.pkl"))
        
        points = load_and_extract_data(pkl_paths, x_key, y_key)
        lumped_by_hyperparam[hyper_name] = lump(points)
    
    output_path = os.path.join(run_name_path, "analysis_2_by_hyperparam", f"{y_key}_dominance.png")
    plot_dominance(lumped_by_hyperparam, y_key, output_path, "Hyperparameter Performance Lumped Over Seeds & Datasets")

def run_analysis_3(run_name_path, x_key, y_key):
    """
    Lump within all seeds and all hyperparameters of a baseline and plot, for each dataset.
    """
    print("\n--- Running Analysis 3: Per-Dataset Baseline Comparison ---")
    datasets = [d for d in os.listdir(run_name_path) if os.path.isdir(os.path.join(run_name_path, d))]

    for dataset in tqdm(datasets, desc="Analysis 3: Lumping by dataset"):
        dataset_path = os.path.join(run_name_path, dataset)
        lumped_by_baseline = {}
        
        for baseline in BASELINES:
            # Get all pkls for this baseline in this dataset, across all hypers and seeds
            pkl_paths = glob.glob(os.path.join(dataset_path, f"*{baseline}*", "*.pkl"))
            
            points = load_and_extract_data(pkl_paths, x_key, y_key)
            lumped_by_baseline[baseline] = lump(points)

        output_path = os.path.join(dataset_path, f"analysis_3_baseline_comparison_{y_key}.png")
        plot_dominance(lumped_by_baseline, y_key, output_path, f"Baseline Performance on Dataset: {dataset}")

def run_analysis_4(run_name_path, x_key, y_key):
    """
    Lump all seeds, hyperparameters and all datasets for each baseline.
    """
    print("\n--- Running Analysis 4: Global Baseline Comparison ---")
    
    global_lumped_data = {}
    for baseline in tqdm(BASELINES, desc="Analysis 4: Lumping globally"):
        # Get all pkls for this baseline across EVERYTHING
        pkl_paths = glob.glob(os.path.join(run_name_path, "*", f"*{baseline}*", "*.pkl"))
        
        points = load_and_extract_data(pkl_paths, x_key, y_key)
        global_lumped_data[baseline] = lump(points)

    output_path = os.path.join(run_name_path, "analysis_4_global", f"{y_key}_dominance.png")
    plot_dominance(global_lumped_data, y_key, output_path, "Global Baseline Performance")

def main():
    """Main function to parse arguments and run analyses."""
    parser = argparse.ArgumentParser(description="Lump and plot experiment results.")
    parser.add_argument("run_name", type=str, help="The name of the run directory under RESULTS_PREFIX.")
    parser.add_argument("--skip-setup", action="store_true", help="Skip dummy pkl generation.")
    
    args = parser.parse_args()

    run_name_path = os.path.join(RESULTS_PREFIX, args.run_name)
    
    if not os.path.isdir(run_name_path) and not args.skip_setup:
        print(f"Run directory not found. Creating a dummy structure at '{run_name_path}' for demonstration.")
        # Create dummy data for testing
        dummy_datasets = ['cifar10', 'imagenet']
        dummy_hyperparams = {
            'batch_qr': ['qr_lr1e-3', 'qr_lr1e-4'],
            'batch_cal': ['cal_temp2', 'cal_temp5'],
            'batch_int': ['int_alpha0.1', 'int_alpha0.05']
        }
        for dset in dummy_datasets:
            for baseline, hypers in dummy_hyperparams.items():
                for hyper in hypers:
                    for seed in SEEDS:
                        pkl_dir = os.path.join(run_name_path, dset, f"{hyper}_{baseline}")
                        os.makedirs(pkl_dir, exist_ok=True)
                        pkl_path = os.path.join(pkl_dir, f"model_{seed}.pkl")
                        dummy_data = plots_utils.create_dummy_pkl_data(seed, baseline)
                        with open(pkl_path, 'wb') as f:
                            import pickle
                            pickle.dump(dummy_data, f)
        print("Dummy data generation complete.")
    elif not os.path.isdir(run_name_path):
         print(f"Error: Run directory '{run_name_path}' not found. Aborting.")
         return

    for is_recal in [False, True]:
        prefix = "recal_" if is_recal else ""
        x_key = f"{prefix}te_ece_controlled"
        
        print(f"\n{'='*20} PROCESSING {'RECALIBRATED' if is_recal else 'NON-RECALIBRATED'} METRICS {'='*20}")

        for y_metric in Y_METRICS:
            y_key = f"{prefix}{y_metric}"
            print(f"\n--- Plotting for metric: {y_key} ---")
            
            # Run all four analyses for the current metric
            run_analysis_1(run_name_path, x_key, y_key)
            run_analysis_2(run_name_path, x_key, y_key)
            run_analysis_3(run_name_path, x_key, y_key)
            run_analysis_4(run_name_path, x_key, y_key)

if __name__ == "__main__":
    main()