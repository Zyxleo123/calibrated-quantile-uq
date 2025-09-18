#!/usr/bin/env python3
"""
Lumps and plots experiment results to analyze model performance.

This script processes .pkl files containing model evaluation metrics. It uses the
"average distance to the global Pareto front" as its primary comparison metric.
A lower average distance indicates better, more consistent performance.

The script performs three distinct analyses:
1. Best hyperparameter model comparison: For each dataset, finds the best 
   hyperparameter configuration for each baseline based on the lowest average
   distance score. It then aggregates these "best" models for a final comparison.
2. Per-dataset comparison: For each dataset, it compares baselines by calculating
   the average distance score using all seeds and hyperparameters for that dataset.
3. Global comparison: Aggregates all results over all datasets, seeds, and
   hyperparameters for a final, high-level summary.
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

import plot_utils

RESULTS_PREFIX = "/home/scratch/yixiz/results"
BASELINES = ['batch_qr', 'batch_cal', 'batch_int']
SEEDS = list(range(5))
DATASETS = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht']

Y_METRICS = [
    'te_sharp_score_controlled',
    'te_bag_nll_controlled',
    'te_crps_controlled',
    'te_mpiw_controlled',
    'te_interval_controlled',
    'te_check_controlled',
    'te_variance_controlled'
]

def get_valid_dataset_dirs(run_name_path):
    """
    Finds all subdirectories in the run path and returns only those
    whose names are in the official DATASETS list.
    """
    all_subdirs = glob.glob(os.path.join(run_name_path, '*/'))
    valid_dataset_dirs = [
        d for d in all_subdirs 
        if os.path.isdir(d) and os.path.basename(os.path.normpath(d)) in DATASETS
    ]
    return sorted(valid_dataset_dirs)

def load_and_extract_data(pkl_paths, x_key, y_key):
    """
    Loads data from pkl files and extracts (x, y) pairs, assuming files are valid.
    """
    all_points = []
    for pkl_path in pkl_paths:
        data = plot_utils.load_pickle(pkl_path)
        xs = data[x_key]
        ys = data[y_key]
        all_points.extend(zip(xs, ys))
    
    all_points.sort(key=lambda p: p[0])
    return all_points

def lump(all_points, window_frac=0.05):
    """
    Calculates moving median and quantiles on a set of points for visualization.
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

    df = df.dropna(subset=['median', 'upper', 'lower'])
    median_line = list(zip(df['x'], df['median']))
    upper_line = list(zip(df['x'], df['upper']))
    lower_line = list(zip(df['x'], df['lower']))

    return median_line, upper_line, lower_line

def find_pareto_front(points):
    """
    Finds the Pareto front from a set of 2D points.
    Assumes lower values are better for both dimensions.
    """
    if not points:
        return []
    
    pareto_front = []
    points = sorted(points, key=lambda p: p[0])
    
    for p in points:
        is_dominated = False
        for other_p in pareto_front:
            if other_p[0] <= p[0] and other_p[1] <= p[1]:
                is_dominated = True
                break
        if not is_dominated:
            pareto_front = [other_p for other_p in pareto_front if not (p[0] <= other_p[0] and p[1] <= other_p[1])]
            pareto_front.append(p)
            
    return pareto_front

def calculate_avg_distance_to_front(all_pkl_paths_grouped_by_method, x_key, y_key):
    """
    Calculates the average distance to the global Pareto front for each method.
    """
    method_names = sorted(list(all_pkl_paths_grouped_by_method.keys()))
    
    all_points_with_method = []
    for method in method_names:
        points = load_and_extract_data(all_pkl_paths_grouped_by_method[method], x_key, y_key)
        for p in points:
            all_points_with_method.append({'point': p, 'method': method})

    if not all_points_with_method:
        return {name: float('inf') for name in method_names}
        
    all_points = [d['point'] for d in all_points_with_method]
    x_coords, y_coords = zip(*all_points)

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0

    for d in all_points_with_method:
        x, y = d['point']
        norm_x = (x - min_x) / range_x
        norm_y = (y - min_y) / range_y
        d['norm_point'] = (norm_x, norm_y)

    all_norm_points = [d['norm_point'] for d in all_points_with_method]
    global_pareto_front = find_pareto_front(all_norm_points)

    if not global_pareto_front:
        return {name: 0.0 for name in method_names}
    
    global_pareto_front_np = np.array(global_pareto_front)

    for d in all_points_with_method:
        point_np = np.array(d['norm_point'])
        distances = np.linalg.norm(global_pareto_front_np - point_np, axis=1)
        d['distance_to_front'] = np.min(distances)

    avg_distances = defaultdict(list)
    for d in all_points_with_method:
        avg_distances[d['method']].append(d['distance_to_front'])
        
    final_scores = {
        method: np.mean(distances) if distances else float('inf')
        for method, distances in avg_distances.items()
    }
    
    return final_scores

# --- Plotting ---
def plot_results(lumped_results_for_plot, y_metric_name, output_path, title, avg_distance_scores):
    """
    Generates and saves a scatter plot of the lumped median data points.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.get_cmap('viridis', len(lumped_results_for_plot))
    markers = ['o', 's', '^', 'D', 'v', '<', '>'] 
    
    method_names = sorted(list(avg_distance_scores.keys()))
    scores_str = ", ".join([f"{name}: {avg_distance_scores.get(name, float('inf')):.4f}" for name in method_names])

    all_x, all_y = [], []
    for i, name in enumerate(method_names):
        median_line, upper_line, lower_line = lumped_results_for_plot[name]
        x_vals, y_vals = zip(*median_line)
        # shaded area between upper and lower quantiles and median line. all in scatter plot
        if upper_line and lower_line:
            upper_y = [y for x, y in upper_line]
            lower_y = [y for x, y in lower_line]
            ax.fill_between(x_vals, lower_y, upper_y, color=colors(i), alpha=0.2)
        ax.scatter(x_vals, y_vals, label=f"{name}", color=colors(i), marker=markers[i % len(markers)], s=50)
        all_x.extend(x_vals)
        all_y.extend(y_vals)
    
    if not all_x:
        print(f"Skipping plot for {title}, no data to plot.")
        plt.close(fig); return

    ax.set_xlabel("Test ECE")
    ax.set_ylabel(f"{y_metric_name}")
    ax.set_title(f"{title}\nAvg Distance to Front (Lower is Better): {scores_str}", fontsize=14)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1.5)
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")

# --- Analysis Functions ---

def run_analysis_1(run_name_path, x_key, y_key):
    """Analysis 1: Best hyperparameter (per-dataset) comparison."""
    print("\n--- Running Analysis 1: Best Hyperparameter (Per-Dataset) Comparison ---")
    
    # Use the helper to get a clean list of dataset directories that actually exist
    valid_dataset_dirs = get_valid_dataset_dirs(run_name_path)
    best_hyperparam_pkls_by_baseline = defaultdict(list)

    for dataset_path in tqdm(valid_dataset_dirs, desc="Analysis 1: Finding best hyperparameters per dataset"):
        hyperparam_dirs = [p for p in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(p)]

        for baseline in BASELINES:
            hyperparam_pkls_for_scoring = {os.path.basename(hp_dir): glob.glob(os.path.join(hp_dir, f"*{baseline}*.pkl")) for hp_dir in hyperparam_dirs}
            hyperparam_pkls_for_scoring = {k: v for k, v in hyperparam_pkls_for_scoring.items() if v}
            
            distance_scores = calculate_avg_distance_to_front(hyperparam_pkls_for_scoring, x_key, y_key)
            
            best_hyper_dir_name = min(distance_scores, key=distance_scores.get)
            best_pkls = hyperparam_pkls_for_scoring[best_hyper_dir_name]
            best_hyperparam_pkls_by_baseline[baseline].extend(best_pkls)

    lumped_data_for_plot = {b: lump(load_and_extract_data(p, x_key, y_key), window_frac=window_frac) for b, p in best_hyperparam_pkls_by_baseline.items()}
    avg_distance_scores = calculate_avg_distance_to_front(best_hyperparam_pkls_by_baseline, x_key, y_key)
    output_dir = os.path.join(run_name_path, f"best_hyperparams_all_datasets_w{window_frac:.2f}")
    output_path = os.path.join(output_dir, f"{y_key}_result.png")
    plot_results(lumped_data_for_plot, y_key, output_path, f"Best Hyperparameters - {y_key}", avg_distance_scores)

def run_analysis_3(run_name_path, x_key, y_key):
    """Analysis 3: Per-dataset baseline comparison."""
    print("\n--- Running Analysis 3: Per-Dataset Baseline Comparison ---")
    valid_dataset_dirs = get_valid_dataset_dirs(run_name_path)
    for dataset_path in tqdm(valid_dataset_dirs, desc="Analysis 3: Lumping by dataset"):
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        all_pkls_by_baseline = {b: glob.glob(os.path.join(dataset_path, "*", f"*{b}*.pkl")) for b in BASELINES}
        lumped_data = {b: lump(load_and_extract_data(p, x_key, y_key), window_frac=window_frac) for b, p in all_pkls_by_baseline.items()}
        avg_scores = calculate_avg_distance_to_front(all_pkls_by_baseline, x_key, y_key)
        output_dir = os.path.join(run_name_path, f"all_hyperparams_per_dataset_w{window_frac:.2f}", dataset_name)
        output_path = os.path.join(output_dir, f"{y_key}_result.png")
        plot_results(lumped_data, y_key, output_path, f"All Hyperparameters on {dataset_name} - {y_key}", avg_scores)

def run_analysis_4(run_name_path, x_key, y_key):
    """Analysis 4: Global baseline comparison."""
    print("\n--- Running Analysis 4: Global Baseline Comparison ---")
    valid_dataset_dirs = get_valid_dataset_dirs(run_name_path)
    all_pkls_by_baseline = defaultdict(list)
    
    for baseline in tqdm(BASELINES, desc="Analysis 4: Lumping globally"):
        for dataset_path in valid_dataset_dirs:
            pattern = os.path.join(dataset_path, "*", f"*{baseline}*.pkl")
            all_pkls_by_baseline[baseline].extend(glob.glob(pattern))

    lumped_data = {b: lump(load_and_extract_data(p, x_key, y_key), window_frac=window_frac) for b, p in all_pkls_by_baseline.items()}
    avg_scores = calculate_avg_distance_to_front(all_pkls_by_baseline, x_key, y_key)
    output_dir = os.path.join(run_name_path, f"global_all_datasets_w{window_frac:.2f}")
    output_path = os.path.join(output_dir, f"{y_key}_results.png")
    plot_results(lumped_data, y_key, output_path, f"Global Performance - {y_key}", avg_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lump and plot experiment results using average distance to Pareto front.")
    parser.add_argument("-n", "--run_name", type=str, required=True, help="The name of the run directory under RESULTS_PREFIX.")
    parser.add_argument("--window_frac", '-w', type=float, default=0.05, help="Fraction of data for moving window in lumping.")
    args = parser.parse_args()

    window_frac = args.window_frac
    run_name_path = os.path.join(RESULTS_PREFIX, args.run_name)
    if not os.path.isdir(run_name_path):
        print(f"Error: Run directory not found at {run_name_path}")
    
    for is_recal in [True, False]:
        prefix = "recal_" if is_recal else ""
        x_key = f"{prefix}te_ece_controlled"
        
        print(f"\n{'='*20} PROCESSING {'RECALIBRATED' if is_recal else 'NON-RECALIBRATED'} METRICS {'='*20}")

        for y_metric in Y_METRICS:
            y_key = f"{prefix}{y_metric}"
            print(f"\n--- Analyzing metric: {y_key} ---")
            
            run_analysis_1(run_name_path, x_key, y_key)
            run_analysis_3(run_name_path, x_key, y_key)
            run_analysis_4(run_name_path, x_key, y_key)