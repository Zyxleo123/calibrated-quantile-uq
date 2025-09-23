import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For distinct colors
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plots.plot_utils import (
    load_pickle,
    RESULTS_BASE_DIR,
    BASELINE_NAMES,
    SEEDS,
    Y_METRICS,
    TITLE_METRICS,
    TITLE_METHODS,
    HP_DIR_NAME_TO_FILE_NAME,
    METHOD_COLORS,
    METHOD_MARKERS,
    PERFORMANCE_METRICS
)
from plots.plot_utils import safe_get
from utils.metrics import compute_igd, compute_gd, compute_hv, compute_igd_plus, compute_gd_plus

def process_hp_dir(hp_dir: Path, key_prefix: str, title_suffix: str, filename_suffix: str, dataset_name: str, hp_config_name: str) -> dict:
    """
    Processes a single hyperparameter directory:
    - Extracts data from pkl files for each method and seed.
    - Computes multi-objective metrics (IGD, GD, HV) for each seed and reports mean/variance.
    - Writes a detailed report to a text file.
    - Generates scatter plots for ECE vs each Y_METRIC, aggregating points from all seeds.

    Args:
        hp_dir (Path): The path to the current hyperparameter directory.
        key_prefix (str): Prefix for data keys (e.g., 'recal_').
        title_suffix (str): Suffix for plot titles (e.g., ' (Recalibrated)').
        filename_suffix (str): Suffix for output filenames.
        dataset_name (str): Name of the dataset.
        hp_config_name (str): Name of the hyperparameter configuration.

    Returns:
        Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]: A dictionary with data
        structured as {seed: {y_metric: {method: [(ece, y_value), ...]}}}.
    """
    print(f"\nProcessing: {dataset_name} / {hp_config_name}")

    # Stores (ece, y_metric_value) pairs for plotting and multi-objective metrics
    # Structure: seed -> y_metric_name -> method_name -> [(ece, y_value), ...]
    method_ece_y_points = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    all_y_metrics_to_process = Y_METRICS

    # --- Step 1 & 2: Iterate and Load Data ---
    # Find and load pkl files for each method and seed
    for method in BASELINE_NAMES:
        for seed in SEEDS:
            hp_filename = HP_DIR_NAME_TO_FILE_NAME[hp_config_name]
            old_pkl_filenames = glob.glob(os.path.join(hp_dir, f"*loss{method}*{seed}*.pkl"))
            pkl_filenames = [pkl_filename for pkl_filename in old_pkl_filenames if hp_filename in os.path.basename(pkl_filename)]
            if len(old_pkl_filenames) != len(pkl_filenames):
                print(f"  Note: Filtered out {set(old_pkl_filenames) - set(pkl_filenames)} not matching hp config {hp_filename} in {hp_dir}.")

            if not pkl_filenames:
                print(f"  - {os.path.join(hp_dir, f'*loss{method}*{seed}*.pkl')} not found. Skipping {method} {seed}.")
                continue
            pkl_path = pkl_filenames[0] # Take the first match
            try:
                data = load_pickle(pkl_path)
                print(f"  Loaded {pkl_path}")
            except Exception as e:
                print(f"  Error loading or processing {pkl_path}: {e}")
                continue # Skip to the next file if there's an error

            te_ece = data[f'{key_prefix}te_ece_controlled']
            va_ece = data[f'va_ece_controlled'] # Needed for exceedance calculation

            # Store exceedance metric under the specific seed
            y_exceedance = [max(te - va, 0) for te, va in zip(te_ece, va_ece) if va < 1]
            method_ece_y_points[seed]['te_va_ece_exceedance'][method] = [(te, ye) for te, ye, va in zip(te_ece, y_exceedance, va_ece) if va < 1]

            # Store other Y_METRICS under the specific seed
            for y_metric in Y_METRICS:
                y_val = data[f'{key_prefix}{y_metric}']
                method_ece_y_points[seed][y_metric][method] = [(te, yv) for te, yv, va in zip(te_ece, y_val, va_ece) if va < 1]
            

    # --- Step 3: Generate Report Text File ---
    report_filepath = hp_dir / f"metrics_{dataset_name}_{hp_config_name}{filename_suffix}.txt"
    with open(report_filepath, 'w') as f:
        f.write(f"Performance Report for {dataset_name} / {hp_config_name}{title_suffix}\n")
        f.write("=" * 80 + "\n\n")

        # --- Multi-objective metrics (IGD, GD, HV) - Mean and Variance over Seeds ---
        f.write("--- Multi-Objective Metrics (Mean and Variance over Seeds) ---\n")
        for y_metric_name in all_y_metrics_to_process:
            f.write(f"\nMetrics for ECE vs {y_metric_name}:\n")

            # Structure to hold scores for each seed: metric -> method -> [score_seed0, score_seed1, ...]
            all_seed_scores = defaultdict(lambda: defaultdict(list))

            # Loop over each seed to compute MOO metrics independently
            for seed in SEEDS:
                if seed not in method_ece_y_points or y_metric_name not in method_ece_y_points[seed]:
                    continue
                
                # Get points for the current seed: {method: [(x,y), ...]}
                current_method_points_for_moo = method_ece_y_points[seed][y_metric_name]

                # Compute MOO metrics for the current seed
                igd_scores, igdp_scores, gd_scores, gdp_scores, hv_scores = {}, {}, {}, {}, {}
                try:
                    igd_scores = compute_igd(current_method_points_for_moo)
                except Exception as e:
                    print(f"  Error computing IGD for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try:
                    igdp_scores = compute_igd_plus(current_method_points_for_moo)
                except Exception as e:
                    print(f"  Error computing IGD+ for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try:
                    gd_scores = compute_gd(current_method_points_for_moo, ece_step_size=0.001)
                except Exception as e:
                    print(f"  Error computing GD for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try:
                    gdp_scores = compute_gd_plus(current_method_points_for_moo, ece_step_size=0.001)
                except Exception as e:
                    print(f"  Error computing GD+ for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try:
                    hv_scores = compute_hv(current_method_points_for_moo)
                except Exception as e:
                    print(f"  Error computing HV for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                    
                # Store scores for this seed
                for method in BASELINE_NAMES:
                    if method in igd_scores: all_seed_scores['IGD'][method].append(igd_scores[method])
                    if method in igdp_scores: all_seed_scores['IGD+'][method].append(igdp_scores[method])
                    if method in gd_scores: all_seed_scores['GD'][method].append(gd_scores[method])
                    if method in gdp_scores: all_seed_scores['GD+'][method].append(gdp_scores[method])
                    if method in hv_scores: all_seed_scores['HV'][method].append(hv_scores[method])

            method_scores = {m: {} for m in PERFORMANCE_METRICS}
            all_results = {}

            for method in BASELINE_NAMES:
                scores = {}
                for metric in PERFORMANCE_METRICS:
                    score_list = all_seed_scores[metric].get(method, [])
                    if not score_list:
                        scores[f'{metric}_mean'], scores[f'{metric}_std'] = float('nan'), float('nan')
                    elif len(score_list) == 1:
                        scores[f'{metric}_mean'], scores[f'{metric}_std'] = score_list[0], 0.0
                    else:
                        scores[f'{metric}_mean'] = np.mean(score_list)
                        scores[f'{metric}_std'] = np.std(score_list) / np.sqrt(len(score_list))

                    method_scores[metric][method] = scores[f'{metric}_mean']

                all_results[method] = scores

            f.write("\\midrule\n")
            for i, metric in enumerate(PERFORMANCE_METRICS):
                if i == 0:
                    row = f"\\multirow{{{len(PERFORMANCE_METRICS)}}}{{4em}}{{{dataset_name}}} & {metric}"
                else:
                    row = f" & {metric}"

                # collect means/stds for all methods for this dataset+metric
                means = {m: all_results[m][f"{metric}_mean"] for m in BASELINE_NAMES}
                stds  = {m: all_results[m][f"{metric}_std"]  for m in BASELINE_NAMES}

                # best mean + threshold rule
                best_method = min(means, key=means.get) if metric != "HV" else max(means, key=means.get)
                best_mean, best_std = means[best_method], stds[best_method]
                threshold = best_mean - best_std if metric != "HV" else best_mean + best_std

                for m in BASELINE_NAMES:
                    mean, std = means[m], stds[m]
                    cell = f"{mean:.3f} $\\pm$ {std:.3f}"
                    if metric != "HV":  # lower is better
                        if mean + std >= threshold:
                            cell = f"\\textbf{{{cell}}}"
                    else:  # higher is better
                        if mean - std <= threshold:
                            cell = f"\\textbf{{{cell}}}"
                    row += " & " + cell

                row += " && \\\\\n"
                f.write(row)

                
        print(f"Report saved to {report_filepath}")


    # --- Step 4: Generate Scatter Plots ---
    for y_metric_name in all_y_metrics_to_process:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        all_x_coords = []
        all_y_coords = []
        
        has_data_for_plot = False
        for method in BASELINE_NAMES:
            # Collect points from all seeds for the current method
            all_points_for_method = []
            for seed in SEEDS:
                if seed in method_ece_y_points and y_metric_name in method_ece_y_points[seed]:
                    points = method_ece_y_points[seed][y_metric_name].get(method, [])
                    if points:
                        all_points_for_method.extend(points)

            if all_points_for_method:
                has_data_for_plot = True
                # Unzip list of tuples into x and y coordinates
                x_coords, y_coords = zip(*all_points_for_method)
                len_old = len(x_coords)
                # filter out y > 10
                try:
                    y_coords, x_coords = zip(*[(y, x) for x, y in zip(x_coords, y_coords) if y <= 10])
                except ValueError:
                    print(f"  All points for {hp_dir}, method {method} have {y_metric_name} > 10. Skipping these points.")
                    y_coords, x_coords = [], []
                len_new = len(x_coords)
                if len_new < len_old:
                    print(f"  Note: Filtered out {len_old - len_new} points with {y_metric_name} > 10 for {hp_dir}, method {method}.")
                
                ax.scatter(
                        x_coords, y_coords,
                        label=TITLE_METHODS[method],
                        s=50,
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                        alpha=0.3
                    )
                
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)

        if has_data_for_plot:
            ax.set_title(f'ECE vs {TITLE_METRICS[y_metric_name]}{" (Recalibrated)" if title_suffix else ""} for {dataset_name} / {hp_config_name}')
            ax.set_xlabel(f'ECE')
            ax.set_ylabel(f'{TITLE_METRICS[y_metric_name]}')

            # set y_max as 95th percentile of all_y_coords + 0.02. 
            if all_x_coords and all_y_coords:
                min_x, max_x = np.min(all_x_coords), np.max(all_x_coords)
                min_y, max_95_y = np.min(all_y_coords), np.percentile(all_y_coords, 95)
                
                x_buffer = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 0.1
                
                ax.set_xlim(min_x - x_buffer, max_x + x_buffer)
                ax.set_ylim(max(0, min_y - 0.02), max_95_y + 0.02)

            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title="Method")
            plt.tight_layout()
            plot_filename = hp_dir / f"plot_{dataset_name}_{hp_config_name}_{y_metric_name}{filename_suffix}.pdf"
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        else:
            print(f"  No data to plot for ECE vs {y_metric_name}. Skipping plot generation.")
        
        plt.close(fig)

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

    hyperparam_dirs = [d for d in exp_dir.glob('*/*') if d.is_dir()]

    for hp_dir in sorted(hyperparam_dirs):
        dataset_name = hp_dir.parent.name
        hp_config_name = hp_dir.name
        process_hp_dir(hp_dir, key_prefix, title_suffix, filename_suffix, dataset_name, hp_config_name)
    
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
        "--recalibrated",
        '-r',
        action="store_true",
        help="Plot using recalibrated data keys (e.g., 'recal_te_ece')."
    )
    args = parser.parse_args()
    main(args)