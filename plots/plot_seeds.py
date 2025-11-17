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

def generate_report(report_filepath, dataset_name, hp_config_name, hp_dir, method_ece_y_points, title_suffix, filename_suffix):
    with open(report_filepath, 'w') as f:
        f.write(f"Performance Report for {dataset_name} / {hp_config_name}{title_suffix}\n")
        f.write("=" * 80 + "\n\n")

        # --- Multi-objective metrics (IGD, GD, HV) - Mean and Variance over Seeds ---
        f.write("--- Multi-Objective Metrics (Mean and Variance over Seeds) ---\n")
        for y_metric_name in Y_METRICS:
            f.write(f"\nMetrics for ECE vs {TITLE_METRICS[y_metric_name]}:\n")

            # Structure to hold scores for each seed: metric -> method -> [score_seed0, score_seed1, ...]
            all_seed_scores = defaultdict(lambda: defaultdict(list))

            # Loop over each seed to compute MOO metrics independently
            for seed in SEEDS:
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
                    gd_scores = compute_gd(current_method_points_for_moo)
                except Exception as e:
                    print(f"  Error computing GD for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try:
                    gdp_scores = compute_gd_plus(current_method_points_for_moo)
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
                        scores[f'{metric}_mean'] = np.nanmean(score_list)
                        scores[f'{metric}_std'] = np.std(score_list) / np.sqrt(len(score_list))
                    # log the scores
                    if y_metric_name in ['te_sharp_score_controlled', 'te_variance_controlled', 'te_mpiw_controlled']:
                        # save_dir = f'metric_log/{y_metric_name.replace("te_", "").replace("_controlled", "")}/'
                        save_dir = os.path.join('metric_log', args.exp_name, y_metric_name.replace("te_", "").replace("_controlled", ""))
                        os.makedirs(save_dir, exist_ok=True)
                        with open(os.path.join(save_dir, f"{method}_{metric}_{hp_config_name}{filename_suffix}.txt"), 'a') as log_f:
                            log_f.write(f"{dataset_name} {scores[f'{metric}_mean']} {len(score_list)} "
                                        + ' '.join([f" {s}" for s in score_list]) + '\n')

                    method_scores[metric][method] = scores[f'{metric}_mean']

                all_results[method] = scores

            f.write("\\midrule\n")
            for i, metric in enumerate(PERFORMANCE_METRICS):
                if metric == 'HV':
                    arrow = '\\uparrow'
                else:
                    arrow = '\\downarrow'
                if i == 0:
                    row = f"\\multirow{{{len(PERFORMANCE_METRICS)}}}{{4em}}{{{dataset_name.capitalize()}}} & {metric} $({arrow})$"
                else:
                    row = f" & {metric} $({arrow})$"

                # collect means/stds for all methods for this dataset+metric
                means = {m: all_results[m][f"{metric}_mean"] for m in BASELINE_NAMES}
                stds  = {m: all_results[m][f"{metric}_std"]  for m in BASELINE_NAMES}

                # best mean + threshold rule
                best_method = min(means, key=means.get) if metric != "HV" else max(means, key=means.get)
                best_mean, best_std = means[best_method], stds[best_method]
                threshold = best_mean + best_std if metric != "HV" else best_mean - best_std

                for m in BASELINE_NAMES:
                    mean, std = means[m], stds[m]
                    if metric != 'HV':
                        scaled_mean = mean * 100
                        scaled_std = std * 100
                    else:
                        scaled_mean = mean
                        scaled_std = std
                    cell = f"{scaled_mean:.2f} $\\pm$ {scaled_std:.2f}"
                    if metric != "HV":  # lower is better
                        if mean - std <= threshold or m == best_method:
                            cell = f"\\textbf{{{cell}}}"
                    else:  # higher is better
                        if mean + std >= threshold or m == best_method:
                            cell = f"\\textbf{{{cell}}}"
                    row += " & " + cell

                row += " \\\\\n"
                f.write(row)

                
        print(f"Report saved to {report_filepath}")



def process_hp_dir(hp_dir: Path, key_prefix: str, title_suffix: str, filename_suffix: str, dataset_name: str, hp_config_name: str) -> dict:
    print(f"\nProcessing: {dataset_name} / {hp_config_name}")

    # Stores (ece, y_metric_value) pairs for plotting and multi-objective metrics
    # Structure: seed -> y_metric_name -> method_name -> [(ece, y_value), ...]
    method_ece_y_points = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    method_ece_y_point_best_model = defaultdict(lambda: defaultdict(lambda: defaultdict(tuple)))

    all_y_metrics_to_process = Y_METRICS

    # --- Step 1 & 2: Iterate and Load Data ---
    # Find and load pkl files for each method and seed
    for method in BASELINE_NAMES:
        for seed in SEEDS:
            hp_filename = HP_DIR_NAME_TO_FILE_NAME[hp_config_name]
            old_pkl_filenames = glob.glob(os.path.join(hp_dir, f"*loss{method}*{seed}*.pkl"))
            pkl_filenames = [pkl_filename for pkl_filename in old_pkl_filenames if hp_filename in os.path.basename(pkl_filename) and dataset_name in os.path.basename(pkl_filename)]
            if len(old_pkl_filenames) != len(pkl_filenames):
                print(f"  Note: Filtered out {set(old_pkl_filenames) - set(pkl_filenames)} not matching hp config or dataset name {hp_filename}/{dataset_name} in {hp_dir}.")
            pkl_filenames = [pkl_filename for pkl_filename in pkl_filenames if 'models' not in os.path.basename(pkl_filename)]

            if not pkl_filenames:
                print(f"  - {os.path.join(hp_dir, f'*loss{method}*{seed}*.pkl')} not found. Skipping {method} {seed}.")
                continue
            pkl_path = pkl_filenames[0]
            try:
                data = load_pickle(pkl_path)
                print(f"  Loaded {pkl_path}")
            except Exception as e:
                print(f"  Error loading or processing {pkl_path}: {e}")
                continue # Skip to the next file if there's an error

            te_ece = data[f'{key_prefix}te_ece_controlled']
            va_ece = data[f'va_ece_controlled']
            try:
                te_ece_best_model = data[f'{key_prefix}te_ece']
                va_ece_best_model = data[f'va_ece']
            except KeyError as e:
                te_ece_best_model = np.nan
                va_ece_best_model = np.nan

            # Store exceedance metric under the specific seed
            y_exceedance = [max(te - va, 0) for te, va in zip(te_ece, va_ece) if va < 0.15]
            method_ece_y_points[seed]['te_va_ece_exceedance'][method] = [(te, ye) for te, ye, va in zip(te_ece, y_exceedance, va_ece)]
            method_ece_y_point_best_model[seed]['te_va_ece_exceedance'][method] = (te_ece_best_model, max(te_ece_best_model - va_ece_best_model, 0))


            # Store other Y_METRICS under the specific seed
            for y_metric in Y_METRICS:
                if y_metric == 'te_va_ece_exceedance':
                    continue # Already handled above
                y_val = data[f'{key_prefix}{y_metric}']
                method_ece_y_points[seed][y_metric][method] = [(te, yv) for te, yv, va in zip(te_ece, y_val, va_ece)]
                try:
                    y_val_best_model = data[f'{key_prefix}{y_metric.replace("_controlled", "")}']
                    method_ece_y_point_best_model[seed][y_metric][method] = (te_ece_best_model, y_val_best_model)
                except KeyError:
                    method_ece_y_point_best_model[seed][y_metric][method] = (te_ece_best_model, np.nan)


    # --- Step 3: Generate Report Text File ---
    report_filepath = hp_dir / f"metrics_{dataset_name}_{HP_DIR_NAME_TO_FILE_NAME[hp_config_name]}{filename_suffix}.txt"
    generate_report(report_filepath, dataset_name, hp_config_name, hp_dir, method_ece_y_points, title_suffix, filename_suffix)
    # --- Step 4: Generate Scatter Plots ---
    for y_metric_name in all_y_metrics_to_process:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        all_x_coords = []
        all_y_coords = []
        
        has_data_for_plot = False
        baseline_order_flipped = BASELINE_NAMES[::-1]
        for method in baseline_order_flipped:
            # Collect points from all seeds for the current method
            all_points_for_method = []
            all_points_for_method_best_model = []
            for seed in SEEDS:
                if seed in method_ece_y_points and y_metric_name in method_ece_y_points[seed]:
                    all_points_for_method.extend(method_ece_y_points[seed][y_metric_name][method])
                    all_points_for_method_best_model.append(method_ece_y_point_best_model[seed][y_metric_name][method])

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
                except TypeError as e:
                    print(f"  Bag NLL is currently skipped")
                len_new = len(x_coords)
                if len_new < len_old:
                    print(f"  Note: Filtered out {len_old - len_new} points with {y_metric_name} > 10 for {hp_dir}, method {method}.")
                
                ax.scatter(
                        x_coords, y_coords,
                        label=TITLE_METHODS[method],
                        s=50,
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                    )
                # Plot best model point
                try:
                    best_model_x, best_model_y = zip(*all_points_for_method_best_model)
                except ValueError:
                    best_model_x, best_model_y = [], []
                ax.scatter(
                        best_model_x, best_model_y,
                        s=150,
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                        edgecolors='black',
                )
                
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)

        # Set labels and axis limits
        if has_data_for_plot:
            ax.set_title(dataset_name.capitalize(), fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel(f'ECE', fontsize=20)
            ax.set_ylabel(f'{TITLE_METRICS[y_metric_name]}', fontsize=20)

            # set y_max as 95th percentile of all_y_coords + 0.02. 
            if all_x_coords and all_y_coords:
                min_x, max_95_x = np.min(all_x_coords), np.percentile(all_x_coords, 95)
                try:
                    min_10_y, max_90_y = np.percentile(all_y_coords, 10), np.percentile(all_y_coords, 90)
                except TypeError as e:
                    print(f"  Bag NLL is currently skipped")
                    continue
                
                x_buffer = (max_95_x - min_x) * 0.1 if (max_95_x - min_x) > 0 else 0.1
                y_buffer = (max_90_y - min_10_y) * 1 if (max_90_y - min_10_y) > 0 else 0.1

                if dataset_name not in ['concrete', 'kin8nm', 'power', 'boston', 'wine', 'elevator', 'protein'] \
                and filename_suffix == '':
                    min_y, max_y = max(0, min_10_y - y_buffer), min(0.05, max_90_y + y_buffer)
                else:
                    min_y, max_y = min_10_y - y_buffer, max_90_y + y_buffer


                ax.set_xlim(min_x - x_buffer, max_95_x + x_buffer)
                ax.set_ylim(min_y, max_y)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plot_filename = hp_dir / f"plot_{dataset_name}_{hp_config_name}_{y_metric_name}{filename_suffix}.pdf"
            plt.legend(fontsize=12)
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