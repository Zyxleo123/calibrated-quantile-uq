import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For distinct colors
import glob
import math

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

# Add 'te_va_ece_exceedance' to the list of metrics to process and give it a title
Y_METRICS_AND_EXCEEDANCE = Y_METRICS + ['te_va_ece_exceedance']
TITLE_METRICS['te_va_ece_exceedance'] = 'ECE Exceedance'


def process_hp_dir(hp_dir: Path, key_prefix: str, title_suffix: str, filename_suffix: str, dataset_name: str, hp_config_name: str) -> dict:
    """
    Processes a single hyperparameter directory:
    - Extracts data from pkl files for each method and seed.
    - Computes multi-objective metrics (IGD, GD, HV) for each seed and reports mean/variance.
    - Writes a detailed report to a text file.
    - Aggregates points from all seeds for plotting and returns them.

    Args:
        hp_dir (Path): The path to the current hyperparameter directory.
        key_prefix (str): Prefix for data keys (e.g., 'recal_').
        title_suffix (str): Suffix for plot titles (e.g., ' (Recalibrated)').
        filename_suffix (str): Suffix for output filenames.
        dataset_name (str): Name of the dataset.
        hp_config_name (str): Name of the hyperparameter configuration.

    Returns:
        Dict[str, Dict[str, List[Tuple[float, float]]]]: A dictionary with aggregated plot data
        structured as {y_metric: {method: [(ece, y_value), ...]}}.
    """
    print(f"\nProcessing: {dataset_name} / {hp_config_name}")

    # Stores (ece, y_metric_value) pairs for plotting and multi-objective metrics
    # Structure: seed -> y_metric_name -> method_name -> [(ece, y_value), ...]
    method_ece_y_points = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # --- Step 1 & 2: Iterate and Load Data ---
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
            pkl_path = pkl_filenames[0]
            try:
                data = load_pickle(pkl_path)
                print(f"  Loaded {pkl_path}")
            except Exception as e:
                print(f"  Error loading or processing {pkl_path}: {e}")
                continue

            te_ece = data[f'{key_prefix}te_ece_controlled']
            va_ece = data[f'va_ece_controlled']

            y_exceedance = [max(te - va, 0) for te, va in zip(te_ece, va_ece) if va < 1]
            method_ece_y_points[seed]['te_va_ece_exceedance'][method] = [(te, ye) for te, ye, va in zip(te_ece, y_exceedance, va_ece) if va < 1]

            for y_metric in Y_METRICS:
                y_val = data[f'{key_prefix}{y_metric}']
                method_ece_y_points[seed][y_metric][method] = [(te, yv) for te, yv, va in zip(te_ece, y_val, va_ece) if va < 1]
            
    # --- Step 3: Generate Report Text File ---
    report_filepath = hp_dir / f"metrics_{dataset_name}_{hp_config_name}{filename_suffix}.txt"
    with open(report_filepath, 'w') as f:
        f.write(f"Performance Report for {dataset_name} / {hp_config_name}{title_suffix}\n")
        f.write("=" * 80 + "\n\n")
        f.write("--- Multi-Objective Metrics (Mean and Variance over Seeds) ---\n")
        for y_metric_name in Y_METRICS_AND_EXCEEDANCE:
            f.write(f"\nMetrics for ECE vs {TITLE_METRICS.get(y_metric_name, y_metric_name)}:\n")
            all_seed_scores = defaultdict(lambda: defaultdict(list))
            for seed in SEEDS:
                if seed not in method_ece_y_points or y_metric_name not in method_ece_y_points[seed]:
                    continue
                current_method_points_for_moo = method_ece_y_points[seed][y_metric_name]
                igd_scores, igdp_scores, gd_scores, gdp_scores, hv_scores = {}, {}, {}, {}, {}
                try: igd_scores = compute_igd(current_method_points_for_moo)
                except Exception as e: print(f"  Error computing IGD for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try: igdp_scores = compute_igd_plus(current_method_points_for_moo)
                except Exception as e: print(f"  Error computing IGD+ for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try: gd_scores = compute_gd(current_method_points_for_moo, ece_step_size=0.001)
                except Exception as e: print(f"  Error computing GD for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try: gdp_scores = compute_gd_plus(current_method_points_for_moo, ece_step_size=0.001)
                except Exception as e: print(f"  Error computing GD+ for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                try: hv_scores = compute_hv(current_method_points_for_moo)
                except Exception as e: print(f"  Error computing HV for {hp_dir}, seed {seed}, y_metric {y_metric_name}: {e}")
                for method in BASELINE_NAMES:
                    if method in igd_scores: all_seed_scores['IGD'][method].append(igd_scores[method])
                    if method in igdp_scores: all_seed_scores['IGD+'][method].append(igdp_scores[method])
                    if method in gd_scores: all_seed_scores['GD'][method].append(gd_scores[method])
                    if method in gdp_scores: all_seed_scores['GD+'][method].append(gdp_scores[method])
                    if method in hv_scores: all_seed_scores['HV'][method].append(hv_scores[method])

            all_results = {}
            for method in BASELINE_NAMES:
                scores = {}
                for metric in PERFORMANCE_METRICS:
                    score_list = all_seed_scores[metric].get(method, [])
                    if not score_list: scores[f'{metric}_mean'], scores[f'{metric}_std'] = float('nan'), float('nan')
                    elif len(score_list) == 1: scores[f'{metric}_mean'], scores[f'{metric}_std'] = score_list[0], 0.0
                    else:
                        scores[f'{metric}_mean'] = np.mean(score_list)
                        scores[f'{metric}_std'] = np.std(score_list) / np.sqrt(len(score_list))
                all_results[method] = scores

            f.write("\\midrule\n")
            for i, metric in enumerate(PERFORMANCE_METRICS):
                row = f" & {metric}" if i > 0 else f"\\multirow{{{len(PERFORMANCE_METRICS)}}}{{4em}}{{{dataset_name}}} & {metric}"
                means = {m: all_results[m][f"{metric}_mean"] for m in BASELINE_NAMES}
                stds = {m: all_results[m][f"{metric}_std"] for m in BASELINE_NAMES}
                best_method = min(means, key=means.get) if metric != "HV" else max(means, key=means.get)
                best_mean, best_std = means[best_method], stds[best_method]
                threshold = best_mean - best_std if metric != "HV" else best_mean + best_std
                for m in BASELINE_NAMES:
                    mean, std = means[m], stds[m]
                    cell = f"{mean:.3f} $\\pm$ {std:.3f}"
                    is_best = (mean + std >= threshold) if metric != "HV" else (mean - std <= threshold)
                    if is_best: cell = f"\\textbf{{{cell}}}"
                    row += " & " + cell
                row += " && \\\\\n"
                f.write(row)
        print(f"Report saved to {report_filepath}")

    # --- Step 4: Aggregate points across all seeds for plotting and return ---
    aggregated_plot_data = defaultdict(lambda: defaultdict(list))
    for y_metric_name in Y_METRICS_AND_EXCEEDANCE:
        for method in BASELINE_NAMES:
            all_points_for_method = []
            for seed in SEEDS:
                points = method_ece_y_points[seed][y_metric_name][method]
                if points:
                    all_points_for_method.extend(points)

            if all_points_for_method:
                x_coords, y_coords = zip(*all_points_for_method)
                len_old = len(x_coords)
                filtered_points = [(x, y) for x, y in zip(x_coords, y_coords) if y <= 10]
                len_new = len(filtered_points)
                if len_new < len_old:
                    print(f"  Note: Filtered out {len_old - len_new} points with {y_metric_name} > 10 for plotting.")
                if filtered_points:
                    aggregated_plot_data[y_metric_name][method] = filtered_points

    return aggregated_plot_data

def main(args):
    """Main function to find all experiment configs and generate plots."""
    if args.recalibrated:
        key_prefix, title_suffix, filename_suffix = 'recal_', ' (Recalibrated)', '_recalibrated'
    else:
        key_prefix, title_suffix, filename_suffix = '', '', ''

    exp_name = args.exp_name
    exp_dir = Path(RESULTS_BASE_DIR) / exp_name
    print(f"--- Starting analysis for experiment: {exp_name} ---")
    if args.recalibrated: print("--- Plotting RECALIBRATED data ---")

    hyperparam_dirs = [d for d in exp_dir.glob('*/*') if d.is_dir()]

    # Group hyperparameter directories by their config name
    hp_configs = defaultdict(list)
    for hp_dir in hyperparam_dirs:
        hp_configs[hp_dir.name].append(hp_dir)

    for hp_config_name, hp_dirs_for_config in sorted(hp_configs.items()):
        print(f"\n--- Processing HP Config: {hp_config_name} ---")

        # Step 1: Collect data from all datasets for this hp_config
        all_datasets_plot_data = {}
        dataset_names = []
        for hp_dir in sorted(hp_dirs_for_config):
            dataset_name = hp_dir.parent.name
            dataset_names.append(dataset_name)
            plot_data = process_hp_dir(hp_dir, key_prefix, title_suffix, filename_suffix, dataset_name, hp_config_name)
            all_datasets_plot_data[dataset_name] = plot_data

        # Step 2: Generate one big plot for each y_metric
        for y_metric_name in Y_METRICS_AND_EXCEEDANCE:
            is_data_available = any(y_metric_name in d for d in all_datasets_plot_data.values())
            if not is_data_available:
                print(f"  No data for '{y_metric_name}' in hp_config '{hp_config_name}'. Skipping plot.")
                continue

            n_datasets = len(dataset_names)
            if n_datasets == 0: continue
            
            ncols = 2 if n_datasets > 1 else 1
            nrows = math.ceil(n_datasets / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
            axes = axes.flatten()

            for i, dataset_name in enumerate(dataset_names):
                ax = axes[i]
                data_for_subplot = all_datasets_plot_data[dataset_name][y_metric_name]

                if not data_for_subplot:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
                    ax.set_title(dataset_name, fontsize=16)
                    continue

                all_x_coords, all_y_coords = [], []
                for method in BASELINE_NAMES:
                    points = data_for_subplot.get(method, [])
                    if points:
                        x_coords, y_coords = zip(*points)
                        ax.scatter(x_coords, y_coords, label=TITLE_METHODS[method], s=50, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method])
                        all_x_coords.extend(x_coords)
                        all_y_coords.extend(y_coords)

                ax.set_title(dataset_name, fontsize=18)
                ax.set_xlabel('ECE', fontsize=16)
                ax.set_ylabel(TITLE_METRICS[y_metric_name], fontsize=16)

                if all_x_coords and all_y_coords:
                    min_x, max_x = np.min(all_x_coords), np.max(all_x_coords)
                    min_y, max_95_y = np.min(all_y_coords), np.percentile(all_y_coords, 95)
                    x_buffer = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 0.1
                    ax.set_xlim(min_x - x_buffer, min(max_x + x_buffer, 0.2) + 0.02)
                    ax.set_ylim(max(0, min_y - 0.02), max_95_y + 0.02)
                ax.grid(True, linestyle='--', alpha=0.6)

            for j in range(n_datasets, len(axes)):
                axes[j].axis('off')

            handles, labels = None, None
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            
            if handles:
                n_cols_legend = min(len(BASELINE_NAMES), 4)
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=n_cols_legend, fontsize=14)

            title_text = f'ECE vs {TITLE_METRICS[y_metric_name]}{title_suffix} for HP Config: {hp_config_name}'
            fig.suptitle(title_text, fontsize=22, y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for legend and suptitle
            
            plot_filename = exp_dir / f"plot_all-datasets_{hp_config_name}_{y_metric_name}{filename_suffix}.pdf"
            plt.savefig(plot_filename, bbox_inches='tight')
            print(f"\nSaved combined plot to {plot_filename}")
            plt.close(fig)

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