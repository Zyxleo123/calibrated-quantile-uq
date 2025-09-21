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

from plots.plot_utils import load_pickle
from utils.metrics import compute_igd, compute_gd, compute_hv

RESULTS_BASE_DIR = os.path.join(os.environ.get('SCRATCH', '.'), "results")
BASELINE_NAMES = ['batch_qr', 'batch_cal', 'batch_int', 'maqr', 'calipso']
SEEDS = ['_0', '_1', '_2', '_3', '_4']
Y_METRICS = [
    'te_sharp_score_controlled',
    'te_variance_controlled',
    'te_mpiw_controlled',
    'te_crps_controlled',
    'te_interval_controlled',
    'te_check_controlled',
    'te_bag_nll_controlled',
]
TITLE_METRICS = {
    'te_sharp_score_controlled': 'Sharpness',
    'te_variance_controlled': 'Variance',
    'te_mpiw_controlled': 'MPIW',
    'te_crps_controlled': 'CRPS',
    'te_interval_controlled': 'Interval Score',
    'te_check_controlled': 'Check Score',
    'te_bag_nll_controlled': 'Bag NLL',
    'te_va_ece_exceedance': 'ECE Exceedance'
}
colors = cm.get_cmap('tab10', len(BASELINE_NAMES))
method_colors = {method: colors(i) for i, method in enumerate(BASELINE_NAMES)}
method_markers = {
    'batch_qr': 'o',  # Circle
    'batch_cal': 's', # Square
    'batch_int': '^', # Triangle up
    'maqr': 'D',      # Diamond
    'calipso': 'X',   # X
}


def process_hp_dir(hp_dir: Path, key_prefix: str, title_suffix: str, filename_suffix: str, dataset_name: str, hp_config_name: str):
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
            pkl_filenames = glob.glob(os.path.join(hp_dir, f"*loss{method}*{seed}*.pkl"))
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
            y_exceedance = [max(te - va, 0) for te, va in zip(te_ece, va_ece)]
            method_ece_y_points[seed]['te_va_ece_exceedance'][method] = [(te, ye) for te, ye in zip(te_ece, y_exceedance)]

            # Store other Y_METRICS under the specific seed
            for y_metric in Y_METRICS:
                y_val = data[f'{key_prefix}{y_metric}']
                method_ece_y_points[seed][y_metric][method] = [(te, yv) for te, yv in zip(te_ece, y_val)]
            

    # --- Step 3: Generate Report Text File ---
    report_filepath = hp_dir / f"performance_report{filename_suffix}.txt"
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

                if not any(current_method_points_for_moo.values()):
                    continue

                try:
                    igd_scores = compute_igd(current_method_points_for_moo)
                    gd_scores = compute_gd(current_method_points_for_moo)
                    hv_scores = compute_hv(current_method_points_for_moo)
                    
                    # Store scores for this seed
                    for method in BASELINE_NAMES:
                        if method in igd_scores: all_seed_scores['IGD'][method].append(igd_scores[method])
                        if method in gd_scores: all_seed_scores['GD'][method].append(gd_scores[method])
                        if method in hv_scores: all_seed_scores['HV'][method].append(hv_scores[method])

                except Exception as e:
                    print(f"  Error computing MOO metrics for {y_metric_name} on seed {seed}: {e}")
            
            # Now compute and report mean/variance
            header = (f"  {'Method':<15} {'IGD Mean':<12} {'IGD Var':<12} {'GD Mean':<12} "
                      f"{'GD Var':<12} {'HV Mean':<12} {'HV Var':<12}")
            f.write(header + "\n")
            f.write(f"  {'-'*15:<15} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12} "
                    f"{'-'*12:<12} {'-'*12:<12} {'-'*12:<12}\n")

            for method in BASELINE_NAMES:
                scores = {}
                for metric in ['IGD', 'GD', 'HV']:
                    score_list = all_seed_scores[metric].get(method, [])
                    if not score_list:
                        scores[f'{metric}_mean'], scores[f'{metric}_var'] = float('nan'), float('nan')
                    elif len(score_list) == 1:
                        scores[f'{metric}_mean'], scores[f'{metric}_var'] = score_list[0], 0.0
                    else:
                        scores[f'{metric}_mean'] = np.mean(score_list)
                        scores[f'{metric}_var'] = np.var(score_list)
                
                f.write(f"  {method:<15} {scores['IGD_mean']:<12.6f} {scores['IGD_var']:<12.6f} "
                        f"{scores['GD_mean']:<12.6f} {scores['GD_var']:<12.6f} "
                        f"{scores['HV_mean']:<12.6f} {scores['HV_var']:<12.6f}\n")
    
        f.write("\n" + "=" * 80 + "\n")
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
                
                ax.scatter(x_coords, y_coords,
                        label=method,
                        s=50,
                        color=method_colors.get(method, 'gray'),
                        marker=method_markers.get(method, 'o'),
                        alpha=0.7)
                
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)

        if has_data_for_plot:
            ax.set_title(f'ECE vs {TITLE_METRICS[y_metric_name]}{" (Recalibrated)" if title_suffix else ""} for {dataset_name} / {hp_config_name}')
            ax.set_xlabel(f'ECE')
            ax.set_ylabel(f'{TITLE_METRICS[y_metric_name]}')

            if all_x_coords and all_y_coords:
                min_x, max_x = np.min(all_x_coords), np.max(all_x_coords)
                min_y, max_y = np.min(all_y_coords), np.max(all_y_coords)
                
                x_buffer = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 0.1
                y_buffer = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 0.1
                
                ax.set_xlim(min_x - x_buffer, max_x + x_buffer)
                ax.set_ylim(min_y - y_buffer, max_y + y_buffer)

            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title="Method")
            plt.tight_layout()
            plot_filename = hp_dir / f"seeds_scatter_ece_vs_{y_metric_name}{filename_suffix}.png"
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        else:
            print(f"  No data to plot for ECE vs {y_metric_name}. Skipping plot generation.")
        
        plt.close(fig)

    # Convert defaultdicts to regular dicts for the final return structure.
    return {
        seed: {
            ym: dict(y_metric_data)
            for ym, y_metric_data in seed_data.items()
        }
        for seed, seed_data in method_ece_y_points.items()
    }


def main(args):
    """Main function to find all experiment configs and generate plots."""
    if 'SCRATCH' not in os.environ:
        print("Warning: SCRATCH environment variable not set. Using current directory for RESULTS_BASE_DIR.")
    
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

    all_processed_data_by_hp_dir = {} # Stores results for all hp_dirs
    
    for hp_dir in sorted(hyperparam_dirs):
        dataset_name = hp_dir.parent.name
        hp_config_name = hp_dir.name
        
        processed_data_for_hp_dir = process_hp_dir(hp_dir, key_prefix, title_suffix, filename_suffix, dataset_name, hp_config_name)
        all_processed_data_by_hp_dir[f"{dataset_name}/{hp_config_name}"] = processed_data_for_hp_dir
    
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