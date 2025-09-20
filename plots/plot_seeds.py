import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
from plots.plot_utils import load_pickle
from utils.metrics import compute_igd_staircase, compute_gd_staircase, compute_hv

RESULTS_BASE_DIR = os.path.join(os.environ['SCRATCH'], "results")
BASELINE_NAMES = ['batch_qr', 'batch_cal', 'batch_int', 'maqr', 'calipso']

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
