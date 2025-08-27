import os
import sys
import time
import glob
import argparse
import subprocess
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Run main.py across datasets and thresholds and generate plots.")
    parser.add_argument("--num_thresholds", "-n", type=int, default=10,
                        help="Number of thresholds to run (uniformly between 0 (excluded) and 0.01). Default 10.")
    parser.add_argument("--min_threshold", "-min", type=float, default=0.0,
                        help="Minimum threshold value (default: 0.0).")
    parser.add_argument("--max_threshold", "-max", type=float, default=0.2,
                        help="Maximum threshold value (default: 0.2).")
    parser.add_argument("--datasets", "-d", type=str, default="",
                        help="Comma-separated list of datasets to run. If omitted, runs all UCI datasets.")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory passed to main.py via --save_dir and where pickles are saved. Default 'results'.")
    parser.add_argument("--python_cmd", type=str, default=sys.executable,
                        help="Python executable to call main.py with (default: current interpreter).")
    parser.add_argument("--main_path", type=str, default=os.path.join(REPO_ROOT, "main.py"),
                        help="Path to main.py (default: ./main.py in repo root).")
    args, unknown = parser.parse_known_args()
    return args, unknown

def find_new_pickles(before_set, results_dir):
    all_now = set(glob.glob(os.path.join(results_dir, "*.pkl")))
    return list(all_now - before_set)

def run_main_for(dataset: str, min_thres: float, max_thres: float, args, extra_args: List[str]):
    cmd = [
        args.python_cmd,
        args.main_path,
        "--min_thres", f"{min_thres:.8f}",
        "--max_thres", f"{max_thres:.8f}",
        "--data", dataset,
        "--save_dir", args.save_dir
    ]
    # append any extra/unknown args provided to this script so they are forwarded to main.py
    if extra_args:
        cmd += extra_args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def generate_plots_for_pickle(pkl_path: str, out_parent_dir: str):
    # import plotting helpers (do local import so script can run even if plotting deps missing until needed)
    from plots.plot_metrics import plot_training_stats, compare_ece_sharpness, compare_scoring_rules, calibration_plot
    from plots.plot_utils import load_pickle

    base = os.path.basename(pkl_path)
    name = base[:-4] if base.lower().endswith(".pkl") else base
    outdir = os.path.join(out_parent_dir, name)
    os.makedirs(outdir, exist_ok=True)

    print(f"Generating plots for {pkl_path} -> {outdir}")
    data = load_pickle(pkl_path)
    plot_training_stats(data, outpath=os.path.join(outdir, "training_stats.png"))
    compare_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness.png"))
    compare_scoring_rules(data, outpath=os.path.join(outdir, "scoring_rules.png"))
    calibration_plot(data, outpath=os.path.join(outdir, "calibration_plot.png"))

def main():
    args, extra_args = parse_args()

    # default datasets = files under data/UCI_Datasets without .txt suffix if user didn't provide explicit list
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        # datasets = ["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]
        datasets = ["boston", "concrete", "energy", "wine", "yacht"]

    results_dir = args.save_dir
    os.makedirs(results_dir, exist_ok=True)

    for dataset in datasets:
        # snapshot existing pickles
        before = set(glob.glob(os.path.join(results_dir, "*.pkl")))
        try:
            run_main_for(dataset, args.min_threshold, args.max_threshold, args, extra_args)
        except subprocess.CalledProcessError as e:
            print(f"main.py failed for dataset={dataset}, min_thres={args.min_threshold}, max_thres={args.max_threshold}: {e}")
            continue

        # wait briefly to allow file system to settle
        time.sleep(5.0)
        new_pkls = find_new_pickles(before, results_dir)
        if not new_pkls:
            print(f"No new pickle detected for dataset={dataset}, min_thres={args.min_threshold}, max_thres={args.max_threshold}. Check main.py output.")
            continue

        for pkl in sorted(new_pkls):
            try:
                generate_plots_for_pickle(pkl, results_dir)
            except Exception as e:
                print(f"Failed to generate plots for {pkl}: {e}")

    print("All runs and plotting complete.")

if __name__ == "__main__":
    main()