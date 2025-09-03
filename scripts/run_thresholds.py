import os
import sys
import time
import glob
import argparse
import subprocess
from typing import List
from collections import deque
from script_utils import dict_to_cli_args, pick_free_gpu, get_save_file_name

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

def run_main(inputs, gpu_id):
    cmd = ["python", "main.py"] + dict_to_cli_args(inputs)
    print(cmd)
    proc = subprocess.Popen(cmd)
    return proc

def generate_plots_for_pickle(pkl_path: str, out_parent_dir: str):
    from plots.plot_metrics import plot_training_stats, compare_ece_sharpness, compare_scoring_rules, calibration_plot, plot_ece_sharpness
    from plots.plot_utils import load_pickle

    base = os.path.basename(pkl_path)
    name = base[:-4] if base.lower().endswith(".pkl") else base
    outdir = os.path.join(out_parent_dir, name)
    os.makedirs(outdir, exist_ok=True)

    print(f"Generating plots for {pkl_path} -> {outdir}")
    data = load_pickle(pkl_path)
    plot_training_stats(data, outpath=os.path.join(outdir, "training_stats.png"))
    compare_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness_comparison.png"))
    calibration_plot(data, outpath=os.path.join(outdir, "calibration_plot.png"))
    plot_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness_curve.png"))

default_inputs = {
    "save_dir": "/home/scratch/yixiz/results",
    "min_thres": 0.01,
    "max_thres": 0.15,
    "num_thres": 100,
    "seed": 0,
    "wait": 100000,
}
# datasets = ["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]
# loss_fns = ["batch_qr", "batch_cal", "batch_int"]
# num_ens_list = [1, 2, 3]
boots = [0]
residuals = [0]
layer_norms = [0]
batch_norms = [0]
dropouts = [0.0]
datasets = ["kin8nm", "boston", "wine"]
# datasets = ["boston"]
loss_fns = ["batch_qr", "batch_int", "batch_cal"]
num_ens_list = [1]
# learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# batch_sizes = [16, 64, 256]
learning_rates = [1e-3]
batch_sizes = [64]

MAX_JOBS = 15

# Close all subprocesses if the script is interrupted

job_status = {}
import signal
def signal_handler(sig, frame):
    print("Interrupt received, closing all subprocesses...")
    for proc in job_status.values():
        proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    from itertools import product

    # Use a dict mapping expected pickle path -> subprocess.Popen
    job_pool = deque(product(num_ens_list, datasets, loss_fns, boots, residuals, layer_norms, batch_norms, dropouts, learning_rates, batch_sizes))
    while job_pool or job_status:
        if job_pool and len(job_status) < MAX_JOBS:
            num_ens, dataset, loss, boot, residual, ln, bn, dr, lr, bs = job_pool.popleft()
            if ln == 1 and bn == 1 or boot == 1 and num_ens == 1:
                continue
            free_gpu = pick_free_gpu(min_free_mb=1500)
            while free_gpu is None:
                # poll every 10 seconds for available GPU
                print("No GPU available with sufficient free memory. Waiting 10 seconds...")
                time.sleep(10)
                free_gpu = pick_free_gpu(min_free_mb=1500)
            try:
                inputs = {
                    "data": dataset,
                    "loss": loss,
                    "num_ens": num_ens,
                    "boot": boot,
                    "residual": residual,
                    "layer_norm": ln,
                    "batch_norm": bn,
                    "dropout": dr,
                    "lr": lr,
                    "bs": bs,
                    "gpu": free_gpu
                }
                inputs.update(default_inputs)
                pkl_path = get_save_file_name(inputs)
                # Launch process asynchronously and store in dict
                proc = run_main(inputs, free_gpu)
                job_status[pkl_path] = proc
                print(f"Started process for {inputs} at GPU {free_gpu}")
            except Exception as e:
                # If launching fails, do not add to pending and log error
                print(f"Failed to launch main.py for {inputs}: {e}")
                continue
        if job_status:
            for pkl_file, proc in list(job_status.items()):
                ret = proc.poll()
                if ret is None:
                    # still running
                    continue
                # process finished; check return code
                if ret == 0:
                    if os.path.exists(pkl_file):
                        print(f"Process succeeded, generating plots for {pkl_file}")
                        try:
                            generate_plots_for_pickle(pkl_file, default_inputs["save_dir"])
                        except Exception as e:
                            print(f"Plot generation failed for {pkl_file}: {e}")
                    else:
                        print(f"Process finished with exit 0 but pickle not found: {pkl_file}. Removing from pending.")
                    # In all success cases, remove from pending
                    job_status.pop(pkl_file, None)
                else:
                    # process failed; put it back
                    job_pool.append((num_ens, dataset, loss, boot, residual, ln, bn, dr, lr, bs))
        time.sleep(10)

if __name__ == "__main__":
    main()