import os
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

import time
import argparse
import subprocess
from collections import deque
from script_utils import (dict_to_cli_args, 
                          pick_free_gpu_round_robin, 
                          get_one_hot_param,
                          generate_plots_for_pickle, 
                          generate_overlap_plot, 
                          fix_inputs, 
                          invalid_inputs)
from utils.misc_utils import get_save_file_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-f", "--filter_type", type=str, default="one-hot", choices=["one-hot"])
    parser.add_argument("-n", "--name", type=str)
    return parser.parse_args()

def run_main(inputs):
    cmd = ["python", "main.py"] + dict_to_cli_args(inputs)
    print(cmd)
    proc = subprocess.Popen(cmd)
    return proc

RESULT_BASE = os.path.join(os.environ["SCRATCH"], "results")

BASIC_INPUTS = {
    "save_dir": RESULT_BASE,
    "min_thres": 0.01,
    "max_thres": 0.15,
    "num_thres": 100,
    "wait": 100000,
}

DEFAULT_VALUE = {
    "num_ens": 1,
    "nl": 2,
    "hs": 64,
    "residual": 0,
    "batch_norm": 0,
    "layer_norm": 0,
    "dropout": 0.0,
    "activation": "relu",
}

FULL_HYPERPARAMS = {
    "skip_existing": [1],
    "data": ["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0, 1],
    "layer_norm": [0, 1],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "num_ens": [1, 2, 5],
    "boot": [0, 1],
    "nl": [1, 2, 3, 4, 5, 6, 7],
    "hs": [16, 32, 64, 128, 256],
    "residual": [0, 1],
    "seed": [0, 1, 2, 3, 4],
    "loss": ["batch_qr", "batch_int", "batch_cal"],
}

TEST_HYPERPARAMS = {
    "skip_existing": [0],
    "data": ["boston"],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [2],
    "hs": [64],
    "residual": [0],
    "seed": [0, 1, 2],
    "loss": ["batch_qr", "batch_int", "batch_cal"],
}

MAX_JOBS = 15

# Close all subprocesses if the script is interrupted
job_status = {}
import signal
def signal_handler(sig, frame):
    print("Interrupt received, closing all subprocesses...")
    for proc, inputs in job_status.values():
        proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    from itertools import product
    script_args = parse_args()
    if script_args.test:
        job_pool = deque([dict(zip(TEST_HYPERPARAMS, v)) for v in product(*TEST_HYPERPARAMS.values())])
    else:
        job_pool = deque([dict(zip(FULL_HYPERPARAMS, v)) for v in product(*FULL_HYPERPARAMS.values())])
    while job_pool or job_status:
        # First, clean up finished jobs
        if job_status:
            for pkl_file, (proc, inputs) in list(job_status.items()):
                ret = proc.poll()
                if ret is None:
                    # still running
                    continue
                # process finished
                if ret == 0:
                    if os.path.exists(pkl_file):
                        print(f"Process succeeded, generating plots for {pkl_file}")
                        generate_plots_for_pickle(pkl_file, job_dir)
                    else:
                        print(f"Process finished with exit 0 but pickle not found: {pkl_file}. Removing from pending.")
                    print(f"Generating overlap plot for {pkl_file} if all baselines are ready.")
                    ret = generate_overlap_plot(pkl_file, inputs["loss"], FULL_HYPERPARAMS["loss"], job_dir)
                    if not ret:
                        print(f"Overlap plot generation skipped for {pkl_file} due to missing baseline files.")
                    else:
                        print(f"Overlap plot generated for {pkl_file}.")
                    job_status.pop(pkl_file, None)
                else:
                    # process failed; put it back
                    print(f"Process for {pkl_file} failed with exit code {ret}. Re-queuing.")
                    job_status.pop(pkl_file, None)
                    job_pool.append(inputs)
        # Only launch new jobs if we have capacity
        while job_pool and len(job_status) < MAX_JOBS:
            # Check/fix inputs
            inputs = fix_inputs(job_pool.popleft())
            if invalid_inputs(inputs):
                continue

            # Filter job & get job name
            if script_args.filter_type == "one-hot" and script_args.test is False:
                one_hot_key = get_one_hot_param(inputs, DEFAULT_VALUE)
                if one_hot_key is None:
                    continue
                job_name = f"{one_hot_key}-{inputs[one_hot_key]}"
            if script_args.test:
                inputs["num_ep"] = 300
                job_name = "test"
            job_dir = os.path.join(RESULT_BASE, inputs["data"], job_name)
            BASIC_INPUTS["save_dir"] = job_dir
            os.makedirs(job_dir, exist_ok=True)
            free_gpu = pick_free_gpu_round_robin(min_free_mb=1500)
            while free_gpu is None:
                print("No GPU available with sufficient free memory. Waiting 10 seconds...")
                time.sleep(10)
                free_gpu = pick_free_gpu_round_robin(min_free_mb=1500)

            try:
                inputs["gpu"] = free_gpu
                inputs.update(BASIC_INPUTS)
                pkl_path = get_save_file_name(inputs)
                # Launch process asynchronously and store in dict
                proc = run_main(inputs)
                job_status[pkl_path] = (proc, inputs)
                print(f"Started process for {inputs} at GPU {free_gpu}")
            except Exception as e:
                # If launching fails, do not add to pending and log error
                print(f"Failed to launch main.py for {inputs}: {e}")
                continue
        time.sleep(10)

if __name__ == "__main__":
    main()