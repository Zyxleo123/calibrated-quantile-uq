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
                          invalid_inputs,
                          RESULT_BASE,
                          BASIC_INPUTS,
                          DEFAULT_VALUE,
                          HYPERPARAMS)
from utils.misc_utils import get_save_file_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_type", type=str, default="full", choices=["test", "full", "newdata"])
    parser.add_argument("-f", "--filter_type", type=str, default="one-hot", choices=["one-hot"])
    parser.add_argument("-n", "--name", type=str)
    return parser.parse_args()

def run_main(inputs):
    cmd = ["python", "main.py"] + dict_to_cli_args(inputs)
    proc = subprocess.Popen(cmd)
    return proc


MAX_JOBS = 30

# Close all subprocesses if the script is interrupted
job_status = {}
import signal
def signal_handler(sig, frame):
    print("Interrupt received, closing all subprocesses...")
    for proc, inputs, start_time in job_status.values():
        proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    from itertools import product
    script_args = parse_args()
    hyperparam_set = HYPERPARAMS[script_args.run_type.upper()]
    job_pool = deque([dict(zip(hyperparam_set, v)) for v in product(*hyperparam_set.values())])
    while job_pool or job_status:
        # First, clean up finished jobs
        if job_status:
            for pkl_file, (proc, inputs, start_time) in list(job_status.items()):
                ret = proc.poll()
                if ret is None:
                    # still running
                    continue
                # process finished
                if ret == 0:
                    print(f"Process succeeded. Time: {time.time() - start_time:.1f}s. Inputs: {inputs}")
                    job_status.pop(pkl_file, None)
                else:
                    # process failed; put it back
                    print(f"Process exited. Ret: {ret}. Time: {time.time() - start_time:.1f}s. Inputs: {inputs}")
                    job_status.pop(pkl_file, None)
                    job_pool.append(inputs)
        # Only launch new jobs if we have capacity
        while job_pool and len(job_status) < MAX_JOBS:
            # Check/fix inputs
            inputs = job_pool.popleft()
            inputs.update(BASIC_INPUTS)
            inputs = fix_inputs(inputs)
            if invalid_inputs(inputs):
                continue

            # Filter job & get job name
            if script_args.filter_type == "one-hot" and script_args.test is False:
                job_name = get_one_hot_param(inputs, DEFAULT_VALUE)
                if job_name is None:
                    continue
            if script_args.test:
                job_name = "test"
            job_dir = os.path.join(RESULT_BASE, script_args.name, inputs["data"], job_name)
            BASIC_INPUTS["save_dir"] = job_dir
            os.makedirs(job_dir, exist_ok=True)
            free_gpu = pick_free_gpu_round_robin(min_free_mb=1500)
            while free_gpu is None:
                print("No GPU available with sufficient free memory. Waiting 10 seconds...")
                time.sleep(10)
                free_gpu = pick_free_gpu_round_robin(min_free_mb=1500)

            try:
                inputs["gpu"] = free_gpu
                pkl_path = get_save_file_name(inputs)
                # Launch process asynchronously and store in dict
                proc = run_main(inputs)
                job_status[pkl_path] = (proc, inputs, time.time())
                print(f"GPU {free_gpu}: {inputs}")
            except Exception as e:
                # If launching fails, do not add to pending and log error
                print(f"Failed launching {inputs}: {e}")
                continue
            if not script_args.test:
                time.sleep(1)

if __name__ == "__main__":
    main()