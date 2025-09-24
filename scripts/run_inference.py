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
                          find_available_gpus,
                          get_one_hot_param,
                          fix_inputs, 
                          invalid_inputs,
                          RESULT_BASE,
                          DEFAULT_VALUE,
                          HYPERPARAMS)
from utils.misc_utils import get_save_file_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_type", type=str, default="full")
    parser.add_argument("-f", "--filter_type", type=str, default="one-hot", choices=["one-hot"])
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--max_jobs_per_gpu", "-m", type=int, default=1, 
                        help="Maximum concurrent jobs allowed per GPU (replaces global max jobs).")
    parser.add_argument("--max_jobs", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.gpus = None if args.gpus is None else [int(x) for x in args.gpus.split(",")]
    global MAX_JOBS_PER_GPU
    MAX_JOBS_PER_GPU = args.max_jobs_per_gpu
    return args

def run_main(inputs):
    cmd = ["python", "main.py"] + dict_to_cli_args(inputs)
    proc = subprocess.Popen(cmd)
    return proc


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

    # Track running jobs per GPU
    gpu_job_counts = {}

    while job_pool or job_status:
        # First, clean up finished jobs and free GPU slots immediately
        if job_status:
            for pkl_file, (proc, inputs, start_time) in list(job_status.items()):
                ret = proc.poll()
                if ret is None:
                    # still running
                    continue
                # process finished
                gpu_idx = inputs.get("gpu")
                if gpu_idx is not None:
                    gpu_job_counts[gpu_idx] = max(0, gpu_job_counts.get(gpu_idx, 0) - 1)
                if ret == 0:
                    print(f"Process succeeded. Time: {time.time() - start_time:.1f}s. Inputs: {inputs}")
                    job_status.pop(pkl_file, None)
                else:
                    # process failed; put it back
                    print(f"Process exited. Ret: {ret}. Time: {time.time() - start_time:.1f}s. Inputs: {inputs}")
                    job_status.pop(pkl_file, None)
                    job_pool.append(inputs)

        # Launch new jobs while there is work AND at least one GPU has capacity
        while job_pool:
            # find available GPUs with enough memory
            available = find_available_gpus(min_free_mb=1000, choices=script_args.gpus)
            if not available:
                print("No GPU available with sufficient free memory. Waiting 10 seconds...")
                time.sleep(10)
                continue

            # choose first available GPU that is under the per-GPU limit
            available_sorted = sorted(available)
            free_gpu = None
            for g in available_sorted:
                if gpu_job_counts.get(g, 0) < MAX_JOBS_PER_GPU:
                    free_gpu = g
                    break

            if free_gpu is None:
                # All available GPUs are at per-GPU capacity; wait a bit before retrying
                print("All available GPUs are at per-GPU capacity. Waiting 10 seconds...")
                time.sleep(10)
                break  # exit inner while to allow cleanup loop to run

            # We have a GPU slot; dequeue and prepare job
            inputs = job_pool.popleft()
            inputs = fix_inputs(inputs)
            if script_args.run_type == "test":
                inputs["num_ep"] = 400
            if invalid_inputs(inputs):
                continue

            # Filter job & get job name
            if script_args.filter_type == "one-hot" and script_args.run_type != "test":
                job_name = get_one_hot_param(inputs, DEFAULT_VALUE)
                if job_name is None:
                    continue
            if script_args.run_type == "test":
                job_name = "test"
            job_dir = os.path.join(RESULT_BASE, script_args.name, inputs["data"], job_name)
            inputs["save_dir"] = job_dir
            os.makedirs(job_dir, exist_ok=True)

            try:
                inputs["gpu"] = free_gpu
                pkl_path = get_save_file_name(inputs)
                # Launch process asynchronously and store in dict
                proc = run_main(inputs)
                job_status[pkl_path] = (proc, inputs, time.time())
                gpu_job_counts[free_gpu] = gpu_job_counts.get(free_gpu, 0) + 1
                print(f"GPU {free_gpu}({gpu_job_counts[free_gpu]}/{MAX_JOBS_PER_GPU}): {inputs}")
            except Exception as e:
                # If launching fails, do not add to pending and log error
                print(f"Failed launching {inputs}: {e}")
                # ensure we don't leave the count incremented (it wasn't incremented yet in this try)
                continue

            if script_args.run_type != "test":
                time.sleep(1)

if __name__ == "__main__":
    main()