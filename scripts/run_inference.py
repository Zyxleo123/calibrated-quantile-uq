import argparse
import itertools
import shlex
import subprocess
import sys
import os
import time
from collections import deque

BASE = "/home/scratch/yixiz/results/"

def main():
    """
    Main function to parse arguments and manage the experiment execution.
    """
    parser = argparse.ArgumentParser(
        description="Run inference experiments in parallel across multiple GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--gpus', 
        nargs='+', 
        type=int, 
        default=list(range(8)),
        help='List of GPU IDs to use (e.g., --gpus 0 1 2 3).'
    )
    parser.add_argument(
        '--m', 
        type=int, 
        default=1, 
        help='Maximum number of processes to run on each GPU concurrently.'
    )
    parser.add_argument(
        '-n',
        type=str,
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='If set, print commands without executing them.'
    )
    args = parser.parse_args()

    # --- Configuration ---
    datasets = [
        "protein", "concrete", "elevator", "energy", "facebook", 
        "kin8nm", "fusion", "naval", "diamonds", "power", 
        "wine", "yacht", "boston"
    ]
    # losses = [
    #     "calipso", "maqr", "batch_qr", "batch_int", 
    #     "batch_cal", "mpaic", "batch_QRT"
    # ]

    # --- Job Preparation ---
    # Create a queue of all tasks (dataset, loss)
    tasks = deque(itertools.product(datasets))
    total_tasks = len(tasks)
    
    # --- State Tracking ---
    # Dictionary to track available slots on each GPU
    gpu_slots = {gpu_id: args.m for gpu_id in args.gpus}
    # List to keep track of (process, gpu_id) tuples for running jobs
    running_processes = []
    # Cycler for round-robin GPU assignment
    gpu_cycler = itertools.cycle(args.gpus)
    
    print(f"Starting job scheduler...")
    print(f"  Total tasks to run: {total_tasks}")
    print(f"  GPUs available: {args.gpus}")
    print(f"  Max processes per GPU: {args.m}")
    print(f"  Dry run: {args.dry_run}")
    print("-" * 30)

    # --- Main Execution Loop ---
    # Loop continues as long as there are tasks to run OR processes still running
    while tasks or running_processes:
        # 1. Check for and clean up finished processes
        # Iterate over a copy to allow removal from the original list
        for proc, gpu_id in running_processes[:]:
            if proc.poll() is not None:  # process has terminated
                print(f"Process {proc.pid} on GPU {gpu_id} finished.")
                running_processes.remove((proc, gpu_id))
                gpu_slots[gpu_id] += 1  # Free up a slot on the GPU

        # 2. Launch new processes if there are tasks and available GPU slots
        while tasks:
            # Find the next GPU with an available slot
            # We check all GPUs once to find a free slot.
            found_slot = False
            for _ in range(len(args.gpus)):
                next_gpu = next(gpu_cycler)
                if gpu_slots[next_gpu] > 0:
                    dataset, = tasks.popleft()

                    command = [
                        "python",
                        "-u",
                        "inference_all.py",
                        "--data", dataset,
                        "--gpu", str(next_gpu),
                        "--base", os.path.join(BASE, args.n)
                    ]
                    command += ["--dry-run"] if args.dry_run else []

                    # Use shlex.join for safe command line string representation
                    print(f"[{len(tasks)} tasks left] Launching on GPU {next_gpu}: {shlex.join(command)}")

                    process = subprocess.Popen(command)
                    running_processes.append((process, next_gpu))
                    gpu_slots[next_gpu] -= 1
                    
                    found_slot = True
                    break # Stop searching for a GPU and launch the next task
            
            if not found_slot:
                # If no GPU has a free slot, break the inner loop and wait
                break

        # Avoid busy-waiting if all slots are full but processes are still running
        if tasks and not found_slot:
            time.sleep(2) # Wait a couple of seconds before checking for finished jobs again

    print("-" * 30)
    print("All tasks have been launched and completed.")

if __name__ == '__main__':
    main()