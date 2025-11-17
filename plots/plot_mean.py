import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from plots.plot_utils import BASELINE_NAMES, TITLE_METHODS
from collections import defaultdict

def write_latex_table(metric_method_means, metric_method_ste, filename="results_table.tex"):
    """
    Writes a LaTeX table with IGD, GD, HV metrics.
    Only the method with minimum (for ↓) or maximum (for ↑) mean is bolded.
    """
    metrics = [
        ("IGD", "IGD $(\\downarrow)$", "min"),
        ("GD", "GD $(\\downarrow)$", "min"),
        ("HV", "HV ($\\uparrow$)", "max")
    ]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        # Header
        f.write("\\begin{tabular}{l@{\\hspace{-0.1ex}}l" + "c" * len(BASELINE_NAMES) + "r}\n")
        f.write("\\toprule\n")
        f.write("        &  & " + " & ".join([f"{{{TITLE_METHODS[m]}}}" for m in BASELINE_NAMES]) + " \\\\\n\n")
        f.write("\\midrule\n")
        f.write("\\multirow{" + str(len(metrics)) + "}{4em}{All datasets} ")

        # Rows
        for i, (metric_key, metric_name, direction) in enumerate(metrics):
            means = {m: metric_method_means[metric_key][m] for m in BASELINE_NAMES}
            stes = {m: metric_method_ste[metric_key][m] for m in BASELINE_NAMES}

            # find best mean
            if direction == "min":
                best_method = min(means, key=means.get)
            else:
                best_method = max(means, key=means.get)

            values = []
            for method in BASELINE_NAMES:
                mean, ste = means[method], stes[method]
                val_str = f"{mean} $\\pm$ {ste}"

                if method == best_method:
                    val_str = "\\textbf{" + val_str + "}"

                values.append(val_str)

            prefix = "& " + metric_name if i > 0 else "& " + metric_name
            row_line = f"{prefix} & " + " & ".join(values) + " \\\\\n"
            f.write(row_line)

        # Footer
        f.write("\\end{tabular}\n")

def main(args):
    all_metric_files = glob(os.path.join("metric_log", args.n, args.base_metric, "*.txt"))
    if args.r:
        all_metric_files = [f for f in all_metric_files if 'recalibrated' in f]
    else:
        all_metric_files = [f for f in all_metric_files if 'recalibrated' not in f]

    # nl8_hs256_files = [f for f in all_metric_files if 'nl-8_hs-256' in f and 'recalibrated' not in f]
    nl8_hs256_files = [f for f in all_metric_files if 'nl-8_hs-256' in f]

    all_numbers = defaultdict(list)
    for f in nl8_hs256_files:
        with open(f, 'r') as file:
            if 'HV' in f:
                continue
            lines = file.readlines()
            for line in lines:
                # dataset mean n_seeds number1 number2 ... numbern
                dataset, mean, n_seeds, *rest = line.strip().split()
                all_numbers[dataset].extend([float(x) for x in rest])

    # fit min-max scaler and mean-std scaler
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    min_max_scalers = defaultdict(MinMaxScaler)
    for dataset in all_numbers:
        numbers = np.array(all_numbers[dataset]).reshape(-1, 1)
        min_max_scalers[dataset].fit(numbers)
    # medians = {}
    # for dataset in all_numbers:
    #     numbers = np.array(all_numbers[dataset])
    #     medians[dataset] = np.median(numbers)

    from plots.plot_utils import PERFORMANCE_METRICS, BASELINE_NAMES

    # for each file, apply 2 scalers and compute mean and ste (std / sqrt(n_seeds))
    metric_method_means = defaultdict(dict)
    metric_method_ste = defaultdict(dict)
    for f in nl8_hs256_files:
        parts = os.path.basename(f).split('_')
        metric_name = (set(parts) & set(PERFORMANCE_METRICS)).pop()
        metric_name_idx = parts.index(metric_name)
        method_name = '_'.join(parts[:metric_name_idx])

        print(f"Processing {f} for method {method_name}, metric {metric_name}")
        with open(f, 'r') as file:
            lines = file.readlines()
            total_scaled = 0.0
            total_seeds = 0
            scaled_numbers = []
            for line in lines:
                dataset, _, _, *rest = line.strip().split()
                if metric_name != 'HV':
                    scaler = min_max_scalers[dataset]
                    scaled_nums = scaler.transform(np.array([float(x) for x in rest]).reshape(-1, 1)).flatten()
                    # scaled_nums = [float(x) / medians[dataset] for x in rest]
                else:
                    scaled_nums = [float(x) for x in rest]
                total_scaled += sum(scaled_nums)
                scaled_numbers.extend(scaled_nums)
            mean_min_max = np.mean(scaled_numbers)
            mean_std = np.std(scaled_numbers, ddof=1) / (len(scaled_numbers) ** 0.5)
            metric_method_means[metric_name][method_name] = mean_min_max
            metric_method_ste[metric_name][method_name] = mean_std
    
    write_latex_table(metric_method_means, metric_method_ste, 
                    filename=os.path.join("metric_log", args.n, args.base_metric, 'tables',
                                           f"results_table{'_recalibrated' if args.r else ''}.tex"))

BASE_METRICS = ['sharp_score', 'variance', 'mpiw']
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default=False, action='store_true')
    parser.add_argument('-n', type=str, required=True)
    args = parser.parse_args()
    for base_metric in BASE_METRICS:
        args.base_metric = base_metric
        main(args)

