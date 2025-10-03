import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from plots.plot_utils import BASELINE_NAMES, TITLE_METHODS
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
                val_str = f"{mean:.4f} $\\pm$ {ste:.4f}"

                if method == best_method:
                    val_str = "\\textbf{" + val_str + "}"

                values.append(val_str)

            prefix = "& " + metric_name if i > 0 else "& " + metric_name
            row_line = f"{prefix} & " + " & ".join(values) + " \\\\\n"
            f.write(row_line)

        # Footer
        f.write("\\end{tabular}\n")

if __name__ == "__main__":

    all_metric_files = glob("metric_log/*.txt")
    nl8_hs256_files = [f for f in all_metric_files if 'nl-8_hs-256' in f and 'recalibrated' not in f]

    all_numbers = []
    for f in nl8_hs256_files:
        with open(f, 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset, number, n_seeds = line.strip().split()
                all_numbers.extend([float(number)] * int(n_seeds))

    # fit min-max scaler and mean-std scaler
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    all_numbers = np.array(all_numbers).reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(all_numbers)

    from plots.plot_utils import PERFORMANCE_METRICS, BASELINE_NAMES

    # for each file, apply 2 scalers and compute mean and ste (std / sqrt(n_seeds))
    from collections import defaultdict
    metric_method_means = defaultdict(dict)
    metric_method_ste = defaultdict(dict)
    for f in nl8_hs256_files:
        method_name = [name for name in BASELINE_NAMES if name in f][0]
        metric_name = [name for name in PERFORMANCE_METRICS if name in f][0]
        print(f"Processing {f} for method {method_name}, metric {metric_name}")
        with open(f, 'r') as file:
            lines = file.readlines()
            total_min_max = 0.0
            total_std = 0.0
            total_seeds = 0
            scaled_numbers = []
            for line in lines:
                dataset, number, n_seeds = line.strip().split()
                if number == 'nan':
                    print(f"{f}: {dataset} is nan, skip")
                    continue
                number = float(number)
                n_seeds = int(n_seeds)
                if metric_name != 'HV':
                    number_min_max = min_max_scaler.transform(np.array([[number]]))[0][0]
                else:
                    number_min_max = number  # for HV, higher is better, do not scale
                total_min_max += number_min_max * n_seeds
                total_seeds += n_seeds
                scaled_numbers.extend([number_min_max] * n_seeds)
            mean_min_max = total_min_max / total_seeds
            mean_std = np.std(scaled_numbers, ddof=1) / (total_seeds ** 0.5)
            metric_method_means[metric_name][method_name] = mean_min_max
            metric_method_ste[metric_name][method_name] = mean_std
    
    write_latex_table(metric_method_means, metric_method_ste, filename="results_table.tex")

