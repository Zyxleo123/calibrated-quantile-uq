import os
from typing import Any, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import safe_get
from utils.misc_utils import EceSharpFrontier

def _annotate_bars(ax, bars, fmt="{:.3f}", y_offset=0.0):
    """
    Write the numeric value on top of each bar in 'bars'.
    - ax: matplotlib axis
    - bars: container returned by ax.bar(...)
    - fmt: format for numbers
    - y_offset: additional offset in axis units to avoid overlap
    """
    for rect in bars:
        try:
            h = rect.get_height()
        except Exception:
            continue
        if np.isnan(h):
            continue
        x = rect.get_x() + rect.get_width() / 2.0
        # choose vertical alignment based on sign
        va = 'bottom' if h >= 0 else 'top'
        ax.text(x, h + y_offset, fmt.format(h), ha='center', va=va, fontsize=8, rotation=0)

def plot_training_stats(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Plot training statistics lists (tr_loss_list, va_loss_list, te_loss_list, va_ece_list, va_sharp_list,
    va_bag_nll_list, va_crps_list, va_mpiw_list, va_interval_list, va_check_list).
    Separate outputs: one figure for loss/ece/sharpness, another for other scores (each metric its own subplot).
    """
    tr = safe_get(data, "tr_loss_list")
    va = safe_get(data, "va_loss_list")
    te = safe_get(data, "te_loss_list")

    va_ece = safe_get(data, "va_ece_list")
    va_sharp = safe_get(data, "va_sharp_list")

    other_keys = ["va_bag_nll_list", "va_crps_list", "va_mpiw_list", "va_interval_list", "va_check_list"]
    others = {k: safe_get(data, k) for k in other_keys}

    # Prepare outpaths if provided (base + suffix)
    main_out = None
    others_out = None
    if outpath:
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        main_out = f"{base}_main{ext}"
        others_out = f"{base}_others{ext}"
        # ensure dir exists
        main_dir = os.path.dirname(main_out)
        if main_dir:
            os.makedirs(main_dir, exist_ok=True)

    # Main figure: loss, ece, sharpness
    fig_main, axs_main = plt.subplots(1, 3, figsize=(15, 4))
    ax_loss = axs_main[0]
    ax_ece = axs_main[1]
    ax_sharp = axs_main[2]

    if tr is not None:
        ax_loss.plot(tr, label="train")
    if va is not None:
        ax_loss.plot(va, label="val")
    if te is not None:
        ax_loss.plot(te, label="test")
    ax_loss.set_title("Loss lists")
    ax_loss.legend()
    ax_loss.grid(True)

    if va_ece is not None:
        ax_ece.plot(va_ece, marker="o", label="val ece")
    ax_ece.set_title("Validation ECE (list)")
    ax_ece.grid(True)

    if va_sharp is not None:
        ax_sharp.plot(va_sharp, marker="o", label="val sharpness")
    ax_sharp.set_title("Validation Sharpness (list)")
    ax_sharp.grid(True)

    fig_main.tight_layout()
    if main_out:
        fig_main.savefig(main_out, dpi=150)
    if show:
        plt.show()
    plt.close(fig_main)

    # Other metrics: create a separate figure with one subplot per present metric
    present = [(k, v) for k, v in others.items() if v is not None]
    if not present:
        return

    n = len(present)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig_o, axs_o = plt.subplots(rows, cols, figsize=(5*cols, 3*rows), squeeze=False)
    for i, (k, v) in enumerate(present):
        r = i // cols
        c = i % cols
        ax = axs_o[r][c]
        # plot the list
        ax.plot(v, marker='o', linestyle='-')
        ax.set_title(k.replace("va_", "").replace("_list", ""))
        ax.grid(True)
    # hide any unused axes
    for j in range(n, rows*cols):
        r = j // cols
        c = j % cols
        axs_o[r][c].set_visible(False)

    fig_o.tight_layout()
    if others_out:
        fig_o.savefig(others_out, dpi=150)
    if show:
        plt.show()
    plt.close(fig_o)

# EDITED BAR GRAPH FUNCTIONS
def compare_ece_sharpness(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Compare testing ECE and sharpness between original and best models (trained with different ECE thresholds).
    Creates a 2x2 plot:
    - Row 1: Before Recalibration (ECE, Sharpness)
    - Row 2: After Recalibration (ECE, Sharpness)
    """
    thresholds = safe_get(data, "thresholds")
    if thresholds is None:
        print("Warning: 'thresholds' key not found. Skipping compare_ece_sharpness plot.")
        return

    labels = ["Original"] + [f"Best@{t:.3f}" for t in thresholds]
    x = np.arange(len(labels))

    # Data extraction
    te_ece_orig = safe_get(data, "te_ece")
    te_ece_best_list = safe_get(data, "te_ece_list_best", [])
    te_sharp_orig = safe_get(data, "te_sharp_score")
    te_sharp_best_list = safe_get(data, "te_sharp_score_list_best", [])

    recal_te_ece_orig = safe_get(data, "recal_te_ece")
    recal_te_ece_best_list = safe_get(data, "recal_te_ece_list_best", [])
    recal_te_sharp_orig = safe_get(data, "recal_te_sharp_score")
    recal_te_sharp_best_list = safe_get(data, "recal_te_sharp_score_list_best", [])

    def pad_list(lst, length):
        return (lst + [None] * length)[:length]

    # Combine data for plotting
    ece_vals_before = [te_ece_orig] + pad_list(te_ece_best_list, len(thresholds))
    sharp_vals_before = [te_sharp_orig] + pad_list(te_sharp_best_list, len(thresholds))
    ece_vals_after = [recal_te_ece_orig] + pad_list(recal_te_ece_best_list, len(thresholds))
    sharp_vals_after = [recal_te_sharp_orig] + pad_list(recal_te_sharp_best_list, len(thresholds))

    def to_nan(l):
        return [v if v is not None else np.nan for v in l]

    fig, axs = plt.subplots(2, 2, figsize=(max(12, 1.5 * len(labels)), 10), sharex=True)

    def plot_bars(ax, values, title):
        bars = ax.bar(x, to_nan(values))
        ax.set_title(title)
        ax.grid(True, axis='y')
        _annotate_bars(ax, bars, fmt="{:.4f}")

    # Row 1: Before Recalibration
    plot_bars(axs[0, 0], ece_vals_before, "ECE (Before Recalibration)")
    plot_bars(axs[0, 1], sharp_vals_before, "Sharpness (Before Recalibration)")

    # Row 2: After Recalibration
    plot_bars(axs[1, 0], ece_vals_after, "ECE (After Recalibration)")
    plot_bars(axs[1, 1], sharp_vals_after, "Sharpness (After Recalibration)")

    plt.setp(axs[1, :], xticks=x, xticklabels=labels)
    for ax in axs[1, :]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Test ECE and Sharpness Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outpath:
        out_dir = os.path.dirname(outpath)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def compare_scoring_rules(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Compare various scoring rules on the test set for original vs. best models.
    Plots are arranged in two rows: before and after recalibration.
    """
    thresholds = safe_get(data, "thresholds")
    if thresholds is None:
        print("Warning: 'thresholds' key not found. Skipping compare_scoring_rules plot.")
        return

    metrics = ["bag_nll", "crps", "mpiw", "interval", "check", "cali_score"]
    labels = ["Original"] + [f"Best@{t:.3f}" for t in thresholds]
    x = np.arange(len(labels))

    n_metrics = len(metrics)
    fig, axs = plt.subplots(2, n_metrics, figsize=(max(4 * n_metrics, 1.5 * len(labels)), 8), sharex=True, squeeze=False)

    def to_nan(l):
        return [v if v is not None else np.nan for v in l]
    
    def pad_list(lst, length):
        return (lst + [None] * length)[:length]

    for i, m in enumerate(metrics):
        ax_before = axs[0, i]
        ax_after = axs[1, i]

        k_orig_before = f"te_{m}"
        k_best_before = f"te_{m}_list_best"
        vals_before = [safe_get(data, k_orig_before)] + pad_list(safe_get(data, k_best_before, []), len(thresholds))

        k_orig_after = f"recal_te_{m}"
        k_best_after = f"recal_te_{m}_list_best"
        vals_after = [safe_get(data, k_orig_after)] + pad_list(safe_get(data, k_best_after, []), len(thresholds))

        bars_before = ax_before.bar(x, to_nan(vals_before))
        ax_before.set_title(m)
        ax_before.grid(True, axis='y')
        _annotate_bars(ax_before, bars_before, fmt="{:.4f}")

        bars_after = ax_after.bar(x, to_nan(vals_after))
        ax_after.grid(True, axis='y')
        _annotate_bars(ax_after, bars_after, fmt="{:.4f}")

    plt.setp(axs[1, :], xticks=x, xticklabels=labels)
    for ax in axs[1, :]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    axs[0, 0].set_ylabel("Before Recalibration", fontsize=12)
    axs[1, 0].set_ylabel("After Recalibration", fontsize=12)

    fig.suptitle("Test Scoring Rules Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outpath:
        out_dir = os.path.dirname(outpath)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def calibration_plot(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Plot calibration curves for original and best models.
    Produces two plots:
    1. All curves on a single axis.
    2. Pairwise comparisons of each best model against the original model, in subplots.
    """
    thresholds = safe_get(data, "thresholds")
    if thresholds is None:
        print("Warning: 'thresholds' key not found. Skipping calibration_plot.")
        return

    # --- Data Extraction ---
    orig_exp_props = safe_get(data, "te_exp_props")
    orig_obs_props = safe_get(data, "te_obs_props")
    recal_orig_obs_props = safe_get(data, "recal_te_obs_props")

    best_exp_props_list = safe_get(data, "te_exp_props_list_best", [])
    best_obs_props_list = safe_get(data, "te_obs_props_list_best", [])
    recal_best_obs_props_list = safe_get(data, "recal_te_obs_props_list_best", [])

    # Prepare outpaths for the two approaches
    all_in_one_out = None
    pairwise_out = None
    if outpath:
        base, ext = os.path.splitext(outpath)
        if not ext: ext = ".png"
        out_dir = os.path.dirname(base)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        all_in_one_out = f"{base}_all_in_one{ext}"
        pairwise_out = f"{base}_pairwise{ext}"

    # --- Approach 1: All curves on one plot ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot([0, 1], [0, 1], 'k--', label='Ideal')

    if orig_exp_props is not None and orig_obs_props is not None:
        ax1.plot(orig_exp_props, orig_obs_props, marker='o', linestyle='-', label="Original")
    if orig_exp_props is not None and recal_orig_obs_props is not None:
        ax1.plot(orig_exp_props, recal_orig_obs_props, marker='x', linestyle='--', label="Recal Original")

    for i, t in enumerate(thresholds):
        if i < len(best_exp_props_list) and best_exp_props_list[i] is not None:
            q_preds_best = best_exp_props_list[i]
            if i < len(best_obs_props_list) and best_obs_props_list[i] is not None:
                ax1.plot(q_preds_best, best_obs_props_list[i], marker='.', linestyle='-', label=f"Best@{t:.3f}")
            if i < len(recal_best_obs_props_list) and recal_best_obs_props_list[i] is not None:
                ax1.plot(q_preds_best, recal_best_obs_props_list[i], marker='.', linestyle='--', label=f"Recal Best@{t:.3f}")

    ax1.set_xlabel("Expected proportion (Quantile level)")
    ax1.set_ylabel("Observed proportion")
    ax1.set_title("Calibration Plot: All Models")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    fig1.tight_layout(rect=[0, 0, 0.75, 1])

    if all_in_one_out:
        fig1.savefig(all_in_one_out, dpi=150)
    if show:
        plt.show()
    plt.close(fig1)

    # --- Approach 2: Pairwise comparison plots in subplots ---
    valid_indices = [i for i, qp in enumerate(best_exp_props_list) if qp is not None and i < len(thresholds)]
    if not valid_indices:
        return

    n_plots = len(valid_indices)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))
    fig2, axs2 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, sharex=True, sharey=True)
    axs2 = axs2.flatten()

    for plot_idx, model_idx in enumerate(valid_indices):
        ax = axs2[plot_idx]
        t = thresholds[model_idx]
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal')

        if orig_exp_props is not None and orig_obs_props is not None:
            ax.plot(orig_exp_props, orig_obs_props, marker='o', linestyle='-', label="Original", color='gray')
        if orig_exp_props is not None and recal_orig_obs_props is not None:
            ax.plot(orig_exp_props, recal_orig_obs_props, marker='x', linestyle='--', label="Recal Original", color='dimgray')

        q_preds_best = best_exp_props_list[model_idx]
        if model_idx < len(best_obs_props_list) and best_obs_props_list[model_idx] is not None:
             ax.plot(q_preds_best, best_obs_props_list[model_idx], marker='o', linestyle='-', label="Best")
        if model_idx < len(recal_best_obs_props_list) and recal_best_obs_props_list[model_idx] is not None:
             ax.plot(q_preds_best, recal_best_obs_props_list[model_idx], marker='x', linestyle='--', label="Recal Best")
        
        ax.set_title(f"Threshold <= {t:.3f}")
        ax.legend()
        ax.grid(True)
        if plot_idx >= (rows - 1) * cols:
            ax.set_xlabel("Expected proportion")
        if plot_idx % cols == 0:
            ax.set_ylabel("Observed proportion")

    for i in range(n_plots, len(axs2)):
        axs2[i].set_visible(False)

    fig2.suptitle("Calibration Plot: Pairwise Comparisons with Original", fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    if pairwise_out:
        fig2.savefig(pairwise_out, dpi=150)
    if show:
        plt.show()
    plt.close(fig2)

def plot_ece_sharpness(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Scatter plot of best-model ECE vs. Sharpness for original and recalibrated results.

    Layout: 2x2 grid
    - Row 0: Test before / Test after recalibration
    - Row 1: Val before / Val after recalibration

    X-axis runs from 0 to data.args.max_ece_thres (falls back to 1.0).
    Y-axis lower bound is 0; upper bound is max(0.3, observed_max*1.05) so points above 0.3 are allowed.
    """
    args = safe_get(data, "args", None)
    te_ece_best = safe_get(data, "te_ece_list_best", [])
    te_sharp_best = safe_get(data, "te_sharp_score_list_best", [])
    recal_ece_best = safe_get(data, "recal_te_ece_list_best", [])
    recal_sharp_best = safe_get(data, "recal_te_sharp_score_list_best", [])
    va_ece_best = safe_get(data, "va_ece_list_best", [])
    va_sharp_best = safe_get(data, "va_sharp_score_list_best", [])
    recal_va_ece_best = safe_get(data, "recal_va_ece_list_best", [])
    recal_va_sharp_best = safe_get(data, "recal_va_sharp_score_list_best", [])
    def get_frontier(ece, sharp):
        frontier = EceSharpFrontier.from_list(list(zip(ece, sharp))).get_thresholded_frontier(args.min_thres, args.max_thres, args.num_thres).get_entries()
        ece_frontier = [entry["ece"] for entry in frontier]
        sharp_frontier = [entry["sharp"] for entry in frontier]
        return ece_frontier, sharp_frontier
    import pudb; pudb.set_trace()
    te_ece_best, te_sharp_best = get_frontier(te_ece_best, te_sharp_best)
    recal_ece_best, recal_sharp_best = get_frontier(recal_ece_best, recal_sharp_best)
    va_ece_best, va_sharp_best = get_frontier(va_ece_best, va_sharp_best)
    recal_va_ece_best, recal_va_sharp_best = get_frontier(recal_va_ece_best, recal_va_sharp_best)

    def to_array(lst):
        if lst is None:
            return np.array([], dtype=float)
        return np.array([np.nan if v is None else v for v in lst], dtype=float)

    x_te_before = to_array(te_ece_best)
    y_te_before = to_array(te_sharp_best)
    x_te_after = to_array(recal_ece_best)
    y_te_after = to_array(recal_sharp_best)

    x_va_before = to_array(va_ece_best)
    y_va_before = to_array(va_sharp_best)
    x_va_after = to_array(recal_va_ece_best)
    y_va_after = to_array(recal_va_sharp_best)

    if (x_te_before.size == 0 and x_te_after.size == 0 and
        x_va_before.size == 0 and x_va_after.size == 0):
        print("Warning: no best ECE/sharpness data found (test or val). Skipping plot_ece_sharpness.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    panels = [
        (axs[0,0], x_te_before, y_te_before, "Test: Before Recalibration"),
        (axs[0,1], x_te_after,  y_te_after,  "Test: After Recalibration"),
        (axs[1,0], x_va_before, y_va_before, "Val: Before Recalibration"),
        (axs[1,1], x_va_after,  y_va_after,  "Val: After Recalibration"),
    ]

    for ax, x, y, title in panels:
        ax.plot(x, y, marker='o', linestyle='-', label='Best models')
        ax.scatter(x, y, s=40)

        ax.set_title(title)
        ax.set_xlabel("ECE")
        ax.set_ylabel("Sharpness")
        ax.grid(True)

    fig.suptitle("Best-model ECE vs Sharpness (Test and Val)", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outpath:
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        target = f"{base}_ece_vs_sharp_test_val{ext}"
        fig.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(fig)