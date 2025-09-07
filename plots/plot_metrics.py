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
    if outpath:
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        target = base + ext
        fig_main.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(fig_main)

def compare_ece_sharpness(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Compare testing ECE and sharpness between original and controlled models (trained with different ECE thresholds).
    Creates a 2x2 plot:
    - Row 1: Before Recalibration (ECE, Sharpness)
    - Row 2: After Recalibration (ECE, Sharpness)
    """
    thresholds = safe_get(data, "te_ece_controlled")

    labels = ["Original"] + [f"Controlled@{t:.3f}" for t in thresholds]
    x = np.arange(len(labels))

    # Data extraction
    te_ece_orig = safe_get(data, "te_ece")
    te_ece_controlled = safe_get(data, "te_ece_controlled", [])
    te_sharp_orig = safe_get(data, "te_sharp_score")
    te_sharp_controlled = safe_get(data, "te_sharp_score_controlled", [])

    recal_te_ece_orig = safe_get(data, "recal_te_ece")
    recal_te_ece_best_list = safe_get(data, "recal_te_ece_controlled", [])
    recal_te_sharp_orig = safe_get(data, "recal_te_sharp_score")
    recal_te_sharp_best_list = safe_get(data, "recal_te_sharp_score_controlled", [])

    # Combine data for plotting
    ece_vals_before = [te_ece_orig] + te_ece_controlled
    sharp_vals_before = [te_sharp_orig] + te_sharp_controlled
    ece_vals_after = [recal_te_ece_orig] + recal_te_ece_best_list
    sharp_vals_after = [recal_te_sharp_orig] + recal_te_sharp_best_list

    fig, axs = plt.subplots(2, 2, figsize=(max(12, 1.5 * len(labels)), 10), sharex=True)

    def plot_bars(ax, values, title):
        bars = ax.bar(x, values)
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
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        target = base + ext
        fig.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def compare_scoring_rules(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Compare various scoring rules on the test set for original vs. controlled models.
    Plots are arranged in two rows: before and after recalibration.
    """
    thresholds = safe_get(data, "te_ece_controlled")

    metrics = ["bag_nll", "crps", "mpiw", "interval", "check", "cali_score"]
    labels = ["Original"] + [f"Controlled@{t:.3f}" for t in thresholds]
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
        k_best_before = f"te_{m}_controlled"
        vals_before = [safe_get(data, k_orig_before)] + pad_list(safe_get(data, k_best_before, []), len(thresholds))

        k_orig_after = f"recal_te_{m}"
        k_best_after = f"recal_te_{m}_controlled"
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
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        target = base + ext
        fig.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def calibration_plot(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Plot calibration curves for original and controlled models.
    Produces two plots:
    1. All curves on a single axis.
    2. Pairwise comparisons of each controlled model against the original model, in subplots.
    """
    thresholds = safe_get(data, "te_ece_controlled")
    if thresholds is None:
        print("Warning: 'thresholds' key not found. Skipping calibration_plot.")
        return

    # --- Data Extraction ---
    orig_exp_props = safe_get(data, "te_exp_props")
    orig_obs_props = safe_get(data, "te_obs_props")
    recal_orig_obs_props = safe_get(data, "recal_te_obs_props")

    controlled_exp_props_list = safe_get(data, "te_exp_props_controlled", [])
    controlled_obs_props = safe_get(data, "te_obs_props_controlled", [])
    recal_controlled_obs_props_list = safe_get(data, "recal_te_obs_props_controlled", [])


    valid_indices = [i for i, qp in enumerate(controlled_exp_props_list) if qp is not None and i < len(thresholds)]
    if not valid_indices:
        return

    n_plots = len(valid_indices)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, sharex=True, sharey=True)
    axs = axs.flatten()

    for plot_idx, model_idx in enumerate(valid_indices):
        ax = axs[plot_idx]
        t = thresholds[model_idx]
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal')

        if orig_exp_props is not None and orig_obs_props is not None:
            ax.plot(orig_exp_props, orig_obs_props, marker='o', linestyle='-', label="Original", color='gray')
        if orig_exp_props is not None and recal_orig_obs_props is not None:
            ax.plot(orig_exp_props, recal_orig_obs_props, marker='x', linestyle='--', label="Recal Original", color='dimgray')

        controlled_exp_props = controlled_exp_props_list[model_idx]
        if model_idx < len(controlled_obs_props) and controlled_obs_props[model_idx] is not None:
             ax.plot(controlled_exp_props, controlled_obs_props[model_idx], marker='o', linestyle='-', label="Controlled")
        if model_idx < len(recal_controlled_obs_props_list) and recal_controlled_obs_props_list[model_idx] is not None:
             ax.plot(controlled_exp_props, recal_controlled_obs_props_list[model_idx], marker='x', linestyle='--', label="Recal Controlled")
        
        ax.set_title(f"Threshold <= {t:.3f}")
        ax.legend()
        ax.grid(True)
        if plot_idx >= (rows - 1) * cols:
            ax.set_xlabel("Expected proportion")
        if plot_idx % cols == 0:
            ax.set_ylabel("Observed proportion")

    for i in range(n_plots, len(axs)):
        axs[i].set_visible(False)

    fig.suptitle("Calibration Plot: Pairwise Comparisons with Original", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    
    if outpath:
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        target = base + ext
        fig.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def plot_ece_sharpness(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Scatter plot of best-model ECE vs. Sharpness for original and recalibrated results.

    Layout: 2x2 grid
    - Row 0: Test before / Test after recalibration
    - Row 1: Val before / Val after recalibration

    X-axis runs from 0 to data.args.max_ece_thres (falls back to 1.0).
    Y-axis lower bound is 0; upper bound is max(0.3, observed_max*1.05) so points above 0.3 are allowed.
    """
    te_ece_controlled = safe_get(data, "te_ece_controlled", [])
    te_sharp_controlled = safe_get(data, "te_sharp_score_controlled", [])
    recal_ece_controlled = safe_get(data, "recal_te_ece_controlled", [])
    recal_sharp_controlled = safe_get(data, "recal_te_sharp_score_controlled", [])
    va_ece_controlled = safe_get(data, "va_ece_controlled", [])
    va_sharp_controlled = safe_get(data, "va_sharp_score_controlled", [])
    recal_va_ece_controlled = safe_get(data, "recal_va_ece_controlled", [])
    recal_va_sharp_controlled = safe_get(data, "recal_va_sharp_score_controlled", [])

    va_marginal_sharpness = safe_get(data, "va_marginal_sharpness")
    te_marginal_sharpness = safe_get(data, "te_marginal_sharpness")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    panels = [
        (axs[0,0], te_ece_controlled, te_sharp_controlled, te_marginal_sharpness,"Test: Before Recalibration"),
        (axs[0,1], recal_ece_controlled, recal_sharp_controlled, te_marginal_sharpness, "Test: After Recalibration"),
        (axs[1,0], va_ece_controlled, va_sharp_controlled, va_marginal_sharpness, "Val: Before Recalibration"),
        (axs[1,1], recal_va_ece_controlled, recal_va_sharp_controlled, va_marginal_sharpness, "Val: After Recalibration"),
    ]

    for ax, x, y, marginal_sharpness, title in panels:
        ax.scatter(x, y, s=40)
        # ax.axhline(marginal_sharpness, color='r', linestyle=':', linewidth=1.5, label='Marginal Sharpness')
        ax.set_title(title)
        ax.set_xlabel("ECE")
        ax.set_ylabel("Sharpness")
        ax.grid(True)
        ax.legend()
    fig.suptitle(f"{safe_get(data, 'args')['data']} Seed {safe_get(data, 'args')['seed']}: ECE-Sharpness", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outpath:
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        target = base + ext
        fig.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def overlap_ece_sharpness(datas: list, names: list, outpath: Optional[str]=None, show: bool=False):
    """
    Scatter plot of best-model ECE vs. Sharpness for multiple datasets/models on the same axes.

    Layout: 2x2 grid
    - Row 0: Test before / Test after recalibration
    - Row 1: Val before / Val after recalibration
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    panels = [
        (axs[0,0], "Test: Before Recalibration"),
        (axs[0,1], "Test: After Recalibration"),
        (axs[1,0], "Val: Before Recalibration"),
        (axs[1,1], "Val: After Recalibration"),
    ]

    va_marginal_sharpness = safe_get(datas[0], "va_marginal_sharpness")
    te_marginal_sharpness = safe_get(datas[0], "te_marginal_sharpness")

    for data, name in zip(datas, names):
        te_ece_controlled = safe_get(data, "te_ece_controlled", [])
        te_sharp_controlled = safe_get(data, "te_sharp_score_controlled", [])
        recal_ece_controlled = safe_get(data, "recal_te_ece_controlled", [])
        recal_sharp_controlled = safe_get(data, "recal_te_sharp_score_controlled", [])
        va_ece_controlled = safe_get(data, "va_ece_controlled", [])
        va_sharp_controlled = safe_get(data, "va_sharp_score_controlled", [])
        recal_va_ece_controlled = safe_get(data, "recal_va_ece_controlled", [])
        recal_va_sharp_controlled = safe_get(data, "recal_va_sharp_score_controlled", [])
        panel_data = [
            (te_ece_controlled, te_sharp_controlled, te_marginal_sharpness),
            (recal_ece_controlled, recal_sharp_controlled, te_marginal_sharpness),
            (va_ece_controlled, va_sharp_controlled, va_marginal_sharpness),
            (recal_va_ece_controlled, recal_va_sharp_controlled, va_marginal_sharpness),
        ]
        for (ax, title), (x, y, marginal_sharpness) in zip(panels, panel_data):
            ax.scatter(x, y, s=40, label=name)
            # ax.axhline(marginal_sharpness, color='r', linestyle=':', linewidth=1.5, label='Marginal Sharpness')
            ax.set_title(title)
            ax.set_xlabel("ECE")
            ax.set_ylabel("Sharpness")
            ax.grid(True)
            ax.legend()
        
    fig.suptitle(f"{safe_get(data, 'args')['data']} Seed {safe_get(data, 'args')['seed']}: ECE-Sharpness", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outpath:
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(outpath)
        if ext == "":
            ext = ".png"
        target = base + ext
        fig.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
