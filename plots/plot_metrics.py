import os
import pickle
from typing import Any, Dict, Sequence, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

def load_pickle(path: str) -> Dict[str, Any]:
    """Load a pickle that contains the save_var_names dict data."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default)

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
    tr = _safe_get(data, "tr_loss_list")
    va = _safe_get(data, "va_loss_list")
    te = _safe_get(data, "te_loss_list")

    va_ece = _safe_get(data, "va_ece_list")
    va_sharp = _safe_get(data, "va_sharp_list")

    other_keys = ["va_bag_nll_list", "va_crps_list", "va_mpiw_list", "va_interval_list", "va_check_list"]
    others = {k: _safe_get(data, k) for k in other_keys}

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

def compare_ece_sharpness(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Compare testing ECE and sharpness between original, best, recalibrated original, and recalibrated best.
    Uses keys: te_ece, te_ece_best, recal_te_ece, recal_te_ece_best (and analogous sharpness keys).
    """
    labels = ["Original", "Best", "Recal Original", "Recal Best"]

    # ECE keys
    te_ece = _safe_get(data, "te_ece")
    te_ece_best = _safe_get(data, "te_ece_best")
    recal_te_ece = _safe_get(data, "recal_te_ece")
    recal_te_ece_best = _safe_get(data, "recal_te_ece_best")

    # Sharpness keys
    te_sharp = _safe_get(data, "te_sharp_score")
    te_sharp_best = _safe_get(data, "te_sharp_score_best")
    recal_te_sharp = _safe_get(data, "recal_te_sharp_score")
    recal_te_sharp_best = _safe_get(data, "recal_te_sharp_score_best")

    ece_vals = [te_ece, te_ece_best, recal_te_ece, recal_te_ece_best]
    sharp_vals = [te_sharp, te_sharp_best, recal_te_sharp, recal_te_sharp_best]

    x = np.arange(len(labels))
    width = 0.35

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    ax = axs[0]
    bars_ece = ax.bar(x, [v if v is not None else np.nan for v in ece_vals], color='C0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Test ECE comparison")
    ax.grid(True, axis='y')
    _annotate_bars(ax, bars_ece, fmt="{:.4f}", y_offset=0.0)

    ax2 = axs[1]
    bars_sharp = ax2.bar(x, [v if v is not None else np.nan for v in sharp_vals], color='C1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_title("Test Sharpness comparison")
    ax2.grid(True, axis='y')
    _annotate_bars(ax2, bars_sharp, fmt="{:.4f}", y_offset=0.0)

    fig.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def compare_scoring_rules(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Compare bag_nll, crps, mpiw, interval, check on test set.
    Uses keys with and without _best and recal_ prefixes: e.g., te_bag_nll, te_bag_nll_best, recal_te_bag_nll, recal_te_bag_nll_best.
    """
    metrics = ["bag_nll", "crps", "mpiw", "interval", "check"]
    labels = ["Original", "Best", "Recal Original", "Recal Best"]

    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
    for i, m in enumerate(metrics):
        k0 = f"te_{m}"
        k1 = f"te_{m}_best"
        k2 = f"recal_te_{m}"
        k3 = f"recal_te_{m}_best"
        vals = [_safe_get(data, k) for k in (k0, k1, k2, k3)]
        ax = axs[0, i]
        bars = ax.bar(np.arange(4), [v if v is not None else np.nan for v in vals], color=['C0','C1','C2','C3'])
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labels, rotation=10)
        ax.set_title(m)
        ax.grid(True, axis='y')
        _annotate_bars(ax, bars, fmt="{:.4f}", y_offset=0.0)

    fig.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def calibration_plot(data: Dict[str, Any], outpath: Optional[str]=None, show: bool=False):
    """
    Plot observed props vs expected props for test set for:
     - original (te_exp_props, te_obs_props)
     - best (te_exp_props_best, te_obs_props_best)
     - recalibrated original (use te_exp_props but recal_te_obs_props)
     - recalibrated best (use te_exp_props_best but recal_te_obs_props_best)
    """
    series = []
    # Original
    if _safe_get(data, "te_exp_props") is not None and _safe_get(data, "te_obs_props") is not None:
        series.append(("Original", data["te_exp_props"], data["te_obs_props"]))
    # Best
    if _safe_get(data, "te_exp_props_best") is not None and _safe_get(data, "te_obs_props_best") is not None:
        series.append(("Best", data["te_exp_props_best"], data["te_obs_props_best"]))
    # Recalibrated original
    if _safe_get(data, "te_exp_props") is not None and _safe_get(data, "recal_te_obs_props") is not None:
        series.append(("Recal Original", data["te_exp_props"], data["recal_te_obs_props"]))
    # Recalibrated best
    if _safe_get(data, "te_exp_props_best") is not None and _safe_get(data, "recal_te_obs_props_best") is not None:
        series.append(("Recal Best", data["te_exp_props_best"], data["recal_te_obs_props_best"]))

    if not series:
        return

    fig, ax = plt.subplots(figsize=(6,6))
    for name, exp, obs in series:
        exp_a = np.asarray(exp)
        obs_a = np.asarray(obs)
        ax.plot(exp_a, obs_a, marker='o', linestyle='-', label=name)
    # diagonal
    mn = 0.0
    mx = 1.0
    ax.plot([mn, mx], [mn, mx], 'k--', label='ideal')
    ax.set_xlabel("Expected proportion")
    ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration plot (test)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)