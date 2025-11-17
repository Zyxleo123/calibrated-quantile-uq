import pickle
from typing import Any, Dict
import os
import matplotlib.cm as cm

RESULTS_BASE_DIR = os.path.join(os.environ.get('SCRATCH', '.'), "results")
BASELINE_NAMES = ['batch_qr', 'batch_cal', 'batch_int', 'maqr', 'calipso', 'batch_QRT',
                #   'mpaic0.2', 'mpaic0.4', 'mpaic0.6', 'mpaic0.8', 'mpaic0.95', 'mpaic1.0']
                'mpaic0.6']
HYPER_NAMES = ['nl-1_hs-32', 'nl-2_hs-64', 'nl-4_hs-128', 'nl-8_hs-256']
SEEDS = [f'_{i}' for i in range(10)]
Y_METRICS = [
    'te_sharp_score_controlled',
    'te_variance_controlled',
    'te_mpiw_controlled',
    'te_crps_controlled',
    'te_interval_controlled',
    'te_check_controlled',
    'te_bag_nll_controlled',
    'te_va_ece_exceedance',
]
TITLE_METRICS = {
    'te_sharp_score_controlled': 'Sharpness',
    'te_variance_controlled': 'Variance',
    'te_mpiw_controlled': 'MPIW',
    'te_crps_controlled': 'CRPS',
    'te_interval_controlled': 'Interval Score',
    'te_check_controlled': 'Check Score',
    'te_bag_nll_controlled': 'Bag NLL',
    'te_va_ece_exceedance': 'ECE Exceedance'
}
TITLE_METHODS = {
    'batch_qr': 'CGQR',
    'batch_cal': 'Cali',
    'batch_int': 'Interval',
    'maqr': 'MAQR',
    'calipso': 'Calipso',
    'batch_QRT': 'QRT',
    'mpaic0.2': 'MPAIC 0.2',
    'mpaic0.4': 'MPAIC 0.4',
    'mpaic0.6': 'MPAIC',
    'mpaic0.8': 'MPAIC 0.8',
    'mpaic0.95': 'MPAIC 0.95',
    'mpaic1.0': 'MPAIC 1.0',
}
PERFORMANCE_METRICS = ['IGD', 'IGD+', 'GD', 'GD+', 'HV']
COLORS = ["blue", "red", "green", "brown", "purple", "gray",
          "orange", "pink", "olive", "cyan", "magenta", "teal"]
METHOD_COLORS = {method: COLORS[i] for i, method in enumerate(BASELINE_NAMES)}
HYPER_COLORS = {hyper: COLORS[i] for i, hyper in enumerate(HYPER_NAMES)}
METHOD_MARKERS = {
    'batch_qr': 'o',  # Circle
    'batch_cal': 's', # Square
    'batch_int': '^', # Triangle up
    'maqr': 'D',      # Diamond
    'calipso': 'X',   # X
    'batch_QRT': 'P',     # Plus
    'mpaic0.2': 'v',   # Triangle down
    'mpaic0.4': '<',   # Triangle left
    'mpaic0.6': '>',   # Triangle right
    'mpaic0.8': 'p',   # Pentagon
    'mpaic0.95': '*',  # Star
    'mpaic1.0': 'h'    # Hexagon
}
HYPER_MARKERS = {
    'nl-1_hs-32': 'o',   # Circle
    'nl-2_hs-64': 's',   # Square
    'nl-4_hs-128': '^',  # Triangle up
    'nl-8_hs-256': 'D',  # Diamond
}
HP_DIR_NAME_TO_FILE_NAME = {
    'nl-1_hs-32': 'nl1_hs32',
    'default': 'nl2_hs64',
    'nl-4_hs-128': 'nl4_hs128',
    'nl-8_hs-256': 'nl8_hs256',
}
DATASETS = [
    'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht',
    'diamonds', 'elevator', 'fusion', 'facebook'
]

def load_pickle(path: str) -> Dict[str, Any]:
    """Load a pickle that contains the save_var_names dict data."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def safe_get(d: Dict[str, Any], key: str, default=None):
    """Safely get a value from a dictionary."""
    return d.get(key, default)
 