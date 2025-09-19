import subprocess
from typing import Dict, List, Optional
from plots.plot_metrics import plot_training_stats, compare_ece_sharpness, calibration_plot, plot_ece_sharpness, overlap_ece_sharpness
from plots.plot_utils import load_pickle
import os

def dict_to_cli_args(kwargs: Dict) -> List[str]:
	"""
	Convert a dict of key->value to a list of CLI arguments.
	Example: {"loss_fn":"batch_qr", "num_ens":2} -> ["--loss_fn","batch_qr","--num_ens","2"]
	Booleans are treated as flags when True, omitted when False or None.
	"""
	args = []
	for k, v in kwargs.items():
		if v is None:
			continue
		key = f"--{k}"
		if isinstance(v, bool):
			if v:
				args.append(key)
			# skip False
		elif isinstance(v, (list, tuple)):
			# repeat flag for each element
			for el in v:
				args.extend([key, str(el)])
		else:
			args.extend([key, str(v)])
	return args


def _query_nvidia_smi() -> Optional[str]:
	"""Return raw nvidia-smi query output or None if command not available."""
	try:
		out = subprocess.check_output([
			"nvidia-smi",
			"--query-gpu=index,memory.free",
			"--format=csv,noheader,nounits"
		], stderr=subprocess.DEVNULL)
		return out.decode("utf-8")
	except Exception:
		return None


def parse_nvidia_smi_output(output: str) -> List[dict]:
	"""
	Parse lines like "0, 12345" into list of dicts: [{"index":0,"memory.free":12345}, ...]
	"""
	res = []
	for line in output.strip().splitlines():
		if not line.strip():
			continue
		parts = [p.strip() for p in line.split(",")]
		if len(parts) < 2:
			continue
		try:
			idx = int(parts[0])
			mem = int(parts[1])
			res.append({"index": idx, "memory.free": mem})
		except ValueError:
			continue
	return res


def find_available_gpus(min_free_mb: int = 1000) -> List[int]:
	"""
	Return list of GPU indices that have at least min_free_mb free memory.
	If nvidia-smi is not available, returns empty list.
	"""
	out = _query_nvidia_smi()
	if out is None:
		return []
	infos = parse_nvidia_smi_output(out)
	available = [info["index"] for info in infos if info["memory.free"] >= min_free_mb]
	return available


def pick_free_gpu(min_free_mb: int = 1000) -> Optional[int]:
	"""
	Pick and return the first GPU index with at least min_free_mb free memory.
	Returns None if none available.
	"""
	avail = find_available_gpus(min_free_mb=min_free_mb)
	return avail[0] if avail else None

_last_gpu_idx = -1  # global variable to track last used GPU index

def pick_free_gpu_round_robin(min_free_mb: int = 1000) -> Optional[int]:
	"""
	Pick and return a GPU index with at least min_free_mb free memory in round-robin order.
	Returns None if none available.
	"""
	global _last_gpu_idx
	available = find_available_gpus(min_free_mb=min_free_mb)
	if not available:
		return None
	available_sorted = sorted(available)
	if _last_gpu_idx not in available_sorted:
		# start from the first available if last used is not available
		_next_idx = 0
	else:
		_next_idx = (available_sorted.index(_last_gpu_idx) + 1) % len(available_sorted)
	selected = available_sorted[_next_idx]
	_last_gpu_idx = selected
	return selected

RESULT_BASE = os.path.join(os.environ["SCRATCH"], "results")

BASIC_INPUTS = {
    "save_dir": RESULT_BASE,
    "min_thres": 0.001,
    "max_thres": 0.15,
    "num_thres": 150,
	"num_ep": 1000,
}

DEFAULT_VALUE = {
    "num_ens": 1,
    "nl": 2,
    "hs": 64,
    "batch_norm": 0,
    "layer_norm": 0,
    "dropout": 0.0,
    "activation": "relu",
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
    "nl": [1, 2],
    "hs": [32, 64],
    "residual": [1],
    "seed": [0, 1, 2],
    "loss": ["maqr", "batch_qr"],
	"num_ep": [300],
    # "loss": ["batch_qr"]
}

FULL_HYPERPARAMS = {
    "skip_existing": [1],
    "data": ["boston", "concrete", "energy", "kin8nm", "naval", "power", "wine", "yacht", "protein", 
			 "diamonds", "facebook", "elevators", "fusion"],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8, 1, 2, 4],
    "hs": [256, 32, 64, 128],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4],
    "loss": ["maqr", "batch_qr", "batch_int", "batch_cal"],
}

NEWDATA_HYPERPARAMS = {
    "skip_existing": [1],
    "data": ["diamonds", "facebook", "elevators", "fusion"],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8, 1, 2, 4],
    "hs": [256, 32, 64, 128],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4],
    "loss": ["maqr", "batch_qr", "batch_int", "batch_cal"],
}

NL_HS_COMBINATIONS = [
	(1, 32),
	(2, 64),
	(4, 128),
	(8, 256),
]

HYPERPARAMS = {
	"TEST": TEST_HYPERPARAMS,
	"FULL": FULL_HYPERPARAMS,
	"NEWDATA": NEWDATA_HYPERPARAMS,
}

def get_one_hot_param(inputs: dict, default_value_dict: dict) -> Optional[str]:
    """
    Return the one-hot parameter if the job configuration is one-hot, else None.
    Special case: if both 'nl' and 'hs' are non-defaults, check if the combination is valid.
    """
    # collect non-default keys
    non_default_keys = [
        k for k, v in inputs.items()
        if k in default_value_dict and v != default_value_dict[k]
    ]

    if len(non_default_keys) == 0:
        return "default"

    if len(non_default_keys) == 1:
        k = non_default_keys[0]
        return f"{k}-{inputs[k]}"

    if len(non_default_keys) == 2 and set(non_default_keys) == {"nl", "hs"}:
        nl_val, hs_val = inputs["nl"], inputs["hs"]
        if (nl_val, hs_val) in NL_HS_COMBINATIONS:  # You’d need to implement this
            return f"nl-{nl_val}_hs-{hs_val}"
        return None

    return None


def generate_plots_for_pickle(pkl_path: str, out_parent_dir: str):
    base = os.path.basename(pkl_path)
    name = base[:-4] if base.lower().endswith(".pkl") else base
    outdir = os.path.join(out_parent_dir, name)
    os.makedirs(outdir, exist_ok=True)

    data = load_pickle(pkl_path)
    plot_training_stats(data, outpath=os.path.join(outdir, "training_stats.png"))
    # compare_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness_comparison.png"))
    # calibration_plot(data, outpath=os.path.join(outdir, "calibration_plot.png"))
    plot_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness_curve.png"))

def generate_overlap_plot(current_pkl_path: str, current_baseline_name: str, baseline_names: List[str], out_parent_dir: str, file_name: str):
	os.makedirs(out_parent_dir, exist_ok=True)

	pkl_paths = [current_pkl_path.replace(current_baseline_name, bname) for bname in baseline_names]
	if not all(os.path.exists(p) for p in pkl_paths):
		return False
	datas = [load_pickle(p) for p in pkl_paths]
	overlap_ece_sharpness(datas, baseline_names, outpath=os.path.join(out_parent_dir, file_name))
	return True

def fix_inputs(inputs: Dict) -> Dict:
	"""
	Fix and return a new inputs dict with consistent settings.
	For example, if num_ens=1, set boot=0.
	"""
	new_inputs = inputs.copy()
	if inputs["num_ens"] == 1:
		new_inputs["boot"] = 0
	if inputs["loss"] == "maqr":
		new_inputs["num_ep"] = inputs["num_ep"] // 40
		new_inputs["wait"] = 5
	else:
		new_inputs["wait"] = inputs["num_ep"] // 5
	
	return new_inputs

def invalid_inputs(inputs: Dict) -> bool:
	"""
	Return True if the job configuration should be skipped based on constraints.
	"""
	if inputs["batch_norm"] == 1 and inputs["layer_norm"] == 1:
		return True
	if inputs["num_ens"] == 1 and inputs["boot"] == 1:
		return True
	if (inputs["nl"], inputs["hs"]) not in NL_HS_COMBINATIONS:
		return True
	return False