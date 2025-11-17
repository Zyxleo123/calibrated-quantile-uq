import subprocess
from typing import Dict, List, Optional
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


def find_available_gpus(min_free_mb: int = 1000, choices=None) -> List[int]:
	"""
	Return list of GPU indices that have at least min_free_mb free memory.
	If nvidia-smi is not available, returns empty list.
	"""
	out = _query_nvidia_smi()
	if out is None:
		return []
	infos = parse_nvidia_smi_output(out)
	available = [info["index"] for info in infos if info["memory.free"] >= min_free_mb and (choices is None or info["index"] in choices)]
	return available


_last_gpu_idx = -1
def pick_free_gpu_round_robin(min_free_mb: int = 1000, choices=None) -> Optional[int]:
	"""
	Pick and return a GPU index with at least min_free_mb free memory in round-robin order.
	Returns None if none available.
	"""
	global _last_gpu_idx
	available = find_available_gpus(min_free_mb=min_free_mb, choices=choices)
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
# RESULT_BASE = os.path.join("~/results")

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
    "data": ["boston", "yacht"],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [1],
    "hs": [32],
    "residual": [1],
    "seed": [0],
    "loss": ["batch_int", "batch_qr", "batch_int"],
	"num_ep": [5],
}

# gpu24 done
MAQR_QR_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "loss": ["maqr", "batch_qr"],
    "data": ["protein", "boston", "concrete", "energy", "facebook", "kin8nm", "naval", "power", "elevator", "wine", "yacht",
			 "diamonds", "fusion"],
}

# gpu23 done
INT_CAL_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "loss": ["batch_int", "batch_cal"],
    "data": ["protein", "boston", "concrete", "energy", "facebook", "kin8nm", "naval", "power", "elevator", "wine", "yacht",
			 "diamonds", "fusion"],
}

# gpu21 done
QRT_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "loss": ["batch_QRT"],
    "data": ["protein", "boston", "concrete", "energy", "facebook", "kin8nm", "naval", "power", "elevator", "wine", "yacht",
			 "diamonds", "fusion"],
}
# QRT_QRTC_HYPERPARAMS = {
#     "skip_existing": [1],
#     "lr": [1e-3],
#     "bs": [64],
#     "batch_norm": [0],
#     "layer_norm": [0],
#     "dropout": [0.0],
#     "num_ens": [1],
#     "boot": [0],
#     "nl": [8],
#     "hs": [256],
#     "residual": [1],
#     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     "loss": ["batch_QRT", "batch_QRTC"],
#     "data": ["protein", "boston", "concrete", "energy", "facebook", "kin8nm", "naval", "power", "elevator", "wine", "yacht",
# 			 "diamonds", "fusion"],
# }

# gpu31
CALIPSO_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "loss": ["calipso"],
    "data": ["protein", "boston", "concrete", "energy", "facebook", "kin8nm", "naval", "power", "elevator", "wine", "yacht",
			 "diamonds", "fusion"],
}

# gpu22 and gpu31 (now running)
MPAICFA_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "loss": ["mpaic"],
	"alpha": [0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
    # "data": ["protein", "boston", "concrete", "energy", "facebook", "kin8nm", "naval", "power", "elevator", "wine", "yacht",
	# 		 "diamonds", "fusion"],
	"data": ['facebook']
}

MPAICFU_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "loss": ["mpaic"],
	"alpha": [0.6, 0.2, 0.4, 0.8, 0.95, 1.0],
    # "data": ["protein", "boston", "concrete", "energy", "facebook", "kin8nm", "naval", "power", "elevator", "wine", "yacht",
	# 		 "diamonds", "fusion"],
	"data": ['fusion']
}

OPTIMIZER_OPTIONS = ["adam", "sgd", "adamw", "rmsprop"]

OPTIM_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [1e-3],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2],
	"optimizer": OPTIMIZER_OPTIONS,
    "loss": ["batch_qr"],
	"data": ["concrete", "kin8nm", "protein"],
}

LR_HYPERPARAMS = {
    "skip_existing": [1],
    "lr": [5e-4, 1e-3, 5e-3, 1e-2],
    "bs": [64],
    "batch_norm": [0],
    "layer_norm": [0],
    "dropout": [0.0],
    "num_ens": [1],
    "boot": [0],
    "nl": [8],
    "hs": [256],
    "residual": [1],
    "seed": [0, 1, 2],
    "optimizer": ["adam"],
    "loss": ["batch_qr"],
    "data": ["concrete", "kin8nm", "protein"],
}


NL_HS_COMBINATIONS = [
	(1, 32),
	(2, 64),
	(4, 128),
	(8, 256),
]

HYPERPARAMS = {
	"TEST": TEST_HYPERPARAMS,
	"MAQR_QR": MAQR_QR_HYPERPARAMS,
	"INT_CAL": INT_CAL_HYPERPARAMS,
	"MPAICFA": MPAICFA_HYPERPARAMS,
	"MPAICFU": MPAICFU_HYPERPARAMS,
	"QRT": QRT_HYPERPARAMS,
	"CALIPSO": CALIPSO_HYPERPARAMS,
    "OPTIM": OPTIM_HYPERPARAMS,
    "LR": LR_HYPERPARAMS,
}

def get_job_name(inputs: Dict, run_type: str) -> str:
    if run_type == 'optim':
        job_name = f"opt-{inputs['optimizer']}"
    elif run_type == 'lr':
        job_name = f"lr-{inputs['lr']}"
    else:
        job_name = f"nl-{inputs['nl']}_hs-{inputs['hs']}"
    return job_name

def fix_inputs(inputs: Dict) -> Dict:
	"""
	Fix and return a new inputs dict with consistent settings.
	For example, if num_ens=1, set boot=0.
	"""
	new_inputs = inputs.copy()
	if inputs["num_ens"] == 1:
		new_inputs["boot"] = 0
	if inputs["loss"] == "maqr":
		new_inputs["num_ep"] = 25
		new_inputs["wait"] = 200
	elif inputs["loss"] == "calipso":
		new_inputs["num_ep"] = 100
		new_inputs["wait"] = 200
	else:
		new_inputs["num_ep"] = 1000
		new_inputs["wait"] = 200
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