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

def is_one_hot_job(inputs, default_value_dict) -> bool:
	"""
	Return True if the job configuration should be skipped based on constraints.
	"""
	num_hot = 0
	for key, val in inputs.items():
		if key in default_value_dict and val != default_value_dict[key]:
			num_hot += 1
	if num_hot > 1:
		return True
	return False

def generate_plots_for_pickle(pkl_path: str, out_parent_dir: str):
    base = os.path.basename(pkl_path)
    name = base[:-4] if base.lower().endswith(".pkl") else base
    outdir = os.path.join(out_parent_dir, name)
    os.makedirs(outdir, exist_ok=True)

    data = load_pickle(pkl_path)
    plot_training_stats(data, outpath=os.path.join(outdir, "training_stats.png"))
    compare_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness_comparison.png"))
    calibration_plot(data, outpath=os.path.join(outdir, "calibration_plot.png"))
    plot_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness_curve.png"))

def generate_overlap_plot(current_pkl_path: str, current_baseline_name: str, baseline_names: List[str], out_parent_dir: str):
	base = os.path.basename(current_pkl_path)
	name = base[:-4] if base.lower().endswith(".pkl") else base
	outdir = os.path.join(out_parent_dir, name)
	os.makedirs(outdir, exist_ok=True)

	pkl_paths = [current_pkl_path.replace(current_baseline_name, bname) for bname in baseline_names]
	if not all(os.path.exists(p) for p in pkl_paths):
		return False
	datas = [load_pickle(p) for p in pkl_paths]
	overlap_ece_sharpness(datas, baseline_names, outpath=os.path.join(outdir, "overlap_ece_sharpness.png"))
	return True