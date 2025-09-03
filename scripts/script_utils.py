import subprocess
from typing import Dict, List, Optional


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

def get_save_file_name(args) -> str:
    args["boot"] = bool(args["boot"])
    args["residual"] = bool(args["residual"])
    args["batch_norm"] = bool(args["batch_norm"])
    args["layer_norm"] = bool(args["layer_norm"])
    print(args)
    save_file_name = "{}/{}_loss{}_ens{}_boot{}_res{}_ln{}_bn{}_dr{}_lr{}_bs{}.pkl".format(
        args["save_dir"],
        args["data"],
        args["loss"],
        args["num_ens"],
        args["boot"],
        args["seed"],
        args["min_thres"],
        args["max_thres"],
        args["residual"],
        args["layer_norm"],
        args["batch_norm"],
        args["dropout"],
		args["lr"],
		args["bs"]
    )
    return save_file_name