import pickle
from typing import Any, Dict

def load_pickle(path: str) -> Dict[str, Any]:
    """Load a pickle that contains the save_var_names dict data."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def safe_get(d: Dict[str, Any], key: str, default=None):
    """Safely get a value from a dictionary."""
    return d.get(key, default)
 