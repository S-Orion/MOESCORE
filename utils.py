import json
import torch
import sys
from collections.abc import Mapping, Sequence
from transformers.tokenization_utils_base import BatchEncoding

def load_config(config_path = "config.json"):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def json_dump(json_path, data):
    with open(json_path, "w", encoding="utf-8") as f:
         json.dump(data, f, ensure_ascii=False, indent=4)
    return

def move_to_device(obj, device):
    """
    Recursively move tensors (and common container objects that hold tensors)
    to the target device.

    Parameters
    ----------
    obj : Any
        Arbitrary object that may contain tensors (Tensor, dict-like, list-like,
        or transformers.BatchEncoding). Non-tensor scalars/strings are left unchanged.
    device : torch.device or str
        Target device (e.g., "cuda:0", "cpu").

    Returns
    -------
    Any
        Same structure as `obj`, but with all tensors placed on `device`.
    """
    # 1) Single tensor → move directly
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    # 2) HuggingFace BatchEncoding supports .to(), handle it explicitly
    if isinstance(obj, BatchEncoding):
        return obj.to(device)

    # 3) Mapping types (dict, UserDict, etc.) → recurse on each value
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    # 4) Sequence types (list, tuple, etc.) → preserve container type
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(move_to_device(v, device) for v in obj)

    # 5) Anything else (int, float, str, Path, ...) → leave untouched
    return obj