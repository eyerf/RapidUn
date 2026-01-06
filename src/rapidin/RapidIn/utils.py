import sys
import json
import logging
import collections.abc
from pathlib import Path
from datetime import datetime as dt

from tqdm import tqdm


tqdm_dict = {}


def _redact_sensitive_text(s: str) -> str:
    """
    Best-effort redaction for identity-bearing absolute paths in saved artifacts/logs.
    This function is intentionally conservative and only targets common home directories.
    """
    if not isinstance(s, str):
        return s
    # Linux home: /home/<user>/...
    s = __import__("re").sub(r"/home/[^/\s]+", "/home/USER", s)
    # macOS home: /Users/<user>/...
    s = __import__("re").sub(r"/Users/[^/\s]+", "/Users/USER", s)
    return s


def _redact_obj(obj):
    """
    Recursively redact sensitive strings inside nested dict/list structures.
    """
    if isinstance(obj, dict):
        return {k: _redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_redact_obj(v) for v in obj)
    if isinstance(obj, str):
        return _redact_sensitive_text(obj)
    return obj


def save_json(
    json_obj,
    json_path,
    append_if_exists=False,
    overwrite_if_exists=False,
    unique_fn_if_exists=True,
):
    """Saves a JSON file.

    Args:
        json_obj: JSON-serializable object.
        json_path: Path or str. Target path including the file name.
        append_if_exists: If True and file exists, merge dict keys into existing JSON.
        overwrite_if_exists: If True, overwrite any existing target file.
        unique_fn_if_exists: If True and file exists, append a timestamp to the filename.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    # Anonymize identity-bearing absolute paths inside json_obj (best-effort).
    json_obj = _redact_obj(json_obj)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f"{json_path.stem}_{time_str}{json_path.suffix}"

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, "w+") as fout:
            json.dump(json_obj, fout, indent=None)
        return json_path

    if append_if_exists:
        if json_path.exists():
            with open(json_path, "r") as fin:
                read_file = json.load(fin)
            # Redact what we read as well (in case it already contains sensitive paths)
            read_file = _redact_obj(read_file)

            if isinstance(read_file, dict) and isinstance(json_obj, dict):
                read_file.update(json_obj)
            else:
                # Fallback: overwrite if the existing content is not a dict
                read_file = json_obj

            with open(json_path, "w+") as fout:
                json.dump(read_file, fout, indent=None)
            return json_path

    with open(json_path, "w+") as fout:
        json.dump(json_obj, fout, indent=None)

    return json_path


def load_json(json_path, key2int=True):
    def convert_key_to_int(d):
        new_dict = {}
        for k, v in d.items():
            if isinstance(k, str) and k.isnumeric():
                k = int(k)
            if isinstance(v, dict):
                v = convert_key_to_int(v)
            new_dict[k] = v
        return new_dict

    with open(json_path, "r") as f:
        result = json.load(f)

    if key2int:
        result = convert_key_to_int(result)

    return result


def display_progress(
    text,
    current_step,
    last_step,
    enabled=True,
    fix_zero_start=True,
    new_line=False,
    run_time=None,
    cur_time=None,
):
    """Draw a progress indicator on the screen with a preceding text label.

    Args:
        text: Label shown in the progress bar.
        current_step: Current step index.
        last_step: Total number of steps.
        enabled: If False, do nothing (silent mode).
        fix_zero_start: If True, shift displayed steps by +1 for 0-based loops.
    """
    if not enabled:
        return

    if fix_zero_start:
        current_step = current_step + 1

    if text not in tqdm_dict:
        tqdm_dict[text] = tqdm(total=last_step, desc=text)

    tqdm_dict[text].n = current_step
    tqdm_dict[text].refresh()


def init_logging(filename=None):
    """Initialize logging to stdout and optionally to a file.

    Args:
        filename: If provided, logs will be written to this file in addition to stdout.
    """
    log_lvl = logging.INFO
    log_format = "%(asctime)s: %(message)s"
    if filename:
        logging.basicConfig(
            handlers=[logging.FileHandler(filename), logging.StreamHandler(sys.stdout)],
            level=log_lvl,
            format=log_format,
        )
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl, format=log_format)


def get_default_config():
    """Return a default config object (as a nested dict)."""
    config = {
        "data": {
            "train_data_path": None,
            "test_data_path": None,
            "begin_id": None,
            "end_id": None,
            "test_begin_id": None,
            "test_end_id": None,
        },
        "influence": {
            "outdir": "outdir",
            "seed": 42,
            "IF": {
                "recursion_depth": 5,
                "r_averaging": 3,
                "scale": 50000,
            },
            "cal_words_infl": False,
            "grads_path": None,
            "load_from_grads_path": False,
            "save_to_grads_path": False,
            "delete_model": False,
            "n_threads": 1,
            "RapidGrad": {
                "enable": False,
                "RapidGrad_M": 1,
                "RapidGrad_K": 65536,
                "shuffle_lambda": 20,
                "multi_k_save_path_list": None,  # assigned by program only
            },
            "deepspeed": {
                "enable": False,
                "config_path": None,
            },
            "offload_test_grad": True,
            "offload_train_grad": False,
            "calculate_infl_in_gpu": False,
            "skip_test": False,
            "skip_influence": False,
            "infl_method": "TracIn",  # TracIn, IF. (default: TracIn)
            "top_k": 1000,
        },
        "model": {
            "model_path": None,
            "lora_path": None,
            "max_length": None,
            "load_in_4bit": False,
        },
    }
    return config


def sanity_check(config):
    if config.influence.RapidGrad.enable and isinstance(config.influence.RapidGrad.RapidGrad_K, list):
        if (config.influence.skip_test is False) or (config.influence.skip_influence is False):
            print("RapidGrad_K is a list; setting `skip_test` and `skip_influence` to True.")
            config.influence.skip_test = True
            config.influence.skip_influence = True

        if (config.influence.save_to_grads_path is False) or (config.influence.grads_path is None):
            raise AssertionError("RapidGrad_K is a list; set `save_to_grads_path=True` and assign `grads_path`.")


class Struct:
    """Recursively build objects with attribute access from nested dictionaries."""

    def __init__(self, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return "{%s}" % str(", ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items()))


def get_config(config_path):
    """Load and return a config file."""
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = get_default_config()
    config = update(config, json.load(open(config_path)))
    config = Struct(config)
    sanity_check(config)
    return config