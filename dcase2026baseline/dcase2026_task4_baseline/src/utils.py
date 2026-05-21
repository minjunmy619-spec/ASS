import os
import sys
import datetime
import logging
import copy
from typing import Dict, List, NoReturn
import yaml
import importlib
import pytz
import torch

LABELS = {
    'dcase2026t4': ["AlarmClock", "BicycleBell", "Blender", "Buzzer",
                    "Clapping", "Cough", "CupboardOpenClose", "Dishes",
                    "Doorbell", "FootSteps", "HairDryer", "MechanicalFans",
                    "MusicalKeyboard", "Percussion", "Pour", "Speech",
                    "Typing", "VacuumCleaner"],
}


def ignore_warnings():
    import warnings

    # Filter out warnings containing "deprecated"
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    warnings.filterwarnings("ignore", message=".*that has Tensor Cores.*")
    warnings.filterwarnings("ignore", message=".*cudnnFinalize Descriptor Failed.*")
    warnings.filterwarnings("ignore", message=".*torch.use_deterministic_algorithms(True, warn_only=True)*")
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except ModuleNotFoundError:
        pass

logging_levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}
def logging_setup(directory, file_log_level = "INFO", console_log_level = 'DEBUG', timezone='Asia/Tokyo'):
    file_log_level = logging_levels[file_log_level]
    console_log_level = logging_levels[console_log_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # create directory, generate logfilename
    os.makedirs(directory, exist_ok=True)
    a = datetime.datetime.now().astimezone(pytz.timezone(timezone))
    logfilename='%04d%02d%02d_%02dh%02d.log'%(a.year, a.month, a.day, a.hour, a.minute)
    log_full_path = os.path.join(directory, logfilename)

    # setup log to file
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # print to console
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s (%(processName)s): %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level) # always debug level
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


def _get_by_path(obj, dotted_path):
    cur = obj
    for part in dotted_path.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)} at '{part}' for path '{dotted_path}'")
    return cur


def _deep_update(base, updates):
    base = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_config_references(obj, base_dir=None):
    if isinstance(obj, list):
        return [_resolve_config_references(x, base_dir=base_dir) for x in obj]
    if not isinstance(obj, dict):
        return obj

    ref_path = obj.get("from_training_config", obj.get("config_ref", None))
    if ref_path is not None:
        candidate = ref_path
        if not os.path.isabs(candidate) and not os.path.exists(candidate) and base_dir:
            candidate = os.path.join(base_dir, candidate)
        with open(candidate, "r") as fr:
            ref_cfg = yaml.load(fr, Loader=yaml.FullLoader)
        ref_cfg = _resolve_config_references(ref_cfg, base_dir=os.path.dirname(os.path.abspath(candidate)))
        key = obj.get("key", "lightning_module.args.model")
        value = copy.deepcopy(_get_by_path(ref_cfg, key))
        overrides = obj.get("overrides", obj.get("override", None))
        if overrides:
            if not isinstance(value, dict) or not isinstance(overrides, dict):
                raise TypeError("override(s) can only be applied when both referenced value and override are dictionaries")
            value = _deep_update(value, overrides)
        return value

    return {k: _resolve_config_references(v, base_dir=base_dir) for k, v in obj.items()}


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        cfg = yaml.load(fr, Loader=yaml.FullLoader)
    return _resolve_config_references(cfg, base_dir=os.path.dirname(os.path.abspath(config_yaml)))

def initialize_config(module_cfg, reload=False):
    if reload and module_cfg["module"] in sys.modules:
        module = importlib.reload(sys.modules[module_cfg["module"]])
    else: module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    return getattr(module, module_cfg["main"])()

def lightning_load_from_checkpoint(lightning_module_cfg, ckpt_path):
    module = importlib.import_module(lightning_module_cfg["module"])
    model_class = getattr(module, lightning_module_cfg["main"])
    model = model_class.load_from_checkpoint(ckpt_path, **lightning_module_cfg['args'])
    return model
