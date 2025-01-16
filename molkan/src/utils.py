import logging
import datetime
from tqdm import tqdm
from pathlib import Path
import os
import sys
import random
import numpy as np
import torch

def init_logger(level_console="info", level_file="debug"):
    
    level_dic = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET
    }

    log_dir = os.path.dirname(os.path.abspath(__file__)).replace("experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_name = os.path.basename(__file__).replace(".py", "txt")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    fmt = logging.Formatter(
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt = "%Y%m%d-%H%M%S"
    )

    fh = logging.FileHandler(filename=Path(log_dir, log_name))
    fh.setLevel(level_dic[level_file])
    fh.setFormatter(fmt)

    th = TqdmLoggingHandler()
    th.setLevel(level_dic[level_console])
    th.setFormatter(fmt)

    logging.basicConfig(
        level=level_dic["notset"],
        handlers=[fh, th]
    )

    logger = logging.getLogger(__name__)
    return logger

class TqdmLoggingHandler(logging.Handler):
    def __init__(self,level=logging.NOTSET):
        super().__init__(level)

    def emit(self,record):
        try:
            msg = self.format(record)
            tqdm.write(msg,file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


def to_logger(
        logger: logging.Logger, name:str='', obj=None,
        skip_keys=None, skip_hidden:bool=True
        ):
    """
    Log attributes of an object using the provided logger.

    Args:
        logger (logging.Logger): Logger instance.
        name (str): Name or header to log before object attributes.
        obj: Object whose attributes will be logged.
        skip_keys (set): Set of keys to skip when logging attributes.
        skip_hidden (bool): If True, skip attributes starting with '_'.

    """
    if skip_keys is None:
        skip_keys = set()
    logger.info(name)
    if obj is not None:
        for k, v in vars(obj).items():
            if k not in skip_keys:
                if skip_hidden and k.startswith('_'):
                    continue
                logger.info('  {0}: {1}'.format(k, v))


# Fix random seed for reproducibility
def fix_seed(seed:int=42, fix_gpu:bool=False):
    """
    Fix random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value for random number generators.
        fix_gpu (bool): If True, GPU-related randomness is also fixed. 
                        This may reduce performance but ensures reproducibility.
                        Default is False.

    Returns:
        None
    """
    # Fix seed for Python's built-in random module
    random.seed(seed)
    # Fix seed for NumPy
    np.random.seed(seed)
    # Fix seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Fix seed for PyTorch on GPU if GPU is available and requested
    if fix_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setup
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True