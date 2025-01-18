import logging
import datetime
from tqdm import tqdm
from pathlib import Path
import os
import sys
import random
import json
import matplotlib.pyplot as plt 
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, mean_absolute_error


# Initializing Logger
def init_logger(outdir, level_console="info", level_file="debug"):
    
    level_dic = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET
    }

    log_name = "log_" + os.path.basename(__file__).replace(".py", "txt")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    fmt = logging.Formatter(
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt = "%Y%m%d-%H%M%S"
    )

    fh = logging.FileHandler(filename=Path(outdir, log_name))
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

# Save and load experiments
def save_experiment(
        outdir, config, model, train_losses, valid_losses,
        metrics, classes, note=None, save_model=False
        ):
    """
    save the experiment: config, model, metrics, and progress plot
    
    Args:
        outdir (str | Pathlike): output directory
        config (dict): configuration dictionary
        model (nn.Module): model to be saved
        train_losses (list): training losses
        test_losses (list): test losses
        accuracies (list): accuracies
        classes (dict): dictionary of class names
        note (str): short note for this running if any
    
    """
    # save config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    # save metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'metrics': metrics,
            'classes': classes,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    # plot progress
    plot_progress(
        outdir, train_losses, valid_losses, config["num_epochs"], note=note
        )
    # save the model
    if save_model:
        save_checkpoint(model, "final", outdir, note=note)


def save_checkpoint(model, epoch, outdir, note):
    """
    save the model checkpoint

    Args:
        experiment_name (str): name of the experiment
        model (nn.Module): model to be saved
        epoch (int): epoch number
        base_dir (str): base directory to save the experiment

    """
    if note is None:
        cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    else:
        cpfile = os.path.join(outdir, f"model_{epoch}_{note}.pt")
    torch.save(model.state_dict(), cpfile)


def plot_progress(
        outdir:str, train_loss:list, valid_loss:list, num_epoch:int,
        base_dir:str="experiments", xlabel="epoch", ylabel="loss", note=None
        ):
    """ plot learning progress """
    epochs = list(range(1, num_epoch + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14
    ax.plot(epochs, train_loss, c='navy', label='train')
    ax.plot(epochs, valid_loss, c='darkgoldenrod', label='valid')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    if note is None:
        plt.savefig(outdir + f'/progress_{ylabel}.tif', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(outdir + f'/progress_{ylabel}_{note}.tif', dpi=300, bbox_inches='tight')


# Timer related functions
def timer(start_time):
    """ Measure the elapsed time """
    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time // 3600)  # hour
    elapsed_minutes = int((elapsed_time % 3600) // 60)  # min
    elapsed_seconds = int(elapsed_time % 60)  # sec
    res = f"{elapsed_hours:02}:{elapsed_minutes:02}:{elapsed_seconds:02}"
    print(f"Elapsed Time: {res}")
    return res


def get_component_list(model, optimizer, loss_func, device, scheduler=None):
    """
    get the components of the model
    
    """
    components = {
    "model": model.__class__.__name__,
    "loss_func": loss_func.__class__.__name__ if isinstance(loss_func, torch.nn.Module) else loss_func.__name__,
    "optimizer": optimizer.__class__.__name__,
    "device": device.__class__.__name__,
    "scheduler": scheduler.__class__.__name__,
    }
    return components


# Metrics functions
class Metrics:

    @staticmethod
    def AUROC(pred, y):
        try:
            return roc_auc_score(y, pred)
        except:
            return None
    
    @staticmethod
    def get_confusion_matrix(pred, y):
        pred = (pred >= 0.5).astype(int)
        TP = np.sum((pred == 1) and (y == 1))
        TN = np.sum((pred == 0) and (y == 0))
        FP = np.sum((pred == 1) and (y == 0))
        FN = np.sum((pred == 0) and (y == 1))
        return TP, TN, FP, FN
    
    @staticmethod
    def accracy(pred, y):
        pred = (pred >= 0.5).astype(int)
        try:
            return np.sum(pred == y) / len(pred)
        except:
            return None

    @staticmethod    
    def sensitivity(pred, y):
        TP, TN, FP, FN = Metrics.get_confusion_matrix(pred, y)
        try:
            return TP / (TP + FN)
        except:
            return None

    @staticmethod   
    def specificity(pred, y):
        TP, TN, FP, FN = Metrics.get_confusion_matrix(pred, y)
        try:
            return TN / (TN + FP)
        except:
            return None

    @staticmethod    
    def precision(pred, y):
        TP, TN, FP, FN = Metrics.get_confusion_matrix(pred, y)
        try:
            return TP / (TP + FP)
        except:
            return None

    @staticmethod    
    def NPV(pred, y):
        TP, TN, FP, FN = Metrics.get_confusion_matrix(pred, y)
        try:
            return TN / (TN + FN)
        except:
            return None

    @staticmethod    
    def F1(pred, y):
        sens = Metrics.sensitivity(pred, y)
        prec = Metrics.precision(pred, y)
        try:
            return 2 * sens * prec / (sens + prec)
        except:
            return None
    
    @staticmethod
    def MCC(pred, y):
        TP, TN, FP, FN = Metrics.get_confusion_matrix(pred, y)
        try:
            return ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        except:
            return None

    @staticmethod
    def R2(pred, y):
        return r2_score(y, pred)
    
    @staticmethod
    def RMSE(pred, y):
        return root_mean_squared_error(y, pred)
    
    @staticmethod
    def MAE(pred, y):
        return mean_absolute_error(y, pred)


# for argparse input
def parse_list_or_int(value):
    try:
        return int(value)
    except ValueError:
        return list(map(int, value.split(', ')))

def parse_list_or_float(value):
    try:
        return float(value)
    except ValueError:
        return list(map(float, value.split(', ')))

def parse_str_list(value):
    return value.split(', ')