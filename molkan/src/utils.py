import logging
import datetime
import time
from tqdm import tqdm
from pathlib import Path
import os
import sys
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, mean_absolute_error


# Initializing Logger
def init_logger(outdir, filename="log.txt", level_console="info", level_file="debug"):
    
    level_dic = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET
    }

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    fmt = logging.Formatter(
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt = "%Y%m%d-%H%M%S"
    )

    fh = logging.FileHandler(filename=Path(outdir, filename))
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
    logging.getLogger("matplotlib").setLevel(level_dic["warning"])
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

# Count parameters num
def count_param(model):
    def format_params(num):
        if num >= 1e9:
            return f"{num / 1e9:.3f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.3f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.3f}K"
        else:
            return str(num)
    total_params = format_params(sum(p.numel() for p in model.parameters()))
    trainable_params = format_params(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total_params, trainable_params

# Save and load experiments
def save_experiment(
        outdir, config, model, train_losses, train_briers, valid_losses, valid_briers, 
        test_scores, classes=None, note=None, save_model=False, save_config=False
        ):
    """
    save the experiment: config, model, metrics, and progress plot
    
    Args:
        outdir (str | Pathlike): output directory
        config (dict): configuration dictionary
        model (nn.Module): model to be saved
        train_losses (list): training losses
        valid_losses (list): valid losses
        test_scores (dict): test_scores
        classes (dict): dictionary of class names
        note (str): short note for this running if any
    
    """
    # save config
    if save_config:
        configfile = os.path.join(outdir, f'config_{note}.json')
        with open(configfile, 'w') as f:
            json.dump(config, f, sort_keys=False, indent=4)
    # save metrics
    jsonfile = os.path.join(outdir, f'loss_scores_{note}.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'train_briers': train_briers,
            'valid_losses': valid_losses,
            'valid_briers': valid_briers,
            'test_scores': test_scores,
            'classes': classes,
        }
        json.dump(data, f, sort_keys=False, indent=4)
    # save the model
    if save_model:
        save_checkpoint(model, outdir, note=note)


def save_checkpoint(model, outdir, note):
    """
    save the model checkpoint

    Args:
        experiment_name (str): name of the experiment
        model (nn.Module): model to be saved
        epoch (int): epoch number
        base_dir (str): base directory to save the experiment

    """
    model_dir = os.path.join(outdir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    if note is None:
        cpfile = os.path.join(model_dir, f"models.pt")
    else:
        cpfile = os.path.join(model_dir, f"{note}.pt")
    torch.save(model.state_dict(), cpfile)


def plot_progress(
        outdir:str, train_loss:list, valid_loss:list, num_epoch:int,
        xlabel="epoch", ylabel="loss", note=None
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
    pltdir = os.path.join(outdir, 'progress')
    os.makedirs(pltdir, exist_ok=True)
    if note is None:
        plt.savefig(os.path.join(pltdir, f"{ylabel}.png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(pltdir, f"{ylabel}_{note}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_experiments_results(
        outdir, results:list[list], xlabel:list, ylabel:list, note=None
        ):
    """ plot the results of experiments """
    bar_width = 0.35
    index = np.arange(len(results[0]))

    top_indices = np.argsort(results[0])[-3:][::-1]
    label_colors = ['black'] * len(xlabel)  # 初期はすべて黒
    top_colors = ['red', 'blue', 'green']
    for i, idx in enumerate(top_indices):
        label_colors[idx] = top_colors[i]

    plt.figure(figsize=(30, 18))
    for i, result in enumerate(results):
        plt.bar(index + i * bar_width, result, bar_width, label=ylabel[i])
    plt.xticks(index + bar_width*len(results)/2, xlabel, rotation=90)
    ax = plt.gca()  # 現在のAxesを取得
    
    for label, color in zip(ax.get_xticklabels(), label_colors):
        label.set_color(color)
    
    plt.legend()
    plt.ylabel("test scores")
    plt.title(f"Test Scores of {note}")
    plt.tight_layout()
    if note is None:
        plt.savefig(os.path.join(outdir, f"test_scores.png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(outdir, f"test_scores_{note}.png"), dpi=300, bbox_inches='tight')


# Timer related functions
def timer(start_time, logger):
    """ Measure the elapsed time """
    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time // 3600)  # hour
    elapsed_minutes = int((elapsed_time % 3600) // 60)  # min
    elapsed_seconds = int(elapsed_time % 60)  # sec
    res = f"{elapsed_hours:02}:{elapsed_minutes:02}:{elapsed_seconds:02}"
    logger.info(f"{res}")
    return res


def get_component_list(model, optimizer, loss_func, device, scheduler=None):
    """
    get the components of the model
    
    """
    components = {
    "model": model.__class__.__name__,
    "loss_func": loss_func.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "device": device.__class__.__name__,
    "scheduler": scheduler.__class__.__name__,
    }
    return components


# Loss
class BCEWithLogitsLoss():
    def __call__(self, y_pred, y_true, weight):
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, weight)

class BCELoss():
    def __call__(self, y_pred, y_true, weight):
        return torch.nn.functional.binary_cross_entropy(y_pred, y_true, weight)

class MSE():
    def __init__(self, brier:bool=False):
        self.brier = brier
    def __call__(self, y_pred, y_true, weight):
        if self.brier:
            return torch.sum(((y_pred - y_true) ** 2) * weight)
        else:
            return torch.sum(((y_pred- y_true) ** 2) * weight) / torch.sum(weight)
    

# Metrics functions
class Metrics:

    @staticmethod
    def AUROC(pred, y):
        n_labels = y.shape[1]
        auroc_list = []
        for i in range(n_labels):
            mask = ~np.isnan(y[:, i])
            try:
                auroc = roc_auc_score(y[mask, i], pred[mask, i])
            except:
                auroc = np.nan
            auroc_list.append(auroc)
        return float(np.nanmean(auroc_list))
    
    @staticmethod
    def get_confusion_matrix(pred, y):
        pred = (pred >= 0.5).astype(int)
        TP = np.sum((pred == 1) & (y == 1))
        TN = np.sum((pred == 0) & (y == 0))
        FP = np.sum((pred == 1) & (y == 0))
        FN = np.sum((pred == 0) & (y == 1))
        return TP, TN, FP, FN
    
    @staticmethod
    def accuracy(pred, y):
        pred = (pred >= 0.5).astype(int)
        try:
            return np.sum(pred == y) / np.sum(~np.isnan(y))
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
        return float(r2_score(y, pred))
    
    @staticmethod
    def RMSE(pred, y):
        return float(root_mean_squared_error(y, pred))
    
    @staticmethod
    def MAE(pred, y):
        return float(mean_absolute_error(y, pred))


# for argparse input
def parse_list_or_int(value):
    try:
        return int(value)
    except ValueError:
        return list(map(int, value.split(',')))

def parse_list_or_float(value):
    try:
        return float(value)
    except ValueError:
        return list(map(float, value.split(',')))

def parse_str_list(value):
    if "," in value:
        return value.split(',')
    else:
        return [value]