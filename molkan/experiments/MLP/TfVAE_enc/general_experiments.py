import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import pickle

import sys
project_path = os.path.abspath(__file__).split("/experiments")[0]
sys.path.append(project_path)

# original packages in src
from src import utils
from src import data_handler as dh
from src.trainer import Trainer, test_func
""" === change based on the model you want to use === """
from src.models import *


""" === Config Setup: change based on your experiment === """

# Example usage for list input:
# --batch_size 32, 64, 128
# --lr 0.001, 0.01, 0.1
# --metrics accuracy, AUROC, sensitivity

parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument("--datasets", type=str, help="datasets name you want to use")
parser.add_argument("--label_columns", type=utils.parse_str_list, help="label columns names")
parser.add_argument("--results_dirname", type=str, help="results directory name")
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--train', type=bool, default=True, help='whether to train or not')
parser.add_argument("--mode", type=str, default="classification", help="classification or regression")
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument("--layer", type=int, help="Number of layer(s)")
parser.add_argument('--batch_size', type=utils.parse_list_or_int, default=128, help="batch size (int or list(int))")
parser.add_argument('--lr', type=utils.parse_list_or_float, default=0.001, help="learning rate (float or list(float))")
parser.add_argument('--lambda', type=utils.parse_list_or_float, default=0.0, help="lambda for regularization")
parser.add_argument("--scheduler_free", type=str, help="whether use scheduler_free or not")
parser.add_argument('--metrics', type=utils.parse_str_list, help="metrics to evaluate (list(str))")
parser.add_argument('--num_workers', type=int, default=16, help='number of workers for dataloader')

args = parser.parse_args() # Namespace object

# convert to config dictionary
cfg = vars(args)

# fix seed
utils.fix_seed(seed=cfg["seed"], fix_gpu=True)

# other setup

dataset_dict = {"BACE": "bace",
                                "ClinTox": "clintox_M",
                                "CYP2C9": "cyp2c9_inhib",
                                "CYP3A4": "cyp3a4_inhib",
                                "hERG": "herg_karim",
                                "LD50": "ld50",
                                "SIDER": "sider",
                                "Tox21": "tox21_M",
                                "ToxCast": "toxcast_M"}

cfg["outdir"] = os.path.join(os.path.dirname(__file__).replace("experiments", "results_and_logs"), cfg["datasets"], cfg["results_dirname"]) # for output
os.makedirs(cfg["outdir"], exist_ok=True)
if isinstance(cfg["batch_size"], int):
    cfg["batch_size"] = [cfg["batch_size"]]
if isinstance(cfg["lr"], float):
    cfg["lr"] = [cfg["lr"]]
cfg["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
if cfg["scheduler_free"] == "True":
    cfg["scheduler_free"] = True
elif cfg["scheduler_free"] == "False":
    cfg["scheduler_free"] = False
else:
    raise ValueError("Invalid scheduler_free argument. True or False ?")
for f in cfg["metrics"]:
    if not hasattr(utils.Metrics, f):
        raise AttributeError(f"Invalid metric: {f}")


""" === Data Preparation: change based on your experiments === """
def prepare_data(logger):
    """
    Load data and prepare data loaders.
    Place the processed data in the data directory or load the data from the path specified by the argument.
    It is also convenient to prepare for loading data for inference.
    """

    logger.info("=== data preparation start ===")

    """ ↓↓↓ LOAD DATA ↓↓↓ """

    data_name = dataset_dict[cfg["datasets"]]
    x = np.load(project_path + f"/data/TfVAE_repr/{data_name}_mu.npy")
    y = pd.read_csv(project_path + f"/data/tox_csv/{data_name}.csv", index_col=0)
    y = np.array(y[cfg["label_columns"]])
    dset = dh.prep_dataset(x, y)
    transform = dh.array_to_tensors_flaot()
    if len(cfg["label_columns"]) == 1:
        train_set, valid_set, test_set = dh.split_dataset_stratified(dset, [0.8, 0.1, 0.1], shuffle=True, transform=transform)
    else:
        train_set, valid_set, test_set = dh.split_dataset(dset, [0.8, 0.1, 0.1], shuffle=True, transform=transform)

    """ ↑↑↑ LOAD DATA ↑↑↑ """

    logger.info(f"num_train_data: {len(train_set)}, num_valid_data: {len(valid_set)}, num_test_data: {len(test_set)}")

    train_loaders = []
    valid_loaders = []
    test_loaders = []
    for bs in cfg['batch_size']:
        train_loader = dh.prep_dataloader(
            train_set, bs, shuffle=True, num_workers=cfg["num_workers"], pin_memory=True
            )
        valid_loader = dh.prep_dataloader(
            valid_set, bs, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
            )
        test_loader = dh.prep_dataloader(
            test_set, bs, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
            )
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)
    return train_loaders, valid_loaders, test_loaders


""" === Model Preparation: change based on your experiments === """
def prepare_model(logger, lr, l):
    """
    Prepare model, loss, optimizer, and scheduler.
    Use if statements as needed to control with arguments.
    """
    logger.info("=== model set preparation start ===")
    
    if cfg["layer"] == 1:
        model = MLP_predictor([512, 1], cfg["mode"]) # change this to your model
    elif cfg["layer"] == 2:
        model = MLP_predictor([512, 64, 1], cfg["mode"])
    elif cfg["layer"] == 3:
        model = MLP_predictor([512, 128, 16, 1], cfg["mode"])
    else:
        raise ValueError("invalid num_layer, must be 1 to 3")
    model.to(cfg["device"])

    if cfg["mode"] == "classification":
        loss_func = nn.BCELoss() # change this to your loss function
    elif cfg["mode"] == "regression":
        loss_func = utils.RMSE()
    else:
        raise NameError("invalid mode, must be classification or regression")
    
    if cfg["scheduler_free"]:
        optimizer = RAdamScheduleFree(model.parameters(), lr=lr, weight_decay=l)
    else:    
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l) # change this to your optimizer

    scheduler = None # change this to your scheduler

    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"loss_func: {loss_func.__class__.__name__}")
    logger.info(f"optimizer: {optimizer.__class__.__name__}")
    logger.info(f"scheduler: {scheduler.__class__.__name__}")

    return model, loss_func, optimizer, scheduler


""" === train func === """
def fit(model, train_loader, valid_loader, loss_func, metrics, optimizer, scheduler, logger, scheduler_free=False, note=None):
    """
    Perform training

    Args:
        model (nn.Module): The model
        train_loader (DataLoader): DataLoader for training
        test_loader (DataLoader): DataLoader for testing
        loss_func (torch.nn or in-house function): Loss function
        optimizer (torch.optim or in-house optimizer): Optimization method
        scheduler (torch.optim.lr_scheduler or in-house scheduler): Scheduler

    Returns:
        nn.Module: The trained model
        list: List of training losses
        list: List of validation losses
    """

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_func=loss_func, metrics=metrics,
        outdir=cfg["outdir"], device=cfg["device"], logger=logger, scheduler=scheduler, scheduler_free=cfg["scheduler_free"]
        )
    train_loss, valid_loss, valid_metrics, best_epoch = trainer.train(
        train_loader, valid_loader, num_epochs=cfg["num_epochs"],
        note=note, earlystopping_patience=30
        )
    return train_loss, valid_loss, valid_metrics, best_epoch


""" === test func === """
def test(model, test_loader, loss_func, metrics, logger):
    """
    Perform testing

    Args:
        model (nn.Module): The model
        test_loader (DataLoader): DataLoader for testing
        loss_func (torch.nn or in-house function): Loss function

    Returns:
        dict: Dictionary of test metrics
    """

    logger.info("=== test start ===")
    scores = test_func(model, test_loader, metrics, cfg["device"])
    for i, score in enumerate(scores):
        logger.info(f"Test {metrics[i]}: {score:.2e}")
    return scores


""" === Main === """
def main():
    if args.train:
        # training mode
        logger = utils.init_logger(cfg["outdir"])

        # 1. data prep
        train_loaders, valid_loaders, test_loaders = prepare_data(logger)
        cfg["num_training_data"] = len(train_loaders[0].dataset)
        cfg["num_valid_data"] = len(valid_loaders[0].dataset)
        cfg["num_test_data"] = len(test_loaders[0].dataset)

        # 2. training
        train_losses = {}
        valid_losses = {}
        valid_metrics_all = {}
        test_scores_all = {}
        for i, (train_loader, valid_loader, test_loader) in enumerate(zip(train_loaders, valid_loaders, test_loaders)):
            for lr in cfg["lr"]:
                for l in cfg["lambda"]:
                    logger.info(f" ===== experiments with batch size: {cfg["batch_size"][i]}, lr: {lr}, lambda: {l} ===== ")
                    start = time.time() # for time stamp
                    model, loss_func, optimizer, scheduler = prepare_model(logger, lr, l)
                    exp_note = f"bs_{cfg["batch_size"][i]}_lr_{lr}_lambda_{l}"
                    train_loss, valid_loss, valid_metrics, best_epoch  = fit(
                        model, train_loader, valid_loader, loss_func, cfg["metrics"], 
                        optimizer, scheduler, logger, note=exp_note
                        )
                    model.load_state_dict(torch.load(f"{cfg['outdir']}/models/model_best_{exp_note}.pt"))
                    test_scores = test(model, test_loader, loss_func, cfg["metrics"], logger)
                    train_losses[exp_note] = train_loss
                    valid_losses[exp_note] = valid_loss
                    valid_metrics_all[exp_note] = valid_metrics
                    test_scores_all[exp_note] = test_scores
                    utils.plot_progress(cfg["outdir"], train_loss, valid_loss, len(train_loss), note=exp_note)
                    elapsed_time = utils.timer(start) # for time stamp
                    logger.info(f" === best_epoch: {best_epoch}, elapsed_time: {elapsed_time} === ")
                    logger.info(f" ===== experiments with batch size: {cfg["batch_size"][i]}, lr: {lr}, lambda: {l} finished ===== ")

        
        # 4. modify config
        logger.info(" === saving config and results === ")
        components = utils.get_component_list(model, optimizer, loss_func, cfg["device"], scheduler)
        cfg.update(components) # update config

        # 5. save experiment & config

        plot_scores = cfg["metrics"][:2]

        for idx, sc in enumerate(plot_scores):
            fig = plt.figure(figsize=(25, 7))
            axes = fig.subplots(1, len(cfg["batch_size"]))
            vmin = min([s[idx] for s in test_scores_all.values()])
            vmax = max([s[idx] for s in test_scores_all.values()])
            for i in range(len(cfg["batch_size"])):
                data = np.zeros((len(cfg["lambda"]), len(cfg["lr"])))
                for j in range(len(cfg["lr"])):
                    for k in range(len(cfg["lambda"])):
                        data[k, j] = test_scores_all[f"bs_{cfg["batch_size"][i]}_lr_{cfg["lr"][j]}_lambda_{cfg["lambda"][k]}"][idx]
                sns.heatmap(data, annot=True, fmt=".3f", xticklabels=cfg["lr"], yticklabels=cfg["lambda"], ax=axes[i], cmap="Blues", cbar=False, vmin=vmin, vmax=vmax)
                axes[i].set_title(f"batch size: {cfg["batch_size"][i]}", fontsize=18)

            plt.suptitle(f"{cfg["note"]} {sc}", fontsize=20, c="red")
            axes[0].set_xlabel("lr", fontsize=14)
            axes[0].set_ylabel("lambda", fontsize=14)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            plt.colorbar(axes[len(cfg["batch_size"])-1].collections[0], cax=cbar_ax)
            cbar_ax.set_title("Score", fontsize=10)
            plt.tight_layout(rect=[0, 0, 0.9, 0.98])
            plt.savefig(os.path.join(cfg["outdir"], f"test_{sc}.png"), dpi=300)
        
        
        utils.save_experiment(
            outdir=cfg["outdir"], config=cfg, model=model, train_losses=train_losses,
            valid_losses=valid_losses, valid_metrics=valid_metrics_all, test_scores=test_scores_all, classes=None
            )
        print(">> experiment finished")

    else:
        # inference only mode
        pass



if __name__ == "__main__":
    main()