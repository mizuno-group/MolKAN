import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# original packages in src
from ..src import utils
from ..src import data_handler as dh
from ..src.trainer import Trainer, test_func
""" === change based on the model you want to use === """
from ..src.models import *


""" === Config Setup: change based on your experiment === """

# Example usage for list input:
# --batch_size 32, 64, 128
# --lr 0.001, 0.01, 0.1
# --metrics accuracy, AUROC, sensitivity

parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument('datadir', type=str, help='working directory that contains the dataset')
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--train', type=bool, default=True, help='whether to train or not')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=utils.parse_list_or_int, default=128, help="batch size (int or list(int))")
parser.add_argument('--lr', type=utils.parse_list_or_float, default=0.001, help="learning rate (float or list(float))")
parser.add_argument('--metrics', type=utils.parse_str_list, help="metrics to evaluate (list(str))")
parser.add_argument('--num_workers', type=int, default=16, help='number of workers for dataloader')

args = parser.parse_args() # Namespace object

# convert to config dictionary
cfg = vars(args)

# fix seed
utils.fix_seed(seed=cfg["seed"], fix_gpu=True)

# other setup
cfg["outdir"] = os.path.dirname(__file__.replace("experiments", "results_and_logs")) + os.path.basename(__file__).replace(".py", "") # for output
os.makedirs(cfg["outdir"], exist_ok=True)
if cfg["batch_size"] is int:
    cfg["batch_size"] = [cfg["batch_size"]]
if cfg["lr"] is float:
    cfg["lr"] = [cfg["lr"]]
cfg["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device


""" === Data Preparation: change based on your experiments === """
def prepare_data(logger):
    """
    Load data and prepare data loaders.
    Place the processed data in the data directory or load the data from the path specified by the argument.
    It is also convenient to prepare for loading data for inference.
    """

    logger.info("=== data preparation start ===")

    """ ↓↓↓ LOAD DATA ↓↓↓ """

    # write your code here

    """ ↑↑↑ LOAD DATA ↑↑↑ """
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
def prepare_model(logger):
    """
    Prepare model, loss, optimizer, and scheduler.
    Use if statements as needed to control with arguments.
    """

    logger.info("=== model preparation start ===")

    model = Model() # change this to your model
    model.to(cfg["device"])
    loss_func = nn.CrossEntropyLoss() # change this to your loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"]) # change this to your optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['num_epochs'], eta_min=0
        ) # change this to your scheduler
    return model, loss_func, optimizer, scheduler


""" === train func === """
def fit(model, train_loader, valid_loader, loss_func, metrics, optimizer, scheduler, logger):
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
        outdir=cfg["outdir"], device=cfg["device"], logger=logger, scheduler=scheduler
        )
    train_loss, valid_loss, valid_metrics = trainer.train(
        train_loader, valid_loader, num_epochs=cfg["num_epochs"],
        earlystopping_patience=10
        )
    return train_loss, valid_loss, valid_metrics


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


""" === Main === """
def main():
    if args.train:
        # training mode
        logger = utils.init_logger(cfg["outdir"])
        start = time.time() # for time stamp
        # 1. data prep
        train_loaders, valid_loaders, test_loaders = prepare_data(logger)
        cfg["num_training_data"] = len(train_loaders[0])
        cfg["num_valid_data"] = len(valid_loaders[0])
        cfg["num_test_data"] = len(test_loaders[0])     
        # 2. model prep
        model, loss_func, optimizers, scheduler = prepare_model(logger)
        # 3. training
        _batch_sizes = [] 
        for bs in cfg["batch_size"]:
            _batch_sizes.extend([bs] * len(cfg["lr"]))
        _lr = []
        _batch_sizes.extend(cfg["lr"] * len(cfg["batch_size"]))
        for i, (train_loader, valid_loader) in enumerate(zip(train_loaders, valid_loaders)):
            for j, optimizer in enumerate(optimizers):
                logger.info(f" ===== experiments with batch size: {_batch_sizes[i]}, lr: {_lr[j]} ===== ")
                train_loss, valid_loss, valid_metrics  = fit(
                    model, train_loader, valid_loader, loss_func, cfg["metrics"], 
                    optimizer, scheduler, logger
                    )
                model.load_state_dict(torch.load(f"{cfg['outdir']}/model.pth"))
        # 4. modify config
        components = utils.get_component_list(model, optimizer, loss_func, cfg["device"], scheduler)
        cfg.update(components) # update config
        elapsed_time = utils.timer(start) # for time stamp
        cfg["elapsed_time"] = elapsed_time
        # 5. save experiment & config
        utils.save_experiment(
            outdir=cfg["outdir"], config=cfg, model=model, train_losses=train_loss,
            test_losses=test_loss, accuracies=accuracies, classes=None, base_dir=cfg["outdir"]
            )
        print(">> Done training")
    else:
        # inference only mode
        pass



if __name__ == "__main__":
    main()