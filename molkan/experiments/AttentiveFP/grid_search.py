"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-20 (Thu)  17:01:12 (+09:00)

Script for grid search of AttentiveFP molecular property prediction
"""

import os
from datetime import datetime
import argparse
from types import SimpleNamespace
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import pickle

from mpi4py import MPI

import sys
project_path = os.path.abspath(__file__).split("/experiments")[0]
sys.path.append(project_path)

from src.utils import init_logger, fix_seed, timer
from src.AttentiveFP.data_handler import AttentiveFPDatasets, split_dataset, prep_threeAttentiveFPDataLoader
from src.AttentiveFP.trainer import AttentiveFP_Trainer

dataset_dict = {"BACE-C": ["bace_c.pkl", "c"],
                                    "BACE-R": ["bace_r.pkl", "r"],
                                    "ClinTox": ["clintox_M.pkl", "c"],
                                    "CYP2C9": ["cyp2c9_inhib.pkl", "c"],
                                    "CYP3A4": ["cyp3a4_inhib.pkl", "c"],
                                    "hERG": ["herg_karim.pkl", "c"],
                                    "LD50": ["ld50.pkl", "r"],
                                    "SIDER": ["sider.pkl", "c"],
                                    "Tox21": ["tox21_M.pkl", "c"],
                                    "ToxCast": ["toxcast_M.pkl", "c"]}

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)

args = parser.parse_args()

if __name__ == "__main__":

    # get mpi process ID
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # makedirs
    outdir = os.path.join(os.path.abspath(__file__).replace("experiments", "results_and_logs").replace(".py", ""), f"{args.dataset}")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "log"), exist_ok=True) # log output dir
    trials_dir = os.path.join(outdir, "trials") # trial progresses output dir
    os.makedirs(trials_dir, exist_ok=True)
    results_dir = os.path.join(outdir, "results") # test results output dir
    os.makedirs(results_dir, exist_ok=True)

    # define config
    models=["MLP-MLP", "KAN-MLP", "MLP-KAN", "KAN-KAN"]
    seeds=[42, 7, 2025, 1234, 31415]
    model = models[rank//5]
    seed = seeds[rank%5]

    config = SimpleNamespace(
        dataset = dataset_dict[args.dataset][0],
        mode = dataset_dict[args.dataset][1],
        model = model,
        seed = seed,
        epoch = 100,
        use_brier = True if dataset_dict[args.dataset][1]=="c" else False,
        weight_decay = 0.0,
        num_layers = 2,
        num_timesteps = 2,
        dropout = 0.2,
        use_KAN_embed = True if (model=="KAN-MLP" or model=="KAN-KAN") else False,
        use_KAN_predictor = True if (model=="MLP-KAN" or model=="KAN-KAN") else False,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        outdir = outdir
    )

    # initialize experiments
    logger = init_logger(os.path.join(outdir, "log"), filename=f"{config.model}_{config.seed}.txt")
    logger.info(f"========== Grid Search: AttentiveFP({config.model}) {args.dataset} seed-{config.seed} ==========")
    fix_seed(config.seed, fix_gpu=True)
    start_time = time.time()

    # prepare data
    logger.info("preparing data...")
    with open(os.path.join(project_path, "data", "AttentiveFP_graphs", config.dataset), "rb") as f:
        graphs = pickle.load(f)
    config.out_dim = graphs[0].y.shape[1]
    dataset = AttentiveFPDatasets(graphs)
    stratify = True if (config.mode=="c" and config.out_dim==1) else False 
    trainset, validset, testset = split_dataset(dataset, stratify, seed=config.seed)
    logger.info(f"Train: {len(trainset)}   Validation: {len(validset)}   Test: {len(testset)}   labels: {config.out_dim}   Batchsize: 32")
    trainloader, validloader, testloader = prep_threeAttentiveFPDataLoader(trainset, validset, testset, 32, num_workers=16, pin_memory=True)
    
    #　grid search
    logger.info("start grid search ...")
    lrs = [5.0e-3, 3.0e-3, 1.0e-3, 7.0e-4, 5.0e-4]
    logger.info(f"learning rate: {lrs}")
    hidden_dims = [128, 256, 512]
    logger.info(f"hidden layer dimension: {hidden_dims}")
    if config.model != "MLP-MLP":
        grids = [1, 2, 4]
        logger.info(f"KAN grids number: {grids}")
        logger.info(f"Total trials: {len(lrs)*len(hidden_dims)*len(grids)}")
    else:
        logger.info(f"Total trials: {len(lrs)*len(hidden_dims)}")

    # dict saving all trials
    data = {}

    for lr in lrs:
        for h_dim in hidden_dims:
            if config.model != "MLP-MLP":
                for grid in grids:
                    logger.info(f">>> lr: {lr}   hidden_dim: {h_dim}   grid: {grid} <<<")
                    config.lr = lr
                    config.hidden_dim = h_dim
                    config.num_grids = grid
                    config.note = f"{config.model}_{config.seed}_{config.lr}_{config.hidden_dim}_{config.num_grids}"

                    trainer = AttentiveFP_Trainer(config, logger)
                    if config.use_brier:
                        train_losses, train_briers, valid_losses, valid_briers = trainer.train(trainloader, validloader)
                        valid_scores = trainer.test(validloader)
                        res = {
                            "valid_scores": valid_scores,
                            "train_losses": train_losses,
                            "train_briers": train_briers,
                            "valid_losses": valid_losses,
                            "valid_briers": valid_briers
                        }
                        data[config.note] = res
                    else:
                        train_losses, valid_losses = trainer.train(trainloader, validloader)
                        valid_scores = trainer.test(validloader)
                        res = {
                            "valid_scores": valid_scores,
                            "train_losses": train_losses,
                            "valid_losses": valid_losses
                        }
                        data[config.note] = res

            else:
                logger.info(f">>> lr: {lr}   hidden_dim: {h_dim} <<<")
                config.lr = lr
                config.hidden_dim = h_dim
                config.num_grids = None
                config.note = f"{config.model}_{config.seed}_{config.lr}_{config.hidden_dim}"

                trainer = AttentiveFP_Trainer(config, logger)
                if config.use_brier:
                    train_losses, train_briers, valid_losses, valid_briers = trainer.train(trainloader, validloader)
                    valid_scores = trainer.test(validloader)
                    res = {
                        "valid_scores": valid_scores,
                        "train_losses": train_losses,
                        "train_briers": train_briers,
                        "valid_losses": valid_losses,
                        "valid_briers": valid_briers
                    }
                    data[config.note] = res
                else:
                    train_losses, valid_losses = trainer.train(trainloader, validloader)
                    valid_scores = trainer.test(validloader)
                    res = {
                        "valid_scores": valid_scores,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses
                    }
                    data[config.note] = res

    logger.info(">>> All trials finished.")
    timer(start_time, logger)

    # saving the trials and select the best models
    logger.info("post prosess: saving trials and selecting the best model...")
    with open(os.path.join(trials_dir, f"{config.model}_{config.seed}.json"), "w") as f:
        json.dump(data, f, sort_keys=False, indent=4)
    
    best_trial = min(data, key=lambda k: data[k]['valid_losses'][-1])
    
    logger.info(f">>> best trial: {best_trial}")
    logger.info(f">>> best valid loss: {data[best_trial]["valid_losses"][-1]}")
    logger.info(f"deleting other models...")
    for lr in lrs:
        for h_dim in hidden_dims:
            if config.model != "MLP-MLP":    
                for grid in grids:
                    trial = f"{config.model}_{config.seed}_{lr}_{h_dim}_{grid}"
                    if trial != best_trial:
                        os.remove(os.path.join(outdir, "models", f"{trial}.pt"))
            else:
                trial = f"{config.model}_{config.seed}_{lr}_{h_dim}"
                if trial != best_trial:
                    os.remove(os.path.join(outdir, "models", f"{trial}.pt"))

    # test
    logger.info("test with best model...")
    best_atrs = best_trial.split("_")
    best_lr = float(best_atrs[2])
    config.lr = best_lr
    best_h_dim = int(best_atrs[3])
    config.hidden_dim = best_h_dim
    if config.model != "MLP-MLP":
        best_grid = int(best_atrs[4])
        config.num_grids = best_grid
        config.note = best_trial
    else: 
        config.num_grids = None
        config.note = best_trial
    trainer = AttentiveFP_Trainer(config, logger)
    test_scores = trainer.test(testloader, os.path.join(outdir, "models", f"{best_trial}.pt"))

    # saving results
    logger.info("saving results...")
    res = {
        "best_trial": best_trial,
        "test_scores": test_scores
    }
    with open(os.path.join(results_dir, f"{config.model}_{config.seed}.json"), "w") as f:
        json.dump(res, f, sort_keys=False, indent=4)
    
    logger.info("========== All Finished ==========")
