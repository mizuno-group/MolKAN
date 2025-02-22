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
parser.add_argument("--model", type=str, help="MLP-MLP or KAN-MLP or MLP-KAN or KAN-KAN")
parser.add_argument("--seed", type=int)

args = parser.parse_args()

if __name__ == "__main__":

    # makedirs
    outdir = os.path.join(os.path.abspath(__file__).replace("experiments", "results_and_logs").replace(".py", ""), f"{args.dataset}")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "log"), exist_ok=True) # log output dir
    results_jsons = os.path.join(outdir, "results_jsons") # results json output dir
    os.makedirs(results_jsons, exist_ok=True)

    # define config
    config = SimpleNamespace(
        dataset = dataset_dict[args.dataset][0],
        mode = dataset_dict[args.dataset][1],
        model = args.model,
        seed = args.seed,
        epoch = 100,
        use_brier = True if dataset_dict[args.dataset][1]=="c" else False,
        weight_decay = 0.0,
        num_layers = 2,
        num_timesteps = 2,
        dropout = 0.2,
        use_KAN_embed = True if (args.model=="KAN-MLP" or args.model=="KAN-KAN") else False,
        use_KAN_predictor = True if (args.model=="MLP-KAN" or args.model=="KAN-KAN") else False,
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
    config.outdim = graphs[0].y.shape[1]
    dataset = AttentiveFPDatasets(graphs)
    stratify = True if (config.mode=="c" and config.outdim==1) else False 
    trainset, validset, testset = split_dataset(dataset, stratify, seed=config.seed)
    logger.info(f"Train: {len(trainset)}   Validation: {len(validset)}   Test: {len(testset)}   labels: {config.outdim}   Batchsize: 32")
    trainloader, validloader, testloader = prep_threeAttentiveFPDataLoader(trainset, validset, testset, 32, num_workers=16, pin_memory=True)
    
    #　grid search
    logger.info("start grid search ...")
    lrs = [5.0e-3, 2.5e-3, 1.0e-3, 7.5e-4, 5.0e-4]
    logger.info(f"learning rate: {lrs}")
    hidden_dims = [128, 256, 512]
    logger.info(f"hidden layer dimension: {hidden_dims}")
    if config.model != "MLP-MLP":
        grids = [1, 2, 4]
        logger.info(f"KAN grids number: {grids}")
        logger.info(f"Total trials: {len(lrs)*len(hidden_dims)*len(grids)}")
    else:
        logger.info(f"Total trials: {len(lrs)*len(hidden_dims)}")

    for lr in lrs:
        for h_dim in hidden_dims:
            if config.model != "MLP-MLP":
                for grid in grids:
                    logger.info(f">>> lr: {lr}   hidden_dim: {h_dim}   grid: {grid} <<<")
                    config.lr = lr
                    config.hidden_dim = h_dim
                    config.num_grids = grid
                    config.note = f"{config.model}_{config.seed}_{config.lr}_{config.hidden_dim}_{config.grid}"

                    trainer = AttentiveFP_Trainer(config, logger)
                    if config.use_brier:
                        train_losses, train_briers, valid_losses, valid_briers = trainer.train(trainloader, validloader)
                    else:
                        train_losses, valid_losses = trainer.train(trainloader, validloader)
                    # calculate scores for validation set
                    valid_scores = trainer.test(validloader)
                    
                    jsonfile = os.path.join(results_jsons, f'{config.note}.json')
                    with open(jsonfile, 'w') as f:
                        if config.use_brier:
                            data = {
                                'valid_scores': valid_scores,
                                'train_losses': train_losses,
                                'train_briers': train_briers,
                                'valid_losses': valid_losses,
                                'valid_briers': valid_briers,
                            }
                        else:
                            data = {
                                'valid_scores': valid_scores,
                                'train_losses': train_losses,
                                'valid_losses': valid_losses
                            }
                        json.dump(data, f, sort_keys=False, indent=4)
            else:
                logger.info(f">>> lr: {lr}   hidden_dim: {h_dim} <<<")
                config.lr = lr
                config.hidden_dim = h_dim
                config.note = f"{config.model}_{config.seed}_{config.lr}_{config.hidden_dim}"

                trainer = AttentiveFP_Trainer(config, logger)
                if config.use_brier:
                    train_losses, train_briers, valid_losses, valid_briers = trainer.train(trainloader, validloader)
                else:
                    train_losses, valid_losses = trainer.train(trainloader, validloader)
                # calculate scores for validation set
                valid_scores = trainer.test(validloader)
                
                jsonfile = os.path.join(results_jsons, f'{config.note}.json')
                with open(jsonfile, 'w') as f:
                    if config.use_brier:
                        data = {
                            'valid_scores': valid_scores,
                            'valid_losses': valid_losses,
                            'valid_briers': valid_briers,
                            'train_losses': train_losses,
                            'train_briers': train_briers,
                        }
                    else:
                        data = {
                            'valid_scores': valid_scores,
                            'valid_losses': valid_losses,
                            'train_losses': train_losses,
                        }
                    json.dump(data, f, sort_keys=False, indent=4)

    logger.info(">>> All trials finished.")
    timer(start_time, logger)

    # Select the best models and jsons
    logger.info("post prosess: selecting the best model...")
    last_valid_loss_dict = {}
    for lr in lrs:
        for h_dim in hidden_dims:
            if config.model != "MLP-MLP":    
                for grid in grids:
                    trial = f"{config.model}_{config.seed}_{lr}_{h_dim}_{grid}"
                    with open(os.path.join(results_jsons, f"{trial}.json"), "r") as f:
                        res = json.load(f)
                    last_valid_loss_dict[trial] = res["valid_losses"][-1]
            else:
                trial = f"{config.model}_{config.seed}_{lr}_{h_dim}"
                with open(os.path.join(results_jsons, f"{trial}.json"), "r") as f:
                    res = json.load(f)
                last_valid_loss_dict[trial] = res["valid_losses"][-1]
    best_trial = min(last_valid_loss_dict, key=last_valid_loss_dict.get)
    logger.info(f">>> best trial: {best_trial}")
    logger.info(f">>> best valid loss: {last_valid_loss_dict[best_trial]}")
    logger.info(f"deleting other results...")
    for lr in lrs:
        for h_dim in hidden_dims:
            if config.model != "MLP-MLP":    
                for grid in grids:
                    trial = f"{config.model}_{config.seed}_{lr}_{h_dim}_{grid}"
                    if trial != best_trial:
                        os.remove(os.path.join(results_jsons, f"{trial}.json"))
                        os.remove(os.path.join(outdir, "models", f"{trial}.pt"))
            else:
                trial = f"{config.model}_{config.seed}_{lr}_{h_dim}"
                if trial != best_trial:
                    os.remove(os.path.join(results_jsons, f"{trial}.json"))
                    os.remove(os.path.join(outdir, "models", f"{trial}.pt"))
    
    # test
    logger.info("test with best model...")
    best_atrs = best_trial.split("_")
    best_lr = best_atrs[2]
    config.lr = best_lr
    best_h_dim = best_atrs[3]
    config.hidden_dim = best_h_dim
    if config.model != "MLP-MLP":
        best_grid = best_atrs[4]
        config.num_grids = best_grid
        config.note = best_trial
    else: 
        config.note = best_trial
    trainer = AttentiveFP_Trainer(config, logger)
    test_scores = trainer.test(testloader, os.path.join(outdir, "models", f"{best_trial}.pt"))

    # saving results
    logger.info("saving results...")
    with open(os.path.join(results_jsons, f"{best_trial}.json"), "r") as f:
        res = json.load(f)
    res["test_scores"] = test_scores
    keys = ["test_scores"] + [key for key in res if key != "test_scores"]
    res = {key: res[key] for key in keys}
    with open(os.path.join(results_jsons, f"{best_trial}.json"), "w") as f:
        json.dump(res, f, sort_keys=False, indent=4)
    
    logger.info("========== All Finished ==========")
