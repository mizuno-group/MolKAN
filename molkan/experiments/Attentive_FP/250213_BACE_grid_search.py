import os
from datetime import datetime
import argparse
from types import SimpleNamespace
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

from src.utils import init_logger, fix_seed
from src.AttentiveFP.data_handler import AttentiveFPDatasets, split_dataset, prep_threeAttentiveFPDataLoader
from src.AttentiveFP.trainer import AttentiveFP_Trainer

if __name__ == "__main__":

    outdir = os.path.abspath(__file__).replace("experiments", "results_and_logs").replace(".py", "")
    os.makedirs(outdir, exist_ok=True)

    config = SimpleNamespace(
        epoch = 500,
        mode = "c",
        weight_decay = 0.0,
        hidden_dim = 256,
        out_dim = 1,
        num_layers = 2,
        num_timesteps = 2,
        dropout = 0.2,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        seed = 42,
        outdir = outdir
    )

    logger = init_logger(outdir)
    
    fix_seed(config.seed, fix_gpu=True)
    # prepare data
    logger.info("Preparing data...")
    with open(os.path.join(project_path, "data", "AttentiveFP_graphs", "bace_c.pkl"), "rb") as f:
        graphs = pickle.load(f)
    datasets = AttentiveFPDatasets(graphs)
    trainset, validset, testset = split_dataset(datasets, stratify=True, seed=config.seed)
    logger.info(f"train: {len(trainset)}, valid: {len(validset)}, test: {len(testset)}")
    trainloader, validloader, testloader = prep_threeAttentiveFPDataLoader(trainset, validset, testset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    # Grid Search
    lrs = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]
    grids = [1, 2, 4, 8]

    # Grid Search
    for use_KAN_predictor in [False, True]:
        for use_KAN_embed in [False, True]:
            trainloss_dict = {}
            validloss_dict = {}
            testscores_dict = {}
            for lr in lrs:
                if use_KAN_predictor or use_KAN_embed:
                    for num_grids in grids:
                        config.use_KAN_predictor = use_KAN_predictor
                        config.use_KAN_embed = use_KAN_embed
                        config.lr = lr
                        config.num_grids = num_grids
                        config.note = f"use_KAN_embed_{use_KAN_embed}_use_KAN_predictor_{use_KAN_predictor}_lr_{lr}_num_grids_{num_grids}"

                        logger.info(f"----- {config.note} -----")
                        trainer = AttentiveFP_Trainer(config, logger)
                        train_losses, valid_losses = trainer.train(trainloader, validloader)
                        test_scores = trainer.test(testloader)
                        trainer.save_experiments()

                        trainloss_dict[config.note] = train_losses
                        validloss_dict[config.note] = valid_losses
                        testscores_dict[config.note] = test_scores
                else:
                    config.use_KAN_predictor = use_KAN_predictor
                    config.use_KAN_embed = use_KAN_embed
                    config.lr = lr
                    config.num_grids = None
                    config.note = f"use_KAN_embed_{use_KAN_embed}_use_KAN_predictor_{use_KAN_predictor}_lr_{lr}"

                    logger.info(f"----- {config.note} -----")
                    trainer = AttentiveFP_Trainer(config, logger)
                    train_losses, valid_losses = trainer.train(trainloader, validloader)
                    test_scores = trainer.test(testloader)
                    trainer.save_experiments()

                    trainloss_dict[config.note] = train_losses
                    validloss_dict[config.note] = valid_losses
                    testscores_dict[config.note] = test_scores
        
            # save the progress
            logger.info("Saving the progress...")
            progress_dir = os.path.join(outdir, "progress")
            os.makedirs(progress_dir, exist_ok=True)
            if use_KAN_predictor or use_KAN_embed:
                fig, ax = plt.subplots(4, 4, figsize=(32, 24))
                fig.suptitle(f"Grid Search for use_KAN_predictor_{use_KAN_predictor}_use_KAN_embed_{use_KAN_embed}", fontsize=24)
                epoch = list(range(1, config.epoch+1))
                for i, (key, value) in enumerate(trainloss_dict.items()):
                    ax[i%4, i//4].plot(epoch, value, label="train loss")
                    ax[i%4, i//4].plot(epoch, validloss_dict[key], label="valid loss")
                    ax[i%4, i//4].set_title(f"{key} - testAUROC:{testscores_dict[key]["AUROC"]:.3f}")
                    ax[i%4, i//4].set_xlabel("epoch")
                    ax[i%4, i//4].set_ylabel("loss")
                    ax[i%4, i//4].legend()
                plt.tight_layout()
                fig.savefig(os.path.join(progress_dir, f"Grid_Search_use_KAN_predictor_{use_KAN_predictor}_use_KAN_embed_{use_KAN_embed}.png"))
                plt.close()
            else:
                fig, ax = plt.subplots(1, 4, figsize=(32, 6))
                fig.suptitle(f"Grid Search for use_KAN_predictor_{use_KAN_predictor}_use_KAN_embed_{use_KAN_embed}", fontsize=24)
                epoch = list(range(1, config.epoch+1))
                for i, (key, value) in enumerate(trainloss_dict.items()):
                    ax[i].plot(epoch, value, label="train loss")
                    ax[i].plot(epoch, validloss_dict[key], label="valid loss")
                    ax[i].set_title(f"{key} - testAUROC:{testscores_dict[key]["AUROC"]:.3f}")
                    ax[i].set_xlabel("epoch")
                    ax[i].set_ylabel("loss")
                    ax[i].legend()
                plt.tight_layout()
                plt.savefig(os.path.join(progress_dir, f"Grid_Search_use_KAN_predictor_{use_KAN_predictor}_use_KAN_embed_{use_KAN_embed}.png"))

    logger.info("All experiments finished.")


