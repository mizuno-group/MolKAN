import os
from datetime import datetime
import argparse
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

import src.utils as utils
from src.AttentiveFP.model import AttentiveFP
from src.AttentiveFP.data_handler import AttentiveFPDatasets, split_dataset, prep_threeAttentiveFPDataLoader


def train_epoch(model, optimizer, loss_func, trainloader, device):
    model.train()
    optimizer.train()
    total_loss = 0
    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch)

        mask = (~torch.isnan(data.y)).float()
        y = torch.where(torch.isnan(data.y), torch.zeros_like(data.y), data.y)
                
        loss = loss_func(pred, y, weight=mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(trainloader.dataset)
        
# evaluation
def evaluate(model, optimizer, loss_func, validloader, device):
    with torch.no_grad():
        model.eval()
        optimizer.eval()
        total_loss = 0
        for data in validloader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)

            mask = (~torch.isnan(data.y)).float()
            y = torch.where(torch.isnan(data.y), torch.zeros_like(data.y), data.y)

            loss = loss_func(pred, y, mask)
            total_loss += loss.item() * data.num_graphs
            
        valid_loss = total_loss / len(validloader.dataset)
    return valid_loss

# test
def test(model, testloader, metrics, device):
    with torch.no_grad():
        model.eval()
        test_scores = []
        pred_list = []
        y_list = []
        for data in testloader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred_list.append(pred.detach().cpu().numpy())
            y_list.append(data.y.detach().cpu().numpy())
        
        pred = np.concatenate(pred_list)
        y = np.concatenate(y_list)
        for f in metrics:
            if hasattr(utils.Metrics, f):
                func = getattr(utils.Metrics, f)
                test_scores.append(func(pred, y))
            else:
                raise AttributeError(f"Metric not found({f}), check your metric name.")

    return test_scores


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    utils.fix_seed(seed, fix_gpu=True)
    
    # prep_data
    df = pd.read_csv(os.path.join(project_path, "data", "tox_csv", "bace.csv"), index_col=0)
    smiles = list(df["cano_smi"])
    y = np.array(df[["Class"]])
    datasets = AttentiveFPDatasets(smiles, y)
    train_set, valid_set, test_set = split_dataset(datasets, stratify=True, seed=seed)
    trainloader, validloader, testloader = prep_threeAttentiveFPDataLoader(train_set, valid_set, test_set,
                                                                           batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    model = AttentiveFP(mode="c", in_channels=39, hidden_channels=256, out_channels=1, edge_dim=10, num_layers=2, num_timesteps=2, dropout=0.2).to(device)
    optimizer = RAdamScheduleFree(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    loss_func = utils.BCELoss()
    metrics = ["accuracy", "AUROC", "sensitivity", "precision", "MCC"]
    
    pbar = tqdm(range(10), desc="train_start", dynamic_ncols=True)
    utils.fix_seed(seed, fix_gpu=True)
    for i in pbar:
        train_loss = train_epoch(model, optimizer, loss_func, trainloader, device)
        valid_loss = evaluate(model, optimizer, loss_func, validloader, device)
        pbar.set_description(f"epoch {i+1} | train loss: {train_loss} | valid loss: {valid_loss}")
    
    scores = test(model, testloader, metrics, device)
    for metric, score in zip(metrics, scores):
        print(f"{metric}: {score}")

# OK! 
    



    
    