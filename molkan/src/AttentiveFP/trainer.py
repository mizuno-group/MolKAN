import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch_geometric as tg
from tqdm import tqdm

from schedulefree import RAdamScheduleFree

from .model import AttentiveFP
from ..utils import fix_seed, count_param, save_experiment, save_checkpoint, BCELoss, MSE, Metrics

def _train_epoch(model, optimizer, loss_func, trainloader, device):
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

def _evaluate(model, optimizer, loss_func, validloader, device):
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

def _test(model, testloader, metrics, device):
    with torch.no_grad():
        model.eval()
        test_scores = {}
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
            if hasattr(Metrics, f):
                func = getattr(Metrics, f)
                test_scores[f] = func(pred, y)
            else:
                raise AttributeError(f"Metric not found({f}), check your metric name.")

    return test_scores

class AttentiveFP_Trainer():
    def __init__(self, config:dict, logger):
        self.config = config

        self.epoch = config.epoch
        self.mode = config.mode
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.use_KAN_embed = config.use_KAN_embed
        self.use_KAN_predictor = config.use_KAN_predictor
        self.num_grids = config.num_grids
        self.hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        self.num_layers = config.num_layers
        self.num_timesteps = config.num_timesteps
        self.dropout = config.dropout
        
        self.device = config.device
        self.seed = config.seed
        self.outdir = config.outdir
        self.note = config.note
        self.logger = logger

        fix_seed(self.seed, fix_gpu=True)
        self.model = AttentiveFP(self.mode, 40, self.hidden_dim, self.out_dim, 10, self.num_layers, self.num_timesteps, 
                                 dropout=self.dropout, use_KAN_embed=self.use_KAN_embed, use_KAN_predictor=self.use_KAN_predictor, num_grids=self.num_grids).to(self.device)
        total_params, trainable_params = count_param(self.model)
        self.logger.info(f"Total parameters: {total_params}")
        self.logger.info(f"Trainable parameters: {trainable_params}")
        self.optimizer = RAdamScheduleFree(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        if self.mode == "c":
            self.loss_func = BCELoss()
            self.metrics = ["accuracy", "AUROC", "sensitivity", "precision", "MCC"]
        else:
            self.loss_func = MSE()
            self.metrics = ["R2", "RMSE", "MAE"]
        
    def train(self, trainloader, validloader):
        self.logger.info(f"=== train start ===")
        
        pbar = tqdm(range(self.epoch), desc="train start", dynamic_ncols=True)
        fix_seed(self.seed, fix_gpu=True)
        
        # train
        train_losses = []
        valid_losses = []
        for e in pbar:
            train_loss = _train_epoch(self.model, self.optimizer, self.loss_func, trainloader, self.device)
            valid_loss = _evaluate(self.model, self.optimizer, self.loss_func, validloader, self.device)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            pbar.set_description(f"epoch {e+1} | train loss: {train_loss:.3e} | valid loss: {valid_loss:.3e}")
            self.logger.debug(f"epoch {e+1} | train loss: {train_loss:.3e} | valid loss: {valid_loss:.3e}")
        save_checkpoint(self.model, self.outdir, self.note)
        self.logger.info(f"=== train finished ===")

        self.train_losses = train_losses
        self.valid_losses = valid_losses

        return train_losses, valid_losses

    def test(self, testloader):
        self.logger.info(f"=== test start ===")
        test_scores = _test(self.model, testloader, self.metrics, self.device)
        for k, v in test_scores.items():
            self.logger.info(f"{k}: {v:.4e}")
        self.logger.info(f"=== test finished ===")

        self.test_scores = test_scores
    
        return test_scores
    
    def save_experiments(self, save_config=False):
        self.logger.info("saving experiments...")
        if not save_config:
            outdir = os.path.join(self.outdir, "results_json")
            os.makedirs(outdir, exist_ok=True)
            save_experiment(outdir, vars(self.config), self.model, self.train_losses, self.valid_losses, self.test_scores, note=self.note, save_config=save_config)
        else:
            save_experiment(outdir, vars(self.config), self.model, self.train_losses, self.valid_losses, self.test_scores, note= self.note, save_config=save_config)


        
        

