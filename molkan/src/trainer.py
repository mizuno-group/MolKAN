'''
 # @ Author: Takuho Ri (Zehao Li)
 # @ Modified time: 2025-01-15 15:36:08
 # @ Description:
 '''

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.utils import save_checkpoint, Metrics


class Trainer:
    def __init__(self, model, optimizer, loss_func, metrics, outdir, device, logger, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics = metrics
        self.outdir = outdir
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        

    def train(self, trainloader, validloader, num_epochs, note=None, earlystopping_patience=10):
        train_losses, valid_losses, valid_metrics = [], [], {}
        for f in self.metrics:
            valid_metrics[f] = []
        
        pbar = tqdm(range(num_epochs), desc="train_start", dynamic_ncols=True)
        self.logger.info("=== train start ===")

        best_loss = float("inf")
        est_cnt = 0

        for i in pbar:
            train_loss = self.train_epoch(trainloader)
            valid_loss, valid_scores = self.evaluate(validloader)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            for f, s in zip(self.metrics, valid_scores):
                valid_metrics[f].append(s)
            pbar.set_description(f"epoch {i} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e} | valid {self.metrics[0]}: {valid_scores[0]:.2e}")
            self.logger.debug(f"epoch {i} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e} | valid {self.metrics[0]}: {valid_scores[0]:.2e}")

            # EarlyStopping
            if best_loss == float("inf"):
                best_loss = valid_loss
                best_epoch = i+1
                save_checkpoint(self.model, self.outdir, note=note)
            elif valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = i+1
                save_checkpoint(self.model, self.outdir, note=note)
            else:
                est_cnt += 1
            
            if est_cnt == earlystopping_patience:
                self.logger.info(f"Earlystopping at epoch {i+1}")
                pbar.close()
                break

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.logger.info("=== train end ===")
        return train_losses, valid_losses, valid_metrics, best_epoch
    

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0
        for x, y in trainloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_func(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(trainloader.dataset)
    

    def evaluate(self, validloader):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            valid_scores = []
            pred_list = []
            y_list = []
            for x, y in validloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_func(pred, y)
                total_loss += loss.item() * len(x)
                pred_list.append(pred.detach().cpu().numpy())
                y_list.append(y.detach().cpu().numpy())
            
            valid_loss = total_loss / len(validloader.dataset)
            pred = np.concatenate(pred_list)
            y = np.concatenate(y_list)
            for f in self.metrics:
                if hasattr(Metrics, f):
                    func = getattr(Metrics, f)
                    valid_scores.append(func(pred, y))
                else:
                    raise AttributeError(f"Metric not found({f}), check your metric name.")
            return valid_loss, valid_scores


def test_func(model, test_loader, metrics, device):
    """
    Perform testing

    Args:
        model (nn.Module): The model
        test_loader (DataLoader): DataLoader for testing
        metrics (list): List of metrics
        device (torch.device): Device
        logger (logging.Logger): Logger
    
    Returns:
        list: List of test scores
    """

    with torch.no_grad():
        model.eval()
        test_scores = []
        pred_list = []
        y_list = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_list.append(pred.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())
        
        pred = np.concatenate(pred_list)
        y = np.concatenate(y_list)
        for f in metrics:
            if hasattr(Metrics, f):
                func = getattr(Metrics, f)
                test_scores.append(func(pred, y))
            else:
                raise AttributeError(f"Metric not found({f}), check your metric name.")

        return test_scores

            

