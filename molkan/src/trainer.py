'''
 # @ Author: Takuho Ri (Zehao Li)
 # @ Modified time: 2025-01-15 15:36:08
 # @ Description:
 '''

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.optim import Adam, AdamW, LBFGS
from utils import save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, loss_func, metrics, exp_name, device, logger, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics = metrics
        self.exp_name = exp_name
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
    

    def train(self, trainloader, validloader, num_epochs, earlystopping_patience=10, save_model_every_n_epochs=0):
        train_losses, valid_losses, valid_metrics = [], [], {}
        for f in self.metrics:
            valid_metrics[f.__name__] = []
        
        pbar = tqdm(range(num_epochs), desc="train_start", dynamic_ncols=True)
        self.logger.info("train start")

        best_loss = float("inf")
        est_cnt = 0

        for i in pbar:
            train_loss = self.train_epoch(trainloader)
            valid_loss, valid_scores = self.evaluate(validloader)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            for f, s in zip(self.metrics, valid_scores):
                valid_metrics[f.__name__].append(s)
            pbar.set_description(f"epoch {i} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e} | valid {self.metrics[0].__name__}: {valid_scores[0]:.2e}")
            self.logger.debug(f"epoch {i} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e} | valid {self.metrics[0].__name__}: {valid_scores[0]:.2e}")

            # EarlyStopping
            if best_loss == float("inf"):
                best_loss = valid_loss
                if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0:
                    save_checkpoint(self.exp_name, self.model, i+1)
            elif valid_loss < best_loss:
                best_loss = valid_loss
                if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0:
                    save_checkpoint(self.exp_name, self.model, i+1)
            else:
                est_cnt += 1
            
            if est_cnt == earlystopping_patience:
                self.logger.info(f"Earlystopping at epoch {i}")
                pbar.close()
                break

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
    

    def train_epoch(self,trainloader):
        
