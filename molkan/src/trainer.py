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
    
    def train(self, trainloader, validloader, num_epochs, save_model_every_n_epochs=0):
        train_losses, valid_losses, valid_metrics = [], [], {}
        for f in self.metrics:
            valid_metrics[f.__name__] = []
        
        pbar = tqdm(range(num_epochs), desc="train_start", dynamic_ncols=True)

        for i in pbar:
            train_loss = self.train_epoch(trainloader)
            valid_loss, valid_scores = self.evaluate(validloader)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            for f, s in zip(self.metrics, valid_scores):
                valid_metrics[f.__name__].append(s)
            