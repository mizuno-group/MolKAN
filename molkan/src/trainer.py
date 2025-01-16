'''
 # @ Author: Takuho Ri (Zehao Li)
 # @ Modified time: 2025-01-15 15:36:08
 # @ Description:
 '''

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, AdamW, LBFGS


class Trainer:
    def __init__(self, model, optimizer, criterion, exp_name, device, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.exp_name = exp_name
        self.device = device
        self.scheduler = scheduler
    
    