'''
 # @ Author: Takuho Ri (Zehao Li)
 # @ Modified time: 2025-01-15 15:33:59
 # @ Description: 
 '''

import numpy as np
import pandas as pd
from torch import nn
from kan import KAN


class KAN_predictor(nn.Module):
    def __init__(self, width:list, grid=20, k=3):
        super().__init__()
        self.width = width
        self.kan = KAN(width=self.width, grid=grid, k=k, auto_save=False)
    
    def forward(self, x):
        return self.kan.forward(x, singularity_avoiding=False, y_th=1000)


class MLP_predictor(nn.Module):
    def __init__(self, width:list, mode):
        super().__init__()
        self.width = width
        self.mode = mode
        num_layers = len(self.width) - 1
        layers = [nn.Linear(self.width[i], self.width[i+1]) for i in range(num_layers)]
        for i in range(num_layers-1):
            layers.insert(i*2+1, nn.ReLU())
        if self.mode == "classification":
            layers.append(nn.Sigmoid())
        self.mlp = nn.ModuleList(layers)
    
    def forward(self, input):
        for layer in self.mlp:
            x = layer(x)
        return x