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

from src.utils import init_logger
from src.AttentiveFP.data_handler import AttentiveFPDatasets, split_dataset, prep_threeAttentiveFPDataLoader
from src.AttentiveFP.trainer import AttentiveFP_Trainer

if __name__ == "__main__":

    outdir = os.path.abspath(__file__).replace("experiments", "results_and_logs").replace(".py", "")
    os.makedirs(outdir, exist_ok=True)

    config = SimpleNamespace(
        epoch = 200,
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
    
    # Grid Search
    for use_KAN_predictor in [False, True]:
        for use_KAN_embed in [False, True]:
            for lr in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]:
                if use_KAN_predictor or use_KAN_embed:
                    



