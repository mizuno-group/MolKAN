import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# original packages in src
from ..src import utils
from ..src import data_handler as dh
from ..src.trainer import Trainer
"""change based on the model you want to use"""
from ..src.models import *


""" === change based on your experiment === """
parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument('datadir', type=str, help='working directory that contains the dataset')
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument('--train', type=bool, default=True, help='whether to train or not')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--save_every_n_epochs', type=float, default=5)
parser.add_argument('--seed', type=str, default=42)
parser.add_argument('--num_workers', type=str, default=16, help='number of workers for dataloader')

args = parser.parse_args() # Namespace object

# convert to config dictionary
cfg = vars(args)

# fix seed
utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# setup
now = datetime.datetime.now().strftime('%y%m%d')
cfg["outdir"] = __file__.replace("experiments", "results_and_logs") # for output
if not os.path.exists(cfg["outdir"]):
    os.makedirs(cfg["outdir"])
cfg["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
