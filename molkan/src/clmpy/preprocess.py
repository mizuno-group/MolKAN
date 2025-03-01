"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-27 (Thu)  14:35:50 (+09:00)

preprocess for clmpy
Original script: clmpy[https://github.com/mizuno-group/clmpy] composed by Shumpei Nemoto
"""

import argparse
import yaml
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader, DistributedSampler

from .data_handler import *
from .utils import EarlyStopping, warmup_schedule


def load_train_objs(args,model):
    criteria = nn.CrossEntropyLoss(reduction="sum")
    optimizer = RAdamScheduleFree(model.parameters(),lr=args.max_lr)
    es = EarlyStopping(patience=args.patience)
    return criteria, optimizer, es

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    seed += worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def prep_train_data(args):
    buckets = (args.buckets_min, args.buckets_max, args.buckets_step)
    trainset = CLM_Dataset(args.train_data,args.token,os.path.join(args.experiment_dir, f"train_{args.global_rank}.dat"),args.SFL) # メモリを抑えるオプションを入れたい
    ddpsampler = DistributedSampler(trainset, shuffle=True)
    train_sampler = DistributedBucketSampler(trainset,buckets,ddpsampler,shuffle=False,batch_size=args.batch_size)
    train_loader = DataLoader(trainset,
                              batch_sampler=train_sampler,
                              collate_fn=collate,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    return train_loader

def prep_valid_data(args):
    validset = CLM_Dataset(args.valid_data,args.token, os.path.join(args.experiment_dir, f"valid_{args.global_rank}.dat"), args.SFL)
    valid_loader = DataLoader(validset,
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    return valid_loader

def prep_encode_data(args,smiles):
    dataset = Encoder_Dataset(smiles,args)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        collate_fn=encoder_collate,
                        num_workers=args.num_workers)
    return loader

def prep_token(token_path):
    tokens = tokens_table(token_path)
    return tokens

def get_notebook_args(config_file,**kwargs):
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    with open(config_file,"r") as f:
        config = yaml.safe_load(f)
    for v,w in config.items():
        args.__dict__[v] = w
    for v,w in kwargs:
        args.__dict__[v] = w
    try:
        args.patience = args.patience_step // args.valid_step_range
    except AttributeError:
        pass
    args.config = config_file
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.model_path = ""
    return args