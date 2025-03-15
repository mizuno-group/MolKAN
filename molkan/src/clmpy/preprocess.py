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
from torch.utils.data import Subset, DataLoader, DistributedSampler

from .data_handler import *
from .utils import EarlyStopping, warmup_schedule


def load_train_objs(args,model, esmode="min"):
    criteria = nn.CrossEntropyLoss(reduction="sum")
    optimizer = RAdamScheduleFree(model.parameters(),lr=args.max_lr)
    es = EarlyStopping(mode=esmode, patience=args.patience)
    return criteria, optimizer, es

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    seed += worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def prep_train_data(args):
    buckets = (args.buckets_min, args.buckets_max, args.buckets_step)
    trainset = CLM_Dataset(args.train_data,args.token,os.path.join("/work/gd43/a97009/MolKAN/molkan/data/pubchem", "110m_train.npy"),args.SFL)
    ddpsampler = DistributedSampler(trainset, shuffle=True)
    train_sampler = DistributedBucketSampler(trainset,buckets,shuffle=False,batch_size=args.batch_size)
    train_loader = DataLoader(trainset,
                              batch_sampler=train_sampler,
                              collate_fn=collate,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    return train_loader

def prep_3train_encoded_data(args):
    trainset1 = CLM_Dataset_v2(args.train_data1, args.train_datanum1, args.train_datadim1)
    trainset2 = CLM_Dataset_v2(args.train_data2, args.train_datanum2, args.train_datadim2)
    trainset3 = CLM_Dataset_v2(args.train_data3, args.train_datanum3, args.train_datadim3)
    ddpsampler1 = DistributedSampler(trainset1, shuffle=True)
    ddpsampler2 = DistributedSampler(trainset2, shuffle=True)
    ddpsampler3 = DistributedSampler(trainset3, shuffle=True)
    print("load DataLoader...")
    train_loader1 = DataLoader(trainset1,
                              sampler=ddpsampler1,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    train_loader2 = DataLoader(trainset2,
                              sampler=ddpsampler2,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    train_loader3 = DataLoader(trainset3,
                              sampler=ddpsampler3,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    return train_loader1, train_loader2, train_loader3

def prep_valid_data(args):
    validset = CLM_Dataset(args.valid_data,args.token, os.path.join("/work/gd43/a97009/MolKAN/molkan/data/pubchem", "110m_valid.npy"), args.SFL)
    valid_loader = DataLoader(validset,
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    return valid_loader

def prep_valid_encoded_data(args):
    validset = CLM_Dataset_v2(args.valid_data, args.valid_datanum, args.valid_datadim)
    valid1 = DataLoader(Subset(validset, range(8000)),
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    valid2 = DataLoader(Subset(validset, range(9500)),
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    full_loader = DataLoader(validset,
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    return valid1, valid2, full_loader

def prep_valid_encoded_data_v2(args):
    validset = CLM_Dataset_v2(args.valid_data, args.valid_datanum, args.valid_datadim)
    valid1 = DataLoader(Subset(validset, range(8000)),
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    valid2 = DataLoader(Subset(validset, range(8000, 9500)),
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    valid3 = DataLoader(Subset(validset, range(9500, 10000)),
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    full = DataLoader(validset,
                      shuffle=False,
                      collate_fn=collate,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      pin_memory=True)
    return valid1, valid2, valid3, full

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