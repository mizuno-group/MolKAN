"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-27 (Thu)  14:50:10 (+09:00)

training module for clmpy.gru (DDP)
Original script: clmpy[https://github.com/mizuno-group/clmpy] composed by Shumpei Nemoto
"""


import os
import sys
from argparse import ArgumentParser, FileType
import yaml
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import get_rank
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("/work/gd43/a97009/MolKAN/molkan/src")

from clmpy.GRU.model import GRU
from clmpy.preprocess import *
from clmpy.utils import plot_loss, init_logger, fix_seed, count_param

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config",type=FileType(mode="r"),default=None)
    parser.add_argument("--max_lr", type=float, default=None)
    args = parser.parse_args()
    config_dict = yaml.load(args.config,Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        arg_dict[key] = value
    args.config = args.config.name
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.patience = args.patience_step // args.valid_step_range
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.local_rank = int(os.environ["LOCAL_RANK"])
    return args


class Trainer():
    def __init__(
        self,
        args,
        model: nn.Module,
        criteria: nn.Module,
        optimizer: optim.Optimizer,
        es,
        logger=None
    ):
        self.logger = logger
        self.model = model
        self.train_path = args.train_data
        self.valid_data = prep_valid_encoded_data(args)
        self.criteria = criteria
        self.optimizer = optimizer
        self.es = es
        self.steps_run = 0
        self.ckpt_path = os.path.join(args.experiment_dir,"checkpoint.pt")
        if os.path.exists(self.ckpt_path):
            self._load(self.ckpt_path)
        self.best_model = None

    def _load(self,path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_run = ckpt["step"]
        self.es.num_bad_steps = ckpt["num_bad_steps"]
        self.es.best = ckpt["es_best"]

    def _save(self,path,step):
        self.optimizer.eval()
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "num_bad_steps": self.es.num_bad_steps,
            "es_best": self.es.best
        }
        torch.save(ckpt,path)

    def _train_batch(self,source,target,device):
        self.model.train()
        self.optimizer.train()
        self.optimizer.zero_grad()
        source = source.to(device)
        target = target.to(device)
        out, _ = self.model(source,target[:-1,:])
        l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
        l.backward()
        self.optimizer.step()
        return l.item()
    
    def _valid_batch(self,source,target,device):
        self.model.eval()
        self.optimizer.eval()
        source = source.to(device)
        target = target.to(device)
        with torch.no_grad():
            out, _ = self.model(source,target[:-1,:])
            l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
        return l.item()
    
    def _train(self,args,train_data):
        l, l2 = [], []
        min_l2 = float("inf")
        end = False
        for datas in train_data:
            self.steps_run += 1
            l_t = self._train_batch(*datas,args.device)
            l.append(l_t)
            if self.steps_run % args.valid_step_range == 0 and args.global_rank == 0:
                l_v = []
                with torch.no_grad():
                    for v, w in self.valid_data:
                        l_v.append(self._valid_batch(v,w,args.device))
                l_v = np.mean(l_v)
                l2.append(l_v)
                end = self.es.step(l_v)
                if len(l) == 1 or l_v < min_l2:
                    self.best_model = self.model
                    min_l2 = l_v
                self._save(self.ckpt_path,self.steps_run)
                self.logger.info(f"step {self.steps_run} | train_loss: {l_t}, valid_loss: {l_v}")
                if end:
                    self.logger.info(f"Early stopping at step {self.steps_run}")
                    torch.distributed.destroy_process_group()
                    return l, l2, end
            if self.steps_run >= args.steps:
                end = True
                return l, l2, end
        return l, l2, end
    
    def train(self,args):
        end = False
        l, l2 = [], []
        train_data = prep_train_encoded_data(args)
        if args.global_rank == 0:
            self.logger.info("train start...")
        while end == False:
            a, b, end = self._train(args,train_data)
            l.extend(a)
            l2.extend(b)
        if self.steps_run == args.steps and args.global_rank == 0:
            self.logger.info(">>> reaching train step limit. TRAIN FINISHED.")
            torch.distributed.destroy_process_group()
        return l, l2
    

def main():
    ts = time.perf_counter()

    args = get_args()
    torch.distributed.init_process_group(backend="nccl")
    args.global_rank = get_rank()
    if args.global_rank == 0:
        logger = init_logger(args.experiment_dir, filename=f"maxlr_{args.max_lr}.log")
    seed = args.seed + args.global_rank
    fix_seed(seed, fix_gpu=False)
    model = GRU(args)
    params, tr_params = count_param(model)
    if args.global_rank == 0:
        logger.info(f"params: {params}  trainable params: {tr_params}")
    if torch.cuda.is_available():
        model = model.to(args.device)
        model = DDP(module=model, device_ids=[args.local_rank])
    else:
        raise ValueError("Can't use CUDA !!! Check your environment !!!")
    criteria, optimizer, es = load_train_objs(args,model)
    if args.global_rank == 0:
        trainer = Trainer(args, model,criteria,optimizer, es, logger)
    else:
        trainer = Trainer(args, model,criteria,optimizer, es)
    loss_t, loss_v = trainer.train(args)
    if args.global_rank == 0:
        logger.info("saving results...")
        torch.save(trainer.best_model.state_dict(),os.path.join(args.experiment_dir,f"best_model_maxlr_{args.max_lr}.pt"))
        os.remove(trainer.ckpt_path)
        if args.plot:
            plot_loss(loss_t,loss_v,dir_name=args.experiment_dir, plot_name=f"maxlr_{args.max_lr}")
        logger.info(">>> experiment finished.")

    tg = time.perf_counter()
    dt = tg - ts
    h = dt // 3600
    m = (dt % 3600) // 60
    s = dt % 60
    if args.global_rank == 0:
        logger.info(f"elapsed time: {h} h {m} min {s} sec")

if __name__ == "__main__":
    main()