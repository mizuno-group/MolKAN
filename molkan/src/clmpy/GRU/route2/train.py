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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import get_rank
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("/work/gd43/a97009/MolKAN/molkan/src")

from clmpy.GRU.route2.model import GRU
from clmpy.GRU.route2.evaluate import Evaluator
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
        self.valid1, self.valid2, self.valid = prep_valid_encoded_data(args)
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
    
    def _train(self,args,train_data,trainfrag):
        l, l2, par = [], [], []
        min_l2 = float("inf")
        end = False
        for datas in train_data:
            self.steps_run += 1
            l_t = self._train_batch(*datas,args.device)
            l.append(l_t)
            if self.steps_run % args.valid_step_range == 0:
                l_v = []
                if trainfrag == 0:
                    for v, w in self.valid1:
                        l_v.append(self._valid_batch(v,w,args.device))
                    l_v = np.mean(l_v)
                    l2.append(l_v)
                    p = Evaluator(self.model, args, train=True).evaluate_train(self.valid1)
                    par.append(p)
                elif trainfrag == 1:
                    for v, w in self.valid2:
                        l_v.append(self._valid_batch(v,w,args.device))
                    l_v = np.mean(l_v)
                    l2.append(l_v)
                    p = Evaluator(self.model, args, train=True).evaluate_train(self.valid2)
                    par.append(p)
                elif trainfrag == 2:
                    for v, w in self.valid:
                        l_v.append(self._valid_batch(v,w,args.device))
                    l_v = np.mean(l_v)
                    l2.append(l_v)
                    p = Evaluator(self.model, args, train=True).evaluate_train(self.valid)
                    par.append(p)

                # 全ノードの検証ロスを集約
                l_v_tensor = torch.tensor([l_v], dtype=torch.float32, requires_grad=False).to(args.device)
                gathered_l_v = [torch.zeros_like(l_v_tensor) for _ in range(args.world_size)]
                torch.distributed.barrier()
                torch.distributed.all_gather(gathered_l_v, l_v_tensor)

                avg_l_v = torch.stack(gathered_l_v).mean().item()
                end = self.es.step(avg_l_v)

                if avg_l_v < min_l2:
                    self.best_model = self.model
                    min_l2 = avg_l_v

                if args.global_rank == 0:
                    self._save(self.ckpt_path,self.steps_run)
                    self.logger.info(f"step {self.steps_run} | train_loss: {l_t}, valid_loss: {avg_l_v}, valid_partial: {p}")
                gc.collect()
                
                if end:
                    if args.global_rank == 0:
                            self.logger.info(f"Early stopping at step {self.steps_run}")
                    torch.distributed.barrier()
                    break
            if self.steps_run >= args.steps:
                end = True
                if args.global_rank == 0:
                    self.logger.info(f"Reaching train steps limit: {args.steps}. Train finished.")
                torch.distributed.barrier()
                break
        return l, l2, par, end
    
    def train(self,args):
        end = False
        l, l2, par = [], [], []
        train_data1, train_data2, train_data3 = prep_3train_encoded_data(args)
        if args.global_rank == 0:
            self.logger.info("train start...")
        for train_data, trainfrag in zip([train_data1, train_data2, train_data3], range(3)):
            while end == False:
                a, b, p, end = self._train(args,train_data, trainfrag)
                l.extend(a)
                l2.extend(b)
                par.extend(p)
            if args.global_rank == 0:
                self.logger.info(f">>> train fragment {trainfrag+1} finished.")
            end = False
        return l, l2, par

def main():
    ts = time.perf_counter()

    args = get_args()
    torch.distributed.init_process_group(backend="nccl")
    args.global_rank = get_rank()
    if args.global_rank == 0:
        logger = init_logger(args.experiment_dir, filename=f"maxlr_{args.max_lr}.log")
    fix_seed(args.seed, fix_gpu=False)
    model = GRU(args)
    if args.global_rank == 0:
        params, tr_params = count_param(model)
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
    loss_t, loss_v, par = trainer.train(args)
    if args.global_rank == 0:
        logger.info("saving results...")
        torch.save(trainer.best_model.state_dict(),os.path.join(args.experiment_dir,f"best_model_maxlr_{args.max_lr}.pt"))
        os.remove(trainer.ckpt_path)
        if args.plot:
            plot_loss(loss_t,loss_v,args.valid_step_range, dir_name=args.experiment_dir, plot_name=f"maxlr_{args.max_lr}")
            plt.plot(par)
            plt.xlabel(f"step * {args.valid_step_range}")
            plt.ylabel("partial accuracy")
            plt.savefig(os.path.join(args.experiment_dir,f"maxlr_{args.max_lr}_partial_accuracy.png"), bbox_inches="tight")
            plt.close()
        args.model_path = os.path.join(args.experiment_dir,f"best_model_maxlr_{args.max_lr}.pt")
        model = GRU(args)
        results, accuracy, partial_accuracy = Evaluator(model, args).evaluate()
        results = results.sort_values("ans_tokenlength")
        results.to_csv(os.path.join(args.experiment_dir,f"evaluate_result_maxlr_{args.max_lr}.csv"))
        logger.info(f"best model perfect accuracy: {accuracy}")
        logger.info(f"best model partial accuracy: {partial_accuracy}") 
        logger.info(">>> experiment finished.")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    tg = time.perf_counter()
    dt = tg - ts
    h = dt // 3600
    m = (dt % 3600) // 60
    s = dt % 60
    if args.global_rank == 0:
        logger.info(f"elapsed time: {h} h {m} min {s} sec")

if __name__ == "__main__":
    main()