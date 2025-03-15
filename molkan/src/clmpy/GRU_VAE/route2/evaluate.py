"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-27 (Thu)  14:54:40 (+09:00)

evaluation module for clmpy.gruvae
Original script: clmpy[https://github.com/mizuno-group/clmpy] composed by Shumpei Nemoto
"""


import os
import sys
from argparse import ArgumentParser, FileType
import yaml
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

sys.path.append("/work/gd43/a97009/MolKAN/molkan/src")

from clmpy.GRU_VAE.route2.model import GRUVAE
from clmpy.preprocess import *
from clmpy.utils import init_logger

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config",type=FileType(mode="r"),default="config.yml")
    parser.add_argument("--max_lr", type=float)
    args = parser.parse_args()
    config_dict = yaml.load(args.config,Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        arg_dict[key] = value
    args.config = args.config.name
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.model_path = os.path.join(args.experiment_dir, f"best_model_maxlr_{args.max_lr}.pt")
    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

class Evaluator():
    def __init__(self,model,args,train=False):
        self.args = args
        self.id2sm = args.token.id2sm
        self.maxlen = args.maxlen
        if not train and args.model_path:
            self.model = model.to(args.device)
            self._load(args.model_path)
        elif train:
            self.model = model

    def _load(self,path):
        self.model.load_state_dict(self.remove_module_prefix(torch.load(path)))

    def _eval_batch(self,source,target,device,train=False):
        source = source.to(device)
        if not train:
            latent, _ = self.model.encoder(source)
        else:
            latent, _ = self.model.module.encoder(source)
        token_ids = np.zeros((self.maxlen,source.size(1)))
        token_ids[0,:] = 1
        token_ids = torch.tensor(token_ids,dtype=torch.long).to(device)
        for i in range(1,self.maxlen):
            token_ids_seq = token_ids[i-1,:].unsqueeze(0)
            if i == 1:
                if not train:
                    output, latent = self.model.decoder(token_ids_seq,latent)
                else:
                    output, latent = self.model.module.decoder(token_ids_seq,latent)
            else:
                if not train:
                    output, latent = self.model.decoder.gru2out(token_ids_seq,latent)
                else:
                    output, latent = self.model.module.decoder.gru2out(token_ids_seq,latent)
            _, new_id = output.max(dim=2)
            is_end_token = token_ids_seq == 2
            is_pad_token = token_ids_seq == 0
            judge = torch.logical_or(is_end_token,is_pad_token)
            if judge.sum().item() == judge.numel():
                token_ids = token_ids[:i,:]
                break
            new_id[judge] = 0
            token_ids[i,:] = new_id
        pred = token_ids[1:,:]
        row = []
        invalid = 0
        partial_accuracy = 0
        for s,t,v in zip(source.T,target.T,pred.T):
            x = [self.id2sm[j.item()] for j in s]
            y = [self.id2sm[j.item()] for j in t]
            p = [self.id2sm[j.item()] for j in v]
            try:
                ans_toklen = y.index(self.id2sm[2]) - 1
            except:
                invalid += 1
                continue
            x_str = "".join(x[1:]).split(self.id2sm[2])[0].replace("R","Br").replace("L","Cl")
            y_str = "".join(y[1:]).split(self.id2sm[2])[0].replace("R","Br").replace("L","Cl")
            p_str = "".join(p).split(self.id2sm[2])[0].replace("R","Br").replace("L","Cl")
            judge = True if y_str == p_str else False
            par = 0
            for i in range(min(len(y_str), len(p_str))):
                if y_str[i] == p_str[i]:
                    par += 1
            par = par / max(len(y_str), len(p_str))
            partial_accuracy += par
            row.append([ans_toklen, judge, x_str,y_str,p_str])
        return row, partial_accuracy, invalid

    def evaluate(self):
        self.model.eval()
        res = []
        t1, t2 , test_data = prep_valid_encoded_data(self.args)
        partial_accuracy = 0
        with torch.no_grad():
            for source, target in test_data:
                row, par, invalid = self._eval_batch(source,target,self.args.device)
                res.extend(row)
                partial_accuracy += par
        pred_df = pd.DataFrame(res,columns=["ans_tokenlength","judge","input","answer","predict"])
        accuracy = len(pred_df.query("judge == True")) / len(pred_df)
        partial_accuracy = partial_accuracy / (self.args.valid_datanum - invalid)
        return pred_df, accuracy, partial_accuracy

    def evaluate_train(self, valid):
        self.model.eval()
        partial_accuracy = 0
        num_invalid = 0
        with torch.no_grad():
            for source, target in valid:
                _, par, invalid = self._eval_batch(source,target,self.args.device, train=True)
                num_invalid += invalid
                partial_accuracy += par
        partial_accuracy = partial_accuracy / (self.args.valid_datanum - num_invalid)
        return partial_accuracy
    
    def remove_module_prefix(self, state_dict):
        """
        state_dictのキーから'module.'プレフィックスを取り除く関数
        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # module.を取り除く
            new_state_dict[name] = v
        return new_state_dict

def main():
    args = get_args()
    logger = init_logger(args.experiment_dir, f"maxlr_{args.max_lr}.log")
    model = GRUVAE(args)
    evaluator = Evaluator(model,args)
    results, accuracy, partial_accuracy = evaluator.evaluate()
    results = results.sort_values("ans_tokenlength")
    results.to_csv(os.path.join(args.experiment_dir,f"evaluate_result_maxlr_{args.max_lr}.csv"))
    logger.info(f"best model perfect accuracy: {accuracy}")
    logger.info(f"best model partial accuracy: {partial_accuracy}")
   

if __name__ == "__main__":
    main()