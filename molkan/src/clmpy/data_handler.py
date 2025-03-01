"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-27 (Thu)  14:30:27 (+09:00)

datahandler for clmpy
Original script: clmpy[https://github.com/mizuno-group/clmpy] composed by Shumpei Nemoto
"""

from collections import defaultdict
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence


class BucketSampler(Sampler):
    def __init__(self,dataset,buckets=(20,150,10),shuffle=True,batch_size=512,drop_last=False):
        super().__init__(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        length = [len(v[0]) for v in dataset]
        bucket_range = np.arange(*buckets)
        
        assert isinstance(buckets,tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0
        buc = torch.bucketize(torch.tensor(length),torch.tensor(bucket_range),right=False)

        bucs = defaultdict(list)
        bucket_max = max(np.array(buc))
        for i,v in enumerate(buc):
            bucs[v.item()].append(i)
        _ = bucs.pop(bucket_max)
        
        self.buckets = dict()
        for bucket_size, bucket in bucs.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket,dtype=torch.int)
        self.__iter__()

    def __iter__(self):
        for bucket_size in self.buckets.keys():
            self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket,self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)
        if self.shuffle == True:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length
    
def tokenize(s):
    global tokens
    s = s.replace("Br","R").replace("Cl","L")
    tok = []
    while len(s) > 0:
        if len(s) >= 2 and (s[0] == "@" or s[0] == "["):
            for j in np.arange(3,0,-1):
                if s[:j] in tokens.table:
                    tok.append(s[:j])
                    s = s[j:]
                    break
        else:
            tok.append(s[0])
            s = s[1:]
    return tok

def sfl_tokenize(s):
    global tokens
    s = s.replace("Br","R").replace("Cl","L")
    tok = []
    char = ""
    for v in s:
        if len(char) == 0 and v != "[":
            tok.append(v)
            continue
        char += v
        if len(char) > 1:
            if v == "]":
                if char in tokens.table:
                    tok.append(char)
                else:
                    tok.append("<unk>")
                char = ""
    return tok
                
def one_hot_encoder(tokenized):
    global tokens
    enc = np.array([tokens.dict[v] for v in tokenized])
    enc = np.concatenate([np.array([1]),enc,np.array([2])]).astype(np.int32)
    return enc

def seq2id(smiles,tokens,sfl=True):
    tok = sfl_tokenize if sfl else tokenize
    tokenized = []
    with ProcessPoolExecutor() as executor:
        for toks in executor.map(tok, smiles, chunksize=10000):
            tokenized += toks
    encoded = []
    with ProcessPoolExecutor() as executor:
        for encs in executor.map(one_hot_encoder, tokenized, chunksize=10000):
            encoded += encs
    return encoded

class tokens_table():
    def __init__(self,token_path):
        with open(token_path,"r") as f:
            tokens = f.read().replace("Br","R").replace("Cl","L").split("\n")
        self.table = tokens
        self.id2sm = {i:v for i,v in enumerate(tokens)}
        self.dict = {w:v for v,w in self.id2sm.items()}
        self.length = len(self.table)

class CLM_Dataset(Dataset):
    def __init__(self,x,y,token,sfl):
        self.tokens = token
        self.input = seq2id(x,self.tokens,sfl)
        self.output = seq2id(y,self.tokens,sfl)
        self.datanum = len(x)

    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        out_i = self.input[idx]
        out_o = self.output[idx]
        return out_i, out_o
    
class Encoder_Dataset(Dataset):
    def __init__(self,x,token,sfl):
        self.tokens = token
        self.input = seq2id(x,self.tokens,sfl)
        self.datanum = len(x)
    
    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        out_i = self.input[idx]
        return out_i

def collate(batch):
    xs, ys = [], []
    for x,y in batch:
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    ys = pad_sequence(ys,batch_first=False,padding_value=0)
    return xs, ys

def encoder_collate(batch):
    xs = []
    for x in batch:
        xs.append(torch.LongTensor(x))
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    return xs