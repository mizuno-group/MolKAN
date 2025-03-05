"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-27 (Thu)  14:30:27 (+09:00)

datahandler for clmpy
Original script: clmpy[https://github.com/mizuno-group/clmpy] composed by Shumpei Nemoto
"""

import os
import sys
import psutil
from collections import defaultdict
from functools import partial
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psutil._common
import gc
import pandas as pd
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

class DistributedBucketSampler(Sampler):
    def __init__(self, dataset, buckets, ddp_sampler, shuffle=True, batch_size=512, drop_last=False):
        """
        dataset: 対象のデータセット
        buckets: (min, max, step) のタプル、例: (20, 150, 10)
        ddp_sampler: DistributedSampler のインスタンス（各プロセスに割り当てたインデックスを提供）
        shuffle: 各バケット内でシャッフルするかどうか
        batch_size: バッチサイズ
        drop_last: 最後のバッチが不完全な場合に落とすかどうか
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.ddp_sampler = ddp_sampler

        # ddp_sampler によって割り当てられたインデックスを取得
        print("loading indices...")
        self.indices = list(ddp_sampler)

        # 割り当てられたインデックスに対して、シーケンス長を計算
        print("loadning sequence length...")
        self.lengths = np.array([(dataset[i][0] != 0).sum() for i in self.indices]).astype(np.int16)
        bucket_range = np.arange(*buckets)
        bucket_assignments = torch.bucketize(torch.tensor(self.lengths), torch.tensor(bucket_range), right=False)

        # バケットごとにインデックスをグループ化
        self.buckets = defaultdict(list)
        for idx, bucket in zip(self.indices, bucket_assignments):
            self.buckets[bucket.item()].append(idx)

        # 必要に応じて、極端に長いシーケンスのバケット（最大値のバケット）を除外
        bucket_max = int(max(bucket_assignments))
        if bucket_max in self.buckets:
            self.buckets.pop(bucket_max)

    def __iter__(self):
        batches = []
        for bucket in self.buckets.values():
            bucket_tensor = torch.tensor(bucket)
            if self.shuffle:
                bucket_tensor = bucket_tensor[torch.randperm(bucket_tensor.nelement())]
            # バッチサイズごとに分割
            bucket_batches = torch.split(bucket_tensor, self.batch_size)
            if self.drop_last and len(bucket_batches) > 1 and len(bucket_batches[-1]) < self.batch_size:
                bucket_batches = bucket_batches[:-1]
            batches.extend(bucket_batches)
        # バッチ単位でもシャッフル
        if self.shuffle:
            random.shuffle(batches)
        # 各バッチはリスト形式で返す
        for batch in batches:
            yield batch.tolist()

    def __len__(self):
        total = 0
        for bucket in self.buckets.values():
            n_batches = len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size:
                n_batches += 1
            total += n_batches
        return total

def read_smiles_from_csv(path):
    for chunk in pd.read_csv(path, usecols=["input", "output"], chunksize=100000):
        for _, row in chunk.iterrows():
            yield row["input"], row["output"]

def tokenize(s, tokens):
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

def sfl_tokenize(s, tokens):
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
                
def one_hot_encoder(tokenized, tokens):
    enc = np.array([tokens.dict[v] for v in tokenized]).astype(np.int16)
    enc = np.concatenate([np.array([1]),enc,np.array([2])]).astype(np.int16)
    assert type(enc) == np.ndarray
    padding_length = 252 - len(enc)
    if padding_length > 0:
        enc = np.concatenate([enc, np.zeros(padding_length, dtype=np.int16)])
    elif padding_length < 0:
        enc = enc[:252]
    return enc

def encode_smiles(smiles_and_args):
    input, output, tokens, tok_func = smiles_and_args
    input_tokenized = tok_func(input, tokens)
    output_tokenized = tok_func(output, tokens)
    return one_hot_encoder(input_tokenized, tokens), one_hot_encoder(output_tokenized, tokens)

def seq2id(csvpath,tokens,npypath,sfl=True):
    tok_func = sfl_tokenize if sfl else tokenize

    print("--> calculating datanum")
    datanum = 0
    for chunk in pd.read_csv(csvpath, usecols=["input"], chunksize=1000000):
        datanum += len(chunk)
    print(f"datanum: {datanum}")
    
    shape = (datanum, 2, 252)
    mm_array = np.memmap(npypath, dtype=np.int16, mode='w+', shape=shape)
    mem = psutil.virtual_memory()
    print(f"memory usage: {psutil._common.bytes2human(mem.used)} ({mem.percent}%)")

    print("---> start encoding")
    args_generator = ((input, output, tokens, tok_func) for input, output in read_smiles_from_csv(csvpath))
    with ProcessPoolExecutor() as executor:
        mapped_smiles = executor.map(encode_smiles, args_generator, chunksize=10000)
        for i, (encoded_input, encoded_output) in enumerate(mapped_smiles):
            mm_array[i, 0] = encoded_input
            mm_array[i, 1] = encoded_output
            if i % 1000000 == 0:
                mm_array.flush()
                gc.collect()
    
    mm_array.flush()
    del mm_array
    gc.collect()

    return datanum

class tokens_table():
    def __init__(self,token_path):
        with open(token_path,"r") as f:
            tokens = f.read().replace("Br","R").replace("Cl","L").split("\n")
        self.table = tokens
        self.id2sm = {i:v for i,v in enumerate(tokens)}
        self.dict = {w:v for v,w in self.id2sm.items()}
        self.length = len(self.table)

class CLM_Dataset(Dataset):
    def __init__(self,csvpath,tokens,npypath,sfl):
        self.tokens = tokens
        print("--> start seq2id encoding...")
        datanum = seq2id(csvpath, tokens, npypath, sfl)
        print("--> encoding finished, .npy saved")
        self.data = np.memmap(npypath, dtype=np.int16, mode="r", shape=(datanum, 2, 252))
        print(self.data[datanum-1])
        self.datanum = datanum
        mem = psutil.virtual_memory()
        print(f"memory usage: {psutil._common.bytes2human(mem.used)} ({mem.percent}%)")

    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        return self.data[idx, 0].copy(), self.data[idx, 1].copy()

class CLM_Dataset_v2(Dataset):
    def __init__(self, path, datanum):
        print("---> start loading tokenized arrays...")
        self.data = np.memmap(path, dtype=np.int16, mode="r", shape=(datanum, 2, 252))
        self.datanum = datanum
    
    def __len__(self):
        return self.datanum
    
    def __getitem__(self, idx):
        return self.data[idx, 0].copy(), self.data[idx, 1].copy()

class Encoder_Dataset(Dataset):
    def __init__(self,x,token,memmapfile,sfl):
        self.tokens = token
        self.input = seq2id(x,self.tokens,memmapfile,sfl)
        self.datanum = len(x)
    
    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        out_i = self.input[idx]
        return out_i

def collate(batch):
    maxlen_x = max((x != 0).sum() for x, _ in batch)
    maxlen_y = max((y != 0).sum() for _, y in batch)
    xs = pad_sequence([torch.LongTensor(x[:maxlen_x]) for x, _ in batch],
                      batch_first=False,padding_value=0) # no padding, just list2tensor and transpose
    ys = pad_sequence([torch.LongTensor(y[:maxlen_y]) for _, y in batch],
                      batch_first=False,padding_value=0)
    return xs, ys

def encoder_collate(batch):
    xs = [torch.LongTensor(x) for x, _ in batch]
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    return xs