"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-12 (Wed)  13:35:12 (+09:00)

Data_handler for AttetiveFP model
"""

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from .featurizer import _prep_feats


class AttentiveFPDatasets(Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs
        self.labels = list(data.y.squeeze(0) for data in graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

class SubsetWrapper(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.indices = indices
        # take over attributes
        self.graphs = [dataset.graphs[i] for i in indices]
        self.labels = [dataset.labels[i] for i in indices]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


def split_dataset(dataset, stratify:bool, split_ratio=[0.8, 0.1, 0.1], seed=42, shuffle=True):
    valid_test_ratio = split_ratio[1] + split_ratio[2]
    full_len = len(dataset)
    train_idx, valid_test_idx = train_test_split(np.arange(full_len),
                                                    test_size=valid_test_ratio,
                                                    stratify=dataset.labels if stratify else None,
                                                    random_state=seed,
                                                    shuffle=shuffle
                                                    )
    
    train_set = SubsetWrapper(dataset, train_idx)
    valid_test_set = SubsetWrapper(dataset, valid_test_idx)
    
    test_ratio = split_ratio[2] * (1 / valid_test_ratio)
    valid_test_len = len(valid_test_idx)
    valid_idx, test_idx = train_test_split(np.arange(valid_test_len),
                                           test_size=test_ratio,
                                           stratify=valid_test_set.labels if stratify else None,
                                           random_state=seed,
                                           shuffle=shuffle
                                           )
    valid_set = SubsetWrapper(valid_test_set, valid_idx)
    test_set = SubsetWrapper(valid_test_set, test_idx)

    return train_set, valid_set, test_set


def prep_AttentiveFPDataLoader(datasets, batch_size, shuffle=True, num_workers=8, pin_memory=True):
    return DataLoader(datasets, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)

def prep_threeAttentiveFPDataLoader(train_set, valid_set, test_set, batch_size, shuffle=True, num_workers=8, pin_memory=True):
    tr = DataLoader(train_set, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val = DataLoader(valid_set, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test = DataLoader(test_set, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return tr, val, test