'''
 # @ Author: Takuho Ri (Zehao Li)
 # @ Modified time: 2025-01-17 20:00:17
 # @ Description:
 '''

import numpy as np
from typing import Tuple, Optional, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split


# frozen
class MyDataset(Dataset):
    """
    Custom dataset implementation for supervised and unsupervised tasks.

    Args:
        data (np.ndarray): Array containing the data samples.
        label (Optional[np.ndarray]): Array containing labels for supervised learning.
                                      Defaults to `None` for unsupervised learning.
        transform (Optional[Union[transforms.Compose, List[callable]]]): 
                 A single transformation or a list of transformations applied to each sample.
    """
    def __init__(self, data:np.ndarray, label:Optional[np.ndarray]=None, transform=None):
        if data is None:
            raise ValueError("`data` cannot be None. Please provide the input data.")
        if label is None:
            label = np.full(len(data), np.nan)  # Assign NaN for unsupervised learning
        if not isinstance(transform, list):
            self.transform = [transform]
        else:
            self.transform = transform
        self.data = data
        self.label = label
        self.datanum = len(self.data)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.datanum

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Retrieves a single data sample and its corresponding label.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            Tuple[torch.Tensor, float]: Transformed data sample and its label.
        """
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            for t in self.transform:
                if t is not None:
                    out_data = t(out_data)
                    out_label = t(out_label)
        return out_data, out_label


class array_to_tensors_flaot:
    """
    Converts numpy arrays to PyTorch tensors (float type).

    This can be extended with additional transformations as needed.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, x:np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a PyTorch tensor.
        
        Args:
            x (np.ndarray): Input numpy array.
        
        Returns:
            torch.Tensor: Converted tensor.
        """
        return torch.from_numpy(x.astype(np.float32))


def prep_dataset(data:np.ndarray, label:Optional[np.ndarray]=None, transform=None) -> MyDataset:
    """
    Prepares a PyTorch dataset from raw data and labels.

    Args:
        data (np.ndarray): Input data samples.
        label (Optional[np.ndarray]): Labels for supervised learning, defaults to None.
        transform (Optional[Union[transforms.Compose, List[callable]]]): 
                  Transformations to apply to the data samples.

    Returns:
        MyDataset: A dataset instance.
    """
    return MyDataset(data, label, transform)


def split_dataset(
    full_dataset:Dataset, split_ratio:list=[0.8, 0.1, 0.1], shuffle:bool=True,
    transform:Optional[List[callable]]=None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits a dataset into training and validation sets.

    Args:
        full_dataset (torch.utils.data.Dataset): Dataset instance.
        split_ratio (list): Ratio of the dataset to use for train/valid/test.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: 
        Training and validation datasets.
    """
    dataset_size = len(full_dataset)
    lengths = [int(round(dataset_size * r)) for r in split_ratio]
    while sum(lengths) > dataset_size:
        lengths[lengths.index(max(lengths))] -= 1
    while sum(lengths) < dataset_size:
        lengths[lengths.index(min(lengths))] += 1
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, valid_indices, test_indices= indices[:lengths[0]], indices[lengths[0]:lengths[0]+lengths[1]], indices[lengths[0]+lengths[1]:] 
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)
    test_dataset = Subset(full_dataset, test_indices)
    # transformの適用
    train_dataset = SubsetWrapper(train_dataset, transform)
    valid_dataset = SubsetWrapper(valid_dataset, transform)
    test_dataset = SubsetWrapper(test_dataset, transform)
    return train_dataset, valid_dataset, test_dataset


def split_dataset_stratified(
    full_dataset:Dataset, split_ratio:list=[0.8, 0.1, 0.1], shuffle:bool=True,
    transform:Optional[List[callable]]=None
) -> Tuple[Dataset, Dataset, Dataset]:
    
    valid_test_ratio = split_ratio[1] + split_ratio[2]
    x_train, x_valid_test, y_train, y_valid_test = train_test_split(full_dataset.data, full_dataset.label, 
                                                                    test_size=valid_test_ratio, stratify=full_dataset.label, shuffle=shuffle)
    test_ratio = split_ratio[2] * (1 / valid_test_ratio)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test,
                                                        test_size=test_ratio, stratify=y_valid_test, shuffle=shuffle)
    train_dataset = prep_dataset(x_train, y_train, transform=transform)
    valid_dataset = prep_dataset(x_valid, y_valid, transform=transform)
    test_dataset = prep_dataset(x_test, y_test, transform=transform)
    return train_dataset, valid_dataset, test_dataset


class SubsetWrapper(Dataset):
    """
    Wrapper class for creating a subset of a given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Original dataset.
        indices (List[int]): List of indices to include in the subset.
    """
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (torch.utils.data.Dataset): original Dataset
            transform (callable, optional): transformation to apply to the data
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mol, label = self.dataset[idx]
        if self.transform is not None:
            mol = self.transform(mol)
            label = self.transform(label)
        return mol, label
    

def prep_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 16,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Prepares a PyTorch DataLoader for training or testing.

    Args:
        dataset (torch.utils.data.Dataset): Dataset instance.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads for data loading.
        pin_memory (bool): Whether to use pinned memory for faster data transfer.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader instance.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )


def _worker_init_fn(worker_id: int):
    """
    Initializes each worker with a unique random seed to ensure reproducibility.

    Args:
        worker_id (int): Unique identifier for the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prep_data(
    train_x: np.ndarray,
    train_y: Optional[np.ndarray],
    test_x: np.ndarray,
    test_y: Optional[np.ndarray],
    batch_size: int,
    transform: Tuple[Optional[List[callable]], Optional[List[callable]]] = (None, None),
    shuffle: Tuple[bool, bool] = (True, False),
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares training and testing data loaders from raw data.

    Args:
        train_x (np.ndarray): Training data samples.
        train_y (Optional[np.ndarray]): Training labels (or None for unsupervised learning).
        test_x (np.ndarray): Testing data samples.
        test_y (Optional[np.ndarray]): Testing labels (or None for unsupervised learning).
        batch_size (int): Number of samples per batch.
        transform (Tuple[Optional[List[callable]], Optional[List[callable]]]): 
                  Transformations for training and testing data, respectively.
        shuffle (Tuple[bool, bool]): Whether to shuffle training and testing data.
        num_workers (int): Number of worker threads for data loading.
        pin_memory (bool): Whether to use pinned memory for faster data transfer.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
        Data loaders for training and testing.
    """
    train_dataset = prep_dataset(train_x, train_y, transform[0])
    test_dataset = prep_dataset(test_x, test_y, transform[1])
    train_loader = prep_dataloader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle[0], 
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = prep_dataloader(
        dataset=test_dataset, batch_size=batch_size, shuffle=shuffle[1], 
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader


class GenFeatures:
    def __init__(self):
        self.symbols = [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
            'Te', 'I', 'At', 'other'
        ]

        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        # Generate AttentiveFP features according to Table 1.
        mol = Chem.MolFromSmiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            degree = [0.] * 6
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data

def GetGraphDataFrameFromSMILES(data, col_smi, col_label):
    