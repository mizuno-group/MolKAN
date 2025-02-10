"""
created on Feb 11 2025

Featurizer for the AttentiveFP model

This featurizer is based on the original paper and the implementations from the authors
[1] Xiong, Zhaoping, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li, et al. 2020. “Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism.” Journal of Medicinal Chemistry 63 (16): 8749–60.
- httpd://github.com/OpenDrugAI/AttentiveFP
"""

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def _atom_features(atom, explicit_H=False, use_chirality=True):
    atom_sets = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "At", "other"]
    hybridization_sets = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, 
                          Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, "other"]
    atom_feats = one_of_k_encoding_unk(atom.GetSymbol(), atom_sets) + \
                          one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
                          [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                          one_of_k_encoding_unk(atom.GetHybridization(), hybridization_sets) + \
                          [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        atom_feats = atom_feats + \
                              one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            atom_feats = atom_feats + \
                                  one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                                  [atom.HasProp('_ChiralityPossible')]
        except:
            atom_feats = atom_feats + \
                                  [False, False] + \
                                  [atom.HasProp('_ChiralityPossible')]

    return torch.tensor(atom_feats, dtype=torch.float)


def _bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + \
                           one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return torch.tensor(bond_feats, dtype=torch.float)

def _prep_feats(smiles, use_chirality=True):
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Failed to featurize SMILES: %s" % smiles)
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(_atom_features(atom, explicit_H=False, use_chirality=use_chirality))
    atom_features_tensor = torch.stack(atom_features_list)
    # bonds
    bond_index_list = []
    bond_features_list = []
    for bond in mol.GetBonds():
        bond_index_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        bond_index_list.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        bond_features_list.append([_bond_features(bond), _bond_features(bond)])
    if len(bond_index_list) == 0:
        return atom_features_tensor, torch.zeros((2, 0), dtype=torch.long), torch.zerops((0, 10), dtype=torch.float)
    else:
        bond_index_tensor = torch.tensor(bond_index_list, dtype=torch.long).t().contiguous()
        bond_features_tensor = torch.stack(bond_features_list)
        return atom_features_tensor, bond_index_tensor, bond_features_tensor