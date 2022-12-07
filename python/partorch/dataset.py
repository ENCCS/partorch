import argparse
from copy import deepcopy
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import trange, tqdm
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer) -> None:
        super().__init__()
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return self.tokenizer.tokenize(self.smiles_list[index]), self.labels[index]

    def __len__(self):
        return len(self.smiles_list)

def collate_function(batch):
    smiles_arrays, labels = zip(*batch) # Common python idiom  for transposing sequence of sequences
    batch_size = len(smiles_arrays)
    lengths = [len(s) for s in smiles_arrays]
    batch_array = pad_sequence(smiles_arrays, padding_value=0)
    return batch_array, lengths, torch.FloatTensor(labels)


def get_dataloader(*, smiles_list, labels, tokenizer, batch_size, num_workers=0, indices=None, shuffle=False):
    if indices is not None:
        smiles_list = [smiles_list[i] for i in indices]
        labels = [labels[i] for i in indices]
    smiles_dataset = SmilesDataset(smiles_list, labels, tokenizer)
    dataloader = DataLoader(smiles_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_function, num_workers=num_workers, drop_last=False)
    return dataloader

def get_ddp_dataloader(*, rank, world_size, smiles_list, labels, tokenizer, batch_size, num_workers=0, indices=None, shuffle=False):
    if indices is not None:
        smiles_list = [smiles_list[i] for i in indices]
        labels = [labels[i] for i in indices]
    smiles_dataset = SmilesDataset(smiles_list, labels, tokenizer)
    sampler = DistributedSampler(smiles_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    dataloader = DataLoader(smiles_dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_function, num_workers=num_workers)
    return dataloader