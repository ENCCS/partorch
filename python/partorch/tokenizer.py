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
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter

class BytesTokenizer:
    def tokenize(self, sequence):
        return torch.LongTensor([ord(x)+1 for x in sequence]) # +1 just to definitely reserve 0 for the padding
    
    def get_num_embeddings(self):
        return 257 # We allow for any extended ascii characted + 1