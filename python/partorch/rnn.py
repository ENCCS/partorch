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



class RNNPredictor(nn.Module):
    def __init__(self, *, tokenizer, device, embedding_dim, d_model, num_layers, bidirectional, dropout, learning_rate, weight_decay) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.embedding = nn.Embedding(tokenizer.get_num_embeddings(), 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        self.recurrent_layers = nn.LSTM(input_size=embedding_dim, 
                                        hidden_size=d_model, 
                                        num_layers=num_layers,
                                        bidirectional=bidirectional,
                                        dropout=dropout,
                                        )
        self.num_directions = 1
        if bidirectional:
            self.num_directions = 2
        self.output_layers = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.num_directions*d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.to(self.device)
    
    def forward(self, sequence_batch, lengths):
        embedded_sequences = self.embedding(sequence_batch)
        packed_sequence = pack_padded_sequence(embedded_sequences, lengths, enforce_sorted=False)
        output, (h_n, c_n) = self.recurrent_layers(packed_sequence)
        final_state = h_n[-1]
        if self.num_directions == 2:
            final_state = torch.cat((final_state, h_n[-1]), dim=-1)
        logits = self.output_layers(final_state)
        return logits

    def loss_on_batch(self, batch):
        sequence_batch, lengths, labels = batch
        logit_prediction = self(sequence_batch.to(self.device), lengths)
        loss = self.loss_fn(logit_prediction.squeeze(), labels.to(self.device))
        return loss

    def train_batch(self, batch):
        self.train()
        self.optimizer.zero_grad()
        loss = self.loss_on_batch(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_batch(self, batch):
        self.eval()
        with torch.no_grad():
            loss = self.loss_on_batch(batch)
            return loss.item()

    def eval_and_predict_batch(self, batch):
        self.eval()
        with torch.no_grad():
            sequence_batch, lengths, labels = batch
            logit_prediction = self(sequence_batch.to(self.device), lengths)
            loss = self.loss_fn(logit_prediction.squeeze(), labels.to(self.device))
            prob_predictions = torch.sigmoid(logit_prediction)
            return loss.item(), labels.cpu().numpy(), prob_predictions.cpu().numpy()

    def set_optimizer(self, learning_rate, weight_decay, **kwargs):
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)