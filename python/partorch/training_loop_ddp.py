from copy import deepcopy

from sklearn.metrics import roc_auc_score
from tqdm import trange, tqdm
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.distributed as dist

def train_ddp(*, train_dataloader, dev_dataloader, writer, max_epochs, model_class, device, model_args=None, model_hparams=None, model_kwargs=None):
    if model_args is None:
        model_args = tuple()
    if model_kwargs is None:
        model_kwargs = dict()
    if model_hparams is None:
        model_hparams = dict()
    
    model = model_class(*model_args, **model_kwargs, device=device, **model_hparams)
    ddp_model = DDP(model)
    
    best_roc_auc = 0
    best_model = None
    best_iteration = 0
    iteration = 0
    
    learning_rate = model_hparams['learning_rate']
    weight_decay = model_hparams['weight_decay']
    optimizer = AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    for e in range(max_epochs):
        training_losses = []
        dev_losses = []
        dev_targets = []
        dev_predictions = []
        
        ddp_model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            sequence_batch, lengths, labels = batch
            logit_prediction = ddp_model(sequence_batch.to(model.device), lengths)
            loss = loss_fn(logit_prediction.squeeze(), labels.to(model.device))
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), iteration)
            training_losses.append(loss.item())
            iteration += 1

        ddp_model.eval()
        for batch in dev_dataloader:
            with torch.no_grad():
                sequence_batch, lengths, labels = batch
                logit_prediction = ddp_model(sequence_batch.to(model.device), lengths)
                loss = loss_fn(logit_prediction.squeeze(), labels.to(model.device))
                prob_predictions = torch.sigmoid(logit_prediction)
            
            dev_losses.append(loss.item())
            dev_targets.extend(labels.cpu().numpy())
            dev_predictions.extend(prob_predictions.cpu().numpy())
        
        dev_roc_auc = roc_auc_score(dev_targets, dev_predictions)

        writer.add_scalar('Loss/dev', np.mean(dev_losses), iteration)
        writer.add_scalar('ROC_AUC/dev', dev_roc_auc, iteration)
        print(f"Training loss {np.mean(training_losses)}\tDev loss: {np.mean(dev_losses)}\tDev ROC AUC:{dev_roc_auc}")
        
        if dist.get_rank() == 0 and dev_roc_auc > best_roc_auc:
            best_roc_auc = dev_roc_auc
            best_model = deepcopy(model)
            best_model.recurrent_layers.flatten_parameters()  # After the deepcopy, the weight matrices are not necessarily in contiguous memory, this fixes that issue
            best_iteration = iteration

    return best_model, best_iteration