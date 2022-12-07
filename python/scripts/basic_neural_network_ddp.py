import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

import torch
from torch.utils.tensorboard import SummaryWriter

from partorch.tokenizer import BytesTokenizer
from partorch.dataset import get_ddp_dataloader, get_dataloader
from partorch.training_loop_ddp import train_ddp
from partorch.rnn import RNNPredictor

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn



def main():
    parser = argparse.ArgumentParser(
        description='Basic Hyper Parameter optimization example')
    parser.add_argument('dataset', type=Path)
    parser.add_argument(
        '--random-seed', help="The random seed to use", type=int, default=1729)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    labels = df['p_np'].tolist()
    smiles_list = df['smiles'].tolist()

    tokenizer = BytesTokenizer()

    max_epochs = 100
    batch_size = 256

    torch.random.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    visible_index, heldout_indices = train_test_split(np.arange(len(
        smiles_list)), stratify=labels, shuffle=True, test_size=0.1, random_state=args.random_seed)
    visible_labels = [labels[i] for i in visible_index]
    train_indices, dev_indices = train_test_split(
        visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)

    world_size = torch.cuda.device_count()

    distributed_kwargs = dict(tokenizer=tokenizer,
                              smiles_list=smiles_list, labels=labels, train_indices=train_indices, batch_size=batch_size,
                              dev_indices=dev_indices, heldout_indices=heldout_indices, max_epochs=max_epochs, backend='nccl')

    mp.spawn(distributed_training,
             args=(world_size, distributed_kwargs),
             join=True, nprocs=world_size)

    

def distributed_training(rank, world_size, kwargs):
    dist.init_process_group(
        kwargs['backend'], rank=rank, world_size=world_size)

    device = torch.device(f'cuda:{rank}')

    smiles_list, labels = kwargs['smiles_list'], kwargs['labels']
    tokenizer = kwargs['tokenizer']
    train_indices, dev_indices, heldout_indices = kwargs[
        'train_indices'], kwargs['dev_indices'], kwargs['heldout_indices']
    batch_size, max_epochs = kwargs['batch_size'], kwargs['max_epochs']

    train_dataloader = get_ddp_dataloader(rank=rank, world_size=world_size,
                                          smiles_list=smiles_list,
                                          labels=labels,  indices=train_indices,
                                          tokenizer=tokenizer, batch_size=batch_size, shuffle=True)
    dev_dataloader = get_ddp_dataloader(rank=rank, world_size=world_size,
                                        smiles_list=smiles_list, labels=labels,  indices=dev_indices,
                                        tokenizer=tokenizer, batch_size=batch_size)

    model_kwargs = dict(tokenizer=tokenizer)

    model_hparams = dict(embedding_dim=128,
                         d_model=128,
                         num_layers=3,
                         bidirectional=True ,
                         dropout=0.2,
                         learning_rate=0.001,
                         weight_decay=0.0001)

    tb_writer = SummaryWriter('basic_runs', filename_suffix=f'rank{rank}')

    best_model, best_iteration = train_ddp(train_dataloader=train_dataloader,
                                           dev_dataloader=dev_dataloader,
                                           writer=tb_writer,
                                           max_epochs=max_epochs,
                                           device=device,
                                           model_class=RNNPredictor,
                                           model_args=tuple(),
                                           model_kwargs=model_kwargs,
                                           model_hparams=model_hparams)

    if rank == 0:
        test_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices,
                                         tokenizer=tokenizer, batch_size=batch_size)
        heldout_losses = []
        heldout_targets = []
        heldout_predictions = []
        loss_fn = nn.BCEWithLogitsLoss()
        for batch in test_dataloader:
            with torch.no_grad():
                sequence_batch, lengths, labels = batch
                logit_prediction = best_model(sequence_batch.to(best_model.device), lengths)
                loss = loss_fn(logit_prediction.squeeze(), labels.to(best_model.device))
                prob_predictions = torch.sigmoid(logit_prediction)
            heldout_losses.append(loss.item())
            heldout_targets.extend(labels.cpu().numpy())
            heldout_predictions.extend(prob_predictions.cpu().numpy())

        heldout_roc_auc = roc_auc_score(heldout_targets, heldout_predictions)

        tb_writer.add_scalar(
            'Loss/test', np.mean(heldout_losses), best_iteration)
        tb_writer.add_scalar(
            'ROC_AUC/test', np.mean(heldout_roc_auc), best_iteration)
        tb_writer.add_hparams(hparam_dict=model_hparams, metric_dict={
                              'hparam/roc_auc': heldout_roc_auc})
        print(f"Final test ROC AUC: {heldout_roc_auc}")
    tb_writer.close()


if __name__ == '__main__':
    main()
