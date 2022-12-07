import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

import torch
from torch.utils.tensorboard import SummaryWriter

from partorch.tokenizer import BytesTokenizer
from partorch.dataset import get_dataloader
from partorch.training_loop import train
from partorch.rnn import RNNPredictor

def main():
    parser = argparse.ArgumentParser(description='Basic Hyper Parameter optimization example')
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--random-seed', help="The random seed to use", type=int, default=1729)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    labels = df['p_np'].tolist()
    smiles_list = df['smiles'].tolist()

    tokenizer = BytesTokenizer()
    
    device = torch.device('cuda')

    max_epochs=5
    batch_size = 256
    num_workers = 4
    
    torch.random.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    visible_index, heldout_indices = train_test_split(np.arange(len(
        smiles_list)), stratify=labels, shuffle=True, test_size=0.1, random_state=args.random_seed)
    
    visible_labels = [labels[i] for i in visible_index]
    train_indices, dev_indices = train_test_split(visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)
    
    train_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=train_indices,
                                        tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices,
                                    tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
    model_kwargs = dict(tokenizer=tokenizer, device=device)

    model_hparams = dict(embedding_dim=128,
                            d_model=128,
                            num_layers=3,
                            bidirectional=True,
                            dropout=0.2,
                            learning_rate=0.001,
                            weight_decay=0.0001)
    
    tb_writer = SummaryWriter('basic_runs')
    best_model, best_iteration = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, writer=tb_writer,
                                max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=model_hparams)
    
    heldout_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices,
                                        tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)

    heldout_losses = []
    heldout_targets = []
    heldout_predictions = []
    for batch in heldout_dataloader:
        loss, batch_targets, batch_predictions = best_model.eval_and_predict_batch(
            batch)
        heldout_losses.append(loss)
        heldout_targets.extend(batch_targets)
        heldout_predictions.extend(batch_predictions)

    heldout_roc_auc = roc_auc_score(heldout_targets, heldout_predictions)

    tb_writer.add_scalar(
        'Loss/test', np.mean(heldout_losses), best_iteration)
    tb_writer.add_scalar(
        'ROC_AUC/test', np.mean(heldout_roc_auc), best_iteration)
    tb_writer.add_hparams(hparam_dict=model_hparams, metric_dict={
                            'hparam/roc_auc': heldout_roc_auc})
    print(f"Final test ROC AUC: {heldout_roc_auc}")
    tb_writer.close()
            
    args = parser.parse_args()


if __name__ == '__main__':
    main()