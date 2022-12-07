from copy import deepcopy

from sklearn.metrics import roc_auc_score
from tqdm import trange, tqdm
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

def train(*, train_dataloader, dev_dataloader, test_dataloader, writer, max_epochs, model_class, model_args=None, model_hparams=None, model_kwargs=None):
    if model_args is None:
        model_args = tuple()
    if model_kwargs is None:
        model_kwargs = dict()
    if model_hparams is None:
        model_hparams = dict()
    
    model = model_class(*model_args, **model_kwargs, **model_hparams)

    best_roc_auc = 0
    best_model = None
    best_iteration = 0
    iteration = 0
    for e in trange(max_epochs, desc='epoch'):
        training_losses = []
        dev_losses = []
        dev_targets = []
        dev_predictions = []

        for batch in tqdm(train_dataloader, desc="Training batch"):
            loss = model.train_batch(batch)
            writer.add_scalar('Loss/train', loss, iteration)
            training_losses.append(loss)
            iteration += 1


        for batch in tqdm(dev_dataloader, desc="Dev batch"):
            loss, batch_targets, batch_predictions = model.eval_and_predict_batch(batch)
            dev_losses.append(loss)
            dev_targets.extend(batch_targets)
            dev_predictions.extend(batch_predictions)
        
        dev_roc_auc = roc_auc_score(dev_targets, dev_predictions)

        writer.add_scalar('Loss/dev', np.mean(dev_losses), iteration)
        writer.add_scalar('ROC_AUC/dev', dev_roc_auc, iteration)
        print(f"Training loss {np.mean(training_losses)}\tDev loss: {np.mean(dev_losses)}\tDev ROC AUC:{dev_roc_auc}")
        
        if dev_roc_auc > best_roc_auc:
            best_roc_auc = dev_roc_auc
            best_model = deepcopy(model)
            best_model.recurrent_layers.flatten_parameters()  # After the deepcopy, the weight matrices are not necessarily in contiguous memory, this fixes that issue
            best_iteration = iteration

    heldout_losses = []
    heldout_targets = []
    heldout_predictions = []

    for batch in tqdm(test_dataloader, desc='Test batch'):
        loss, batch_targets, batch_predictions = best_model.eval_and_predict_batch(batch)
        heldout_losses.append(loss)
        heldout_targets.extend(batch_targets)
        heldout_predictions.extend(batch_predictions)
    heldout_roc_auc = roc_auc_score(heldout_targets, heldout_predictions)

    writer.add_scalar('Loss/test', np.mean(heldout_losses), best_iteration)
    writer.add_scalar('ROC_AUC/test', np.mean(heldout_roc_auc), best_iteration)
    writer.add_hparams(hparam_dict=model_hparams, metric_dict={'hparam/roc_auc': heldout_roc_auc})
    return heldout_roc_auc


def train_ddp(*, rank, world_size, train_dataloader, dev_dataloader, test_dataloader, writer, max_epochs, model_class, device, model_args=None, model_hparams=None, model_kwargs=None):
    if model_args is None:
        model_args = tuple()
    if model_kwargs is None:
        model_kwargs = dict()
    if model_hparams is None:
        model_hparams = dict()
    
    model = model_class(*model_args, **model_kwargs, device=device, **model_hparams)
    ddp_model = DDP(model)
    ddp_model.set_optimizer(**model_hparams)
    
    best_roc_auc = 0
    best_model = None
    best_iteration = 0
    iteration = 0
    for e in trange(max_epochs, desc='epoch'):
        training_losses = []
        dev_losses = []
        dev_targets = []
        dev_predictions = []

        for batch in tqdm(train_dataloader, desc="Training batch"):
            loss = ddp_model.train_batch(batch)
            writer.add_scalar('Loss/train', loss, iteration)
            training_losses.append(loss)
            iteration += 1

        for batch in tqdm(dev_dataloader, desc="Dev batch"):
            loss, batch_targets, batch_predictions = model.eval_and_predict_batch(batch)
            dev_losses.append(loss)
            dev_targets.extend(batch_targets)
            dev_predictions.extend(batch_predictions)
        
        dev_roc_auc = roc_auc_score(dev_targets, dev_predictions)

        writer.add_scalar('Loss/dev', np.mean(dev_losses), iteration)
        writer.add_scalar('ROC_AUC/dev', dev_roc_auc, iteration)
        print(f"Training loss {np.mean(training_losses)}\tDev loss: {np.mean(dev_losses)}\tDev ROC AUC:{dev_roc_auc}")
        
        if dev_roc_auc > best_roc_auc:
            best_roc_auc = dev_roc_auc
            best_model = deepcopy(model)
            best_model.recurrent_layers.flatten_parameters()  # After the deepcopy, the weight matrices are not necessarily in contiguous memory, this fixes that issue
            best_iteration = iteration

    heldout_losses = []
    heldout_targets = []
    heldout_predictions = []

    for batch in tqdm(test_dataloader, desc='Test batch'):
        loss, batch_targets, batch_predictions = best_model.eval_and_predict_batch(batch)
        heldout_losses.append(loss)
        heldout_targets.extend(batch_targets)
        heldout_predictions.extend(batch_predictions)
    heldout_roc_auc = roc_auc_score(heldout_targets, heldout_predictions)

    writer.add_scalar('Loss/test', np.mean(heldout_losses), best_iteration)
    writer.add_scalar('ROC_AUC/test', np.mean(heldout_roc_auc), best_iteration)
    writer.add_hparams(hparam_dict=model_hparams, metric_dict={'hparam/roc_auc': heldout_roc_auc})
    return heldout_roc_auc