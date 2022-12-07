Training a Neural Network in parallel using DistributedDataParallel
===================================================================

We will use an example of a simple recurrent neural network for sequence 
classification and how we can modify this to use the DistributedDataParallel 
feature of PyTorch. You can find the code we will be using 
:download:`here <_static/code_archive.zip>`.

We will mainly work with the file :code:`scripts/basic_neural_network.py`. The 
important part from the file is replicated below:

.. code-block:: python
    :linenos:
    :emphasize-lines: 8-12,23-24

    visible_index, heldout_indices = train_test_split(np.arange(len(
        smiles_list)), stratify=labels, shuffle=True, test_size=0.1, random_state=args.random_seed)

    visible_labels = [labels[i] for i in visible_index]
    train_indices, dev_indices = train_test_split(
        visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)

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

    tb_writer.add_scalar('Loss/test', np.mean(heldout_losses), best_iteration)
    tb_writer.add_scalar('ROC_AUC/test', np.mean(heldout_roc_auc), best_iteration)
    tb_writer.add_hparams(hparam_dict=model_hparams, metric_dict={'hparam/roc_auc': heldout_roc_auc})
    print(f"Final test ROC AUC: {heldout_roc_auc}")
    tb_writer.close()
            


The highlighted lines show the parts we will focus on. These are the ones which we need to take into 
account when adding the parallelization.

Parallel semantics
------------------

Our parallel neural network will consist of multiple process running 
concurrently. These will be spawned from our main process but will 
execute the same code. To make the different processes work on 
different parts of the data, we differentiate them through an 
identifier called *rank*. We often wan't to perform some step only 
once for the whole group, so it's customary that we assign one of
 the ranks a special importance, for convenience this is typically 
 chosen to be the process with rank 0.


Initializing the distributed framework
--------------------------------------

We start by adding distributed functionality, the code we want to execute in 
parallel is wrapped in a function, here called :code:`distributed_training()`, 
which will be the entry point for all spawned processes. We use pytorch's 
:code:`multiprocessing` package to spawn processes with this function as a 
target. We also create a dictionary with all the arguments our training function will need.
The function will be supplied with the rank of the process from the 
:code:`torch.multiprocessing.spawn()` function, but we also supply the 
total size of the process group for convenience.

.. code-block:: python
    :linenos:
    :emphasize-lines: 7-15 

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


The distributed training
------------------------

We need to define the :code:`distributed_training()` function and start with something like this:

.. code-block:: python
    :linenos:
    :emphasize-lines: 2-3, 5

    def distributed_training(rank, world_size, kwargs):
            dist.init_process_group(
                kwargs['backend'], rank=rank, world_size=world_size)

            device = torch.device(f'cuda:{rank}')

            smiles_list, labels = kwargs['smiles_list'], kwargs['labels']
            tokenizer = kwargs['tokenizer']
            train_indices, dev_indices, heldout_indices = kwargs[
                'train_indices'], kwargs['dev_indices'], kwargs['heldout_indices']
            batch_size, max_epochs = kwargs['batch_size'], kwargs['max_epochs']

Most of this code is just unpacking the parameters we gave in the :code:`kwargs` argument, 
but the vital part is the call to :code:`dist.init_process_group()`. This is what actually 
sets up the current process as part of the process group. There's a lot of machinery 
beneath this which we will not cover in this workshop.

One important question is how pytorch should communicate between the processes, 
and the call to :code:`init_process_group`` is where we specify this. There are 
multiple backends which can be used for the interprocess communication, but the 
recommended one when training on multiple GPUs is 'nccl', which is developed by 
NVIDIA, and is what we'll use in this workshop.

We also set the device at this point. A GPU may only be used by one process, here 
we instantiate a device reference using the rank of the process. If you need to 
limit your program to only use a subset of your GPUs, you can set the environmental variable
:code:`CUDA_VISIBLE_GPUS=id1[,id2]` before starting the script.

To simplify setting up the underlying process group, pytorch supplies a convenience script 
:code:`torchrun` which can be used to inform the backend where the master process is located 
which is used to coordinate the processes.

We can test our script by running:

.. code-block:: shell

    $ torchrun --master_port 31133 scripts/basic_neural_network_ddp.py dataset/BBBP.csv


This starts the script with some underlying environmental variables set which allows the process group 
to coordinate, in particular we tell it to use a specific port for the master process (the arbitrary 31133 argument 
to --master_port). We might need to set this port to  different values if we're running multiple 
parallel training at the same time.

We can also use :code:`torchrun` to manually spawn multiple processes at different compute nodes, in 
that case we also tell the program at what IP adress to find our master node by suppliying a 
:code:`--master_addr` argument.

Now we're ready to implement more of :code:`distributed_training()`. The main goal of our data-parallel training 
is to let the different processes work on different parts of the batch. This means that we need to
partition our data based on what process is running the code. Here's the outline of what we'll 
implement next:

.. code-block:: python
    :linenos:
    :emphasize-lines: 12-20, 34-42, 44-69

    def distributed_training(rank, world_size, kwargs):
        dist.init_process_group(kwargs['backend'], rank=rank, world_size=world_size)

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
    dev_dataloader = None
    if rank == 0:
        # We will only do the evaluations on the rank 0 process, so we don't have to pass predictions around
        dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices,
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

We will go through these three highlighted block in order. 

Distributed data loaders
------------------------

First we will have a look at :code:`get_ddp_dataloader` next to :code:`get_dataloader`: 

.. code-block:: python
    :linenos:
    :emphasize-lines: 6, 14-15

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

Conveniently, pytorch already has the functionality we need to 
split our batches in a distributed setting. By telling the 
:code:`DataLoader` to use a :code:`DistributedSampler` with appropriate arguments 
for *rank* and *world* size, the dataloader instantiated in the current 
process will get its own dedicated part of the dataset to work on.

Distributed optimization
------------------------

Now that we've set up partitioned data loaders in the different processes, we will register our model with the :code:`DistributedDataParallel` so that our optimization will be distributed over our processes.
Let's have a look at the old training vs. updated training loop:

.. code-block:: python
    :linenos:
    :emphasize-lines: 9,18,26,30,39,43

    def train(*, train_dataloader, dev_dataloader, writer, max_epochs, model_class, model_args=None, model_hparams=None, model_kwargs=None):
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

        learning_rate = model_hparams['learning_rate']
        weight_decay = model_hparams['weight_decay']
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        for e in trange(max_epochs, desc='epoch'):
            training_losses = []
            dev_losses = []
            dev_targets = []
            dev_predictions = []

            model.train()
            for batch in tqdm(train_dataloader, desc="Training batch"):
                optimizer.zero_grad()
                sequence_batch, lengths, labels = batch
                logit_prediction = model(sequence_batch.to(model.device), lengths)
                loss = loss_fn(logit_prediction.squeeze(), labels.to(model.device))
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/train', loss.item(), iteration)
                training_losses.append(loss.item())
                iteration += 1

            model.eval()
            for batch in tqdm(dev_dataloader, desc="Dev batch"):
                with torch.no_grad():
                    sequence_batch, lengths, labels = batch
                    logit_prediction = model(sequence_batch.to(model.device), lengths)
                    loss = loss_fn(logit_prediction.squeeze(), labels.to(model.device))
                    prob_predictions = torch.sigmoid(logit_prediction)
                
                dev_losses.append(loss.item())
                dev_targets.extend(labels.cpu().numpy())
                dev_predictions.extend(prob_predictions.cpu().numpy())
            
            dev_roc_auc = roc_auc_score(dev_targets, dev_predictions)

            writer.add_scalar('Loss/dev', np.mean(dev_losses), iteration)
            writer.add_scalar('ROC_AUC/dev', dev_roc_auc, iteration)
            print(f"Training loss {np.mean(training_losses)}\tDev loss: {np.mean(dev_losses)}\tDev ROC AUC:{dev_roc_auc}")
            
            if dev_roc_auc > best_roc_auc:
                best_roc_auc = dev_roc_auc
                best_model = deepcopy(model)
                best_model.recurrent_layers.flatten_parameters()  # After the deepcopy, the weight matrices are not necessarily in contiguous memory, this fixes that issue
                best_iteration = iteration

        return best_model, best_iteration

.. code-block:: python
    :linenos:
    :emphasize-lines: 9-10, 19, 27, 31, 40, 45

    def train_ddp(*, train_dataloader, dev_dataloader, writer, max_epochs, model_class, device, model_args=None, model_hparams=None, model_kwargs=None):
        if model_args is None:
            model_args = tuple()
        if model_kwargs is None:
            model_kwargs = dict()
        if model_hparams is None:
            model_hparams = dict()
        
        model = model_class(*model_args, **model_kwargs, device=device, **model_hparams)
        ddp_model = DistributedDataParallel(model)
        
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

            if dist.get_rank() == 0:
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
            
                if dev_roc_auc > best_roc_auc:
                    best_roc_auc = dev_roc_auc
                    best_model = deepcopy(model)
                    best_model.recurrent_layers.flatten_parameters()  # After the deepcopy, the weight matrices are not necessarily in contiguous memory, this fixes that issue
                    best_iteration = iteration

    return best_model, best_iteration

If you compare the two code parts, you can see that we're basically just wrapping 
our model in an :code:`DistributedDataParallel` object, which gives us a new model 
we call :code:`ddp_model`.
We subsequently replace the calls to :code:`model` with :code:`ddp_model` which 
is all we need to do. The optimizer will do the right thing, synchronizing the 
gradients across worker processes, through it's reference 
to :code:`ddp_model.parameters()`.

Centralizing evaluation
-----------------------

Note that we only run the evaluation on the dev set and update the :code:`best_model` copy 
at the process with rank=0. The reason for this is that we don't want to 
have to send results from the predictions around.

This is also what we do in the final block of the :code:`distributed_training` function, 
we only perform the final test set evaluation at the process with rank 0.


Running the code
----------------

We have now completed our augmentation of the model and can run it using :code:`torchrun`:

.. code-block:: shell

    $ torchrun --master_port 31133 scripts/basic_neural_network_ddp.py dataset/BBBP.csv
