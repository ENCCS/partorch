Training a Neural Network in parallel using DistributedDataParallel
===================================================================

We will use an example of a simple recurrent neural network for sequence 
classification and how we can modify this to use the DistributedDataParallel 
feature of PyTorch. You can find the code we will be using 
:download:`here <_static/code_archive.zip>`.

We will mainly work with the file :code:`scripts/basic_neural_network.py`. The 
important part from the file is replicated below:

.. code-block:: python
    :emphasize-lines: 8-13,25-26

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_seed)
    for visible_index, heldout_indices in skf.split(smiles_list, labels):
        tb_writer = SummaryWriter('basic_runs')
        
        visible_labels = [labels[i] for i in visible_index]
        train_indices, dev_indices = train_test_split(visible_index, stratify=visible_labels, shuffle=True, test_size=0.2, random_state=args.random_seed)
        
        train_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=train_indices,
                                          tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        dev_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=dev_indices,
                                        tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
        heldout_dataloader = get_dataloader(smiles_list=smiles_list, labels=labels,  indices=heldout_indices,
                                            tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)

        model_kwargs = dict(tokenizer=tokenizer, device=device)

        model_hparams = dict(embedding_dim=128,
                             d_model=128,
                             num_layers=3,
                             bidirectional=True,
                             dropout=0.2,
                             learning_rate=0.001,
                             weight_decay=0.0001)
        
        heldout_roc_auc = train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, test_dataloader=heldout_dataloader, writer=tb_writer,
                                max_epochs=max_epochs, model_class=RNNPredictor, model_args=tuple(), model_kwargs=model_kwargs, model_hparams=model_hparams)

        tb_writer.close()


The highlighted lines show the parts we will focus on. These are the ones which we need to take into 
account when adding the parallelization.

DistributedDataParallel
-----------------------

Before we start wrapping our models and datasets in the parallel framework, we need to initialize the 
:code:`pytorch.distributed` runtime. We do this by calling :code:`torch.distributed.init_process_group()`. This 
function is responsible for setting up the underlying process groups which the work will be distributed over.

DistributedDataParallel relies on one process per worker, and if GPUs are used, each worker needs 
exclusive access to its GPUs. Each process maintains its own copy of the model and optimizer, while 
gradients are synchronized over the whole process group.

Each process also maintains its own python interpreter, which gets arround issues with the Global Interpreter Lock 
(GIL) which hampers python performence when using multi-threading.