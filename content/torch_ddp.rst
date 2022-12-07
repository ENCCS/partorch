Training a Neural Network in parallel using DistributedDataParallel
===================================================================

We will use an example of a simple recurrent neural network for sequence 
classification and how we can modify this to use the DistributedDataParallel 
feature of PyTorch. You can find the code we will be using 
:download:`here <_static/code_archive.zip>`.

We will mainly work with the file :code:`scripts/basic_neural_network.py`. The 
important part from the file is replicated below:

.. code-block:: python
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


We need to define the :code:`distributed_training()` function and start with something like this:

.. code-block:: python
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
    $ torchrun --master_port 331133 scripts/basic_neural_network_ddp.py dataset/BBBP.csv

This starts the script with some underlying environmental variables set which allows the process group 
to coordinate, in particular we tell it to select a random port for the master process (the 0 argument 
to --master_port). We might need to set this port to  different values if we're running multiple 
parallel training at the same time.


def setup(rank, world_size):
    """Setup the"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
)

Before we start wrapping our models and datasets in the parallel framework, we need to initialize the 
:code:`pytorch.distributed` runtime. We do this by calling :code:`torch.distributed.init_process_group()`. This 
function is responsible for setting up the underlying process groups which the work will be distributed over.

DistributedDataParallel relies on one process per worker, and if GPUs are used, each worker needs 
exclusive access to its GPUs. Each process maintains its own copy of the model and optimizer, while 
gradients are synchronized over the whole process group.

Each process also maintains its own python interpreter, which gets arround issues with the Global Interpreter Lock 
(GIL) which hampers python performence when using multi-threading.