Parallel Training Backround
===========================

With increasing amounts of compute in the form of compute 
clusters, there's a need to train models in parallel.

When training pytorch models, there's essentially three ways of utilizing this parallelism in increasing engineering complexity:

1. Parallelizing experiments: when performing cross-validation or hyper parameter 
optimization, each experiment can essentially be run in parallel.  While efficient 
HP optimization can require communication between jobs, this is outside pytorch responsibility.

2. Parallelizing over batches (data-parallel): The examples of a batch can be spread over multiple GPUs/nodes. This requires the workers to synchronize during optimization to maintain the same model in each process.

3. Parallelizing over model modules (model-parallel): The different modules (layers) of a model can be split over multiple GPUs. This requires the processes to synchronize during both the forward and backward propagation as well as optimization.