# Hyper Parameter Optimization workshop example #

Package with code examples for the ENCCS hyper parameter optimization workshop.

## Install ##
Anaconda is used to manage python dependencies and this package assumes it's what you are using.

First create an envirnoment by running:
```shell
conda env create -f environment.yml
```

This creates environments called `hpo_workshop`. Now we have to install *this* package in  this environment so all code can be found:

```shell
conda activate hpo_workshop
pip install -e .
```
This creates a symlinked version of this python package in this repository to the environment which means that you will be able to refer to the package `hpo_workshop`.

