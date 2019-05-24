# HIU-SAC
> Pytorch implementation of Hierarchical Intentional-Unintentional
Soft Actor-Critic (HIU-SAC) algorithm

This repository contains the source code used for the experiments conducted
in the paper:
<a href="https://arxiv.org/pdf/1905.09668.pdf" target="_blank">
*Hierarchical reinforcement learning for concurrent discovery of compound and 
composable policies*</a>

The algorithm has been tested on continuous control tasks in 
<a href="https://github.com/domingoesteban/robolearn_envs" target="_blank">
RoboLearn environments</a>. 

Some videos can be found at <a href="https://sites.google.com/view/hrl-concurrent-discovery" target="_blank">
*https://sites.google.com/view/hrl-concurrent-discovery*</a>

The code has been tested with PyTorch 1.0.1 and Python 3.5 (or later).

## Pre-Installation
It is recommended to first create either a virtualenv or a conda environment.
- **Option 1: Conda environment**. First install either miniconda (recommended) or anaconda. 
[Installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
```bash
# Create the conda environment
conda create -n <condaenv_name> python=3.5
# Activate the conda environment
conda activate <condaenv_name>

```

- **Option 2: Virtualenv**. First install pip and virtualenv. 
[Installation instructions](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/)
```bash
# Create the virtual environment
virtualenv -p python3.5 <virtualenv_name>
# Activate the virtual environment
source <virtualenv_name>/bin/activate
```

## Installation
1. Clone this repository
```bash
git clone https://github.com/domingoesteban/hiu_sac
```

2. Install the requirements of this repository
```bash
cd hiu_sac
pip install -r requirements.txt
```

## Use

- Run HIU-SAC in one of the environments. Options: navigation2d, reacher, pusher, centauro

```bash
# python train.py -e <env_name>
python train.py -e navigation2d
```

- Visualize the learned policy (Specify the log directory that is printed during the learning process)

```bash
python eval.py <path_to_log_directory>
```

- Plot the learning curves in the composable and compound tasks (Specify the log directory that is printed during the learning process)

```bash
python eval.py <path_to_log_directory> -p
```
