# Stochastic Kernel Regularisation Improves Generalisation in Deep Kernel Machines
This repository contains code to reproduce the experiments in the paper "Stochastic Kernel Regularisation Improves Generalisation in Deep Kernel Machines".

An `environment.yml` file is included to replicate our conda environment.

### CIFAR-10 Experiments
All CIFAR-10 experiments were run with 4 different random seeds.

The folder `nn_experiments/kuanglui_pytorch-cifar/` contains the code we used (a modified version of the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) GitHub repo) to run our neural network experiments in the headline performance table. The script `nn_experiments.sh` shows how to run the experiments exactly as we did. The 8 `nn_<sgd,adam>£<seed>.o` files are the raw output logs from our runs, showing metrics at each epoch.

The folder `dkm_most_experiments/` contains the code we used to run most the CIFAR-10 DKM experiments (with the exception of one the ablations where we modified the library directly for convenience, which we include as a separate folder). Inside are subfolders `headline_numbers/` and `<ablation>_ablation/` which contain bash scripts to show how to run the experiments, alongside the raw `.o` output logs for the experiments we ran. We have named the folders so that the specific experiment / ablation it refers to should be clear alongside the tables in the paper.

The folder `dkm_no_skr_but_still_jitter_ablation/` is similar to `dkm_most_experiments/` but only contains a single ablation subfolder. This is because for this ablation we modified the library directly for convenience, and so running this ablation requires slightly different code to the other ablations.

CIFAR-10 experiments that run for the full 1200 epochs will take about 4 days each on an NVIDIA A100 GPU.