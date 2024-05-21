#%% Imports
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal

from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('../')
import dkm

import math

import argparse
from timeit import default_timer as timer

from matplotlib import pyplot as plt

import pandas as pd
torch.set_num_threads(1)

#%% Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--data_folder_path', type=str, default="../../data/")
parser.add_argument('--device', type=str, nargs='?', default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--dtype', type=str, nargs='?', default='float32', choices=['float32', 'float64'])
parser.add_argument("--dof", type=float, default=0.001)
parser.add_argument("--init_lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--likelihood", type=str, default="categorical", choices=["gaussian","categorical"])
parser.add_argument("--final_layer", type=str, default="GAP", choices = ["GAP", "BFC"])
parser.add_argument("--n_ind_scale", type=int, default=1)
parser.add_argument("--mult_eps", type=float, default=1e-4)
parser.add_argument("--abs_eps", type=float, default=1e-4)
args = parser.parse_args()

#%% Set PyTorch Device and Dtype
device = torch.device(args.device)
dtype = getattr(torch, args.dtype)
torch.set_default_dtype(dtype)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
batch_size=4

#%% Set Experiment Parameters
dataset = args.dataset

dof = args.dof
init_lr = args.init_lr

final_layer_dict = {"GAP":dkm.NystromGAP, "BFC":dkm.BigFinalConvMixup}
Final_layer = final_layer_dict[args.final_layer]

likelihood = args.likelihood

n_ind_scale = args.n_ind_scale

print(f"dataset: {args.dataset}")
print(f"dtype: {args.dtype}")
print(f"dof: {args.dof}")
print(f"init_lr: {args.init_lr}")
print(f"seed: {args.seed}")
print(f"final_layer: {args.final_layer}")
print(f"likelihood: {args.likelihood}")
print(f"n_ind_scale: {args.n_ind_scale}",flush=True)

kwargs = {}
layer_kwargs = {**kwargs, 'sqrt' : dkm.chol_sqrt, 'MAP' : False}
sum_kwargs = {**kwargs}

#%% Create Dataloaders
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
import torchvision.transforms as T

datasets = {"CIFAR10":CIFAR10, "MNIST":MNIST, "CIFAR100":CIFAR100}
DATASET = datasets[args.dataset]

if args.dataset == "MNIST":
    cropsize=28
    normalise_transform = T.Normalize((0.1307,), (0.3081,))
elif args.dataset in ["CIFAR10","CIFAR100"]:
    cropsize=32
    normalise_transform =  T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_image_data = DATASET(args.data_folder_path, download=True, train=True, transform=T.Compose([T.RandomCrop(cropsize, padding=4), T.RandomHorizontalFlip(),T.ToTensor(),normalise_transform,]))
train_data_loader = torch.utils.data.DataLoader(train_image_data, batch_size=batch_size, shuffle=True)

test_image_data = DATASET(args.data_folder_path, download=True, train=False, transform=T.Compose([T.ToTensor(), normalise_transform,]))
test_data_loader = torch.utils.data.DataLoader(test_image_data, batch_size=batch_size, shuffle=True)

# %% ZCA Whitening Class for image data
# Inputs shape (N,C,W,H)
class ZCAWhiten(nn.Module):
    def __init__(self, X_train, epsilon=0.1):
        super().__init__()
        self.register_buffer("Xmean", X_train.mean((0, 2, 3))[None, :, None, None])
        self.register_buffer("Xstd", X_train.std((0, 2, 3))[None, :, None, None])
        Xnorm = (X_train - self.Xmean) / self.Xstd
        eigValX, eigVecX = torch.linalg.eig(Xnorm.reshape(Xnorm.shape[0], -1).t() @ Xnorm.reshape(Xnorm.shape[0], -1))
        self.register_buffer("W_zca", (
                    eigVecX @ (eigValX + epsilon * eigValX.mean()).reciprocal().sqrt().diag() @ eigVecX.t()).real)
        WXtrain = (self.W_zca @ Xnorm.reshape(Xnorm.shape[0], -1).t()).t().reshape(*X_train.shape)
        self.register_buffer("WXmean", WXtrain.mean((0, 2, 3))[None, :, None, None].real)
        self.register_buffer("WXstd", WXtrain.std((0, 2, 3))[None, :, None, None].real)

    def forward(self, X):
        Xnorm = (X - self.Xmean) / self.Xstd
        WX = (self.W_zca @ Xnorm.reshape(Xnorm.shape[0], -1).t()).t().reshape(*X.shape)
        return (WX - self.WXmean) / self.WXstd


#%% Create Model (initialising inducing patches from dataset)
n_ind = 5*n_ind_scale

if args.dataset in ["CIFAR10", "MNIST"]:
    no_classes = 10
elif args.dataset == "CIFAR100":
    no_classes = 100

#Initialise inducing patches at input layer from data
data_loader_full = torch.utils.data.DataLoader(train_image_data, batch_size=train_image_data.__len__(), shuffle=True)
X_full, Y_full = next(iter(data_loader_full))
by_label_indices = [Y_full == i for i in range(no_classes)]
sample_images = torch.concat([X_full[by_label_indices[i]][:50] for i in range(no_classes)])
patch_size = 3
patch_stride = 1
patches = sample_images.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride).permute(0,2,3,1,4,5).flatten(0,2)
patch_choices = torch.randint(patches.shape[0],(n_ind,))
x_ind = patches[patch_choices]
sample_labels = torch.concat([torch.Tensor([i]*(patches.shape[0] // no_classes)) for i in range(no_classes)])
patch_labels = sample_labels[patch_choices]

print(f"x_ind shape: {x_ind.shape}")
ind_learned=True
print(f"Inducing inputs learned: {ind_learned}",flush=True)


if args.final_layer == "GAP":
    final_layer_args = []
elif args.final_layer == "BFC":
    final_layer_args = [10*n_ind_scale,10*n_ind_scale]

#Defining the model
if dof == math.inf:
    model = nn.Sequential(ZCAWhiten(X_full),
                          dkm.Input(x_ind, learned=ind_learned),
                          dkm.ConvF2G(),
                          dkm.ReluKernel(mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          dkm.ConvMixup(n_ind,5*n_ind_scale,filter_size=3, stride=1),
                          dkm.GramLayer(5*n_ind_scale,args.dof),
                          #Spatial_bn(),
                          dkm.GramBatchNorm("none", "none", "none", "none"),
                          dkm.ResNetBlock(1,5*n_ind_scale,10*n_ind_scale,args.dof,3,2,mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          #Final_layer(*final_layer_args),
                          #dkm.NystromGAPMixup(10*n_ind_scale, 10*n_ind_scale),
                          #dkm.BigFinalConvMixup(10*n_ind_scale, 10*n_ind_scale),
                          #dkm.NystromGAP(),
                          dkm.BigFinalConv(),
                          dkm.ReluKernel(mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          dkm.Output(10*n_ind_scale, no_classes, **sum_kwargs),
                          )

if dof == math.inf:
    model = nn.Sequential(ZCAWhiten(X_full),
                          dkm.Input(x_ind, learned=ind_learned),
                          dkm.ConvF2G(),
                          dkm.ReluKernel(mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          dkm.ConvMixup(n_ind, 5 * n_ind_scale, filter_size=3, stride=1),
                          #dkm.GramLayer(5 * n_ind_scale, args.dof),
                          # Spatial_bn(),
                          dkm.GramBatchNorm("none", "none", "none", "none"),
                          dkm.noGram_ResNetBlock(1, 5 * n_ind_scale, 10 * n_ind_scale, args.dof, 3, 2, mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          # Final_layer(*final_layer_args),
                          # dkm.NystromGAPMixup(10*n_ind_scale, 10*n_ind_scale),
                          # dkm.BigFinalConvMixup(10*n_ind_scale, 10*n_ind_scale),
                          # dkm.NystromGAP(),
                          dkm.BigFinalConv(),
                          dkm.ReluKernel(mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          dkm.Output(10 * n_ind_scale, no_classes, **sum_kwargs),
                          )

else:
    model = nn.Sequential(ZCAWhiten(X_full),
                          dkm.Input(x_ind, learned=ind_learned),
                          dkm.ConvF2G(),
                          dkm.ReluKernel(mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          dkm.ConvMixup(n_ind, 5 * n_ind_scale, filter_size=3, stride=1),
                          dkm.GramLayer(5 * n_ind_scale, args.dof),
                          # Spatial_bn(),
                          dkm.GramBatchNorm("none", "none", "none", "none"),
                          dkm.ResNetBlock(1, 5 * n_ind_scale, 10 * n_ind_scale, args.dof, 3, 2, mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          # Final_layer(*final_layer_args),
                          # dkm.NystromGAPMixup(10*n_ind_scale, 10*n_ind_scale),
                          # dkm.BigFinalConvMixup(10*n_ind_scale, 10*n_ind_scale),
                          # dkm.NystromGAP(),
                          dkm.BigFinalConv(),
                          dkm.ReluKernel(mult_eps=args.mult_eps, abs_eps=args.abs_eps),
                          dkm.Output(10 * n_ind_scale, no_classes, **sum_kwargs),
                          )

#%% Training
model = model.to(device=device, dtype=dtype)
model(next(iter(train_data_loader))[0].to(device, dtype)) #To initialize the model
print(f"Model in CUDA: {next(model.parameters()).is_cuda}")

opt = torch.optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9,0.999))
previous_lr = opt.param_groups[0]['lr']
scheduler = ReduceLROnPlateau(opt, 'max', patience=4)
objs = []

#Training loop
for i in range(1):
    print(len(train_data_loader.sampler))
    #Forward pass, backwards pass, and training accuracy
    start = timer()
    train_obj = 0
    train_accs = []
    train_lls = []
    for X, Y in train_data_loader:
        X, Y = X.to(device, dtype), Y.to(device, dtype)

        pred = model(X)

        if args.likelihood == "gaussian":
            obj = dkm.gaussian_expectedloglikelihood(pred, torch.nn.functional.one_hot(Y.flatten().long(), num_classes=no_classes)) + dkm.norm_kl_reg(model, len(train_data_loader.sampler))
        elif args.likelihood == "categorical":
            obj = dkm.categorical_expectedloglikelihood(pred,Y.long()) + dkm.norm_kl_reg(model, len(train_data_loader.sampler))
        print(f"Batch obj: {obj}", flush=True)
        opt.zero_grad()
        (-obj).backward()
        opt.step()

        train_obj += obj.item()*X.size(0)

        if args.likelihood == "gaussian":
            train_preds = torch.argmax(dkm.gaussian_prediction(pred).loc, dim=1)
            train_lls.append(dkm.gaussian_prediction(pred).log_prob(torch.nn.functional.one_hot(Y.long(), num_classes=no_classes)).sum().item())
        elif args.likelihood == "categorical":
            train_preds = torch.argmax(dkm.categorical_prediction(pred).probs, dim=1)
            train_lls.append(dkm.categorical_prediction(pred).log_prob(Y).sum().item())
        train_accs.append((train_preds == Y.long()).sum().item())
    train_accuracy = torch.sum(torch.Tensor(train_accs)) / len(train_data_loader.sampler)
    train_ll = torch.sum(torch.Tensor(train_lls)) / len(train_data_loader.sampler)
    end = timer()

    mean_train_obj = train_obj / len(train_data_loader.sampler)
    objs.append(mean_train_obj)

    #Test accuracy and likelihood
    test_accs = []
    test_lls = []
    for x_test, y_test in test_data_loader:
        x_test, y_test = x_test.to(device, dtype), y_test.to(device, dtype)

        pred = model(x_test)

        if args.likelihood == "gaussian":
            test_preds = torch.argmax(dkm.gaussian_prediction(pred).loc, dim=1)
            test_lls.append(dkm.gaussian_prediction(pred).log_prob(torch.nn.functional.one_hot(y_test.long(), num_classes=no_classes)).sum().item())
        elif args.likelihood == "categorical":
            test_preds = torch.argmax(dkm.categorical_prediction(pred).probs, dim=1)
            test_lls.append(dkm.categorical_prediction(pred).log_prob(y_test).sum().item())
        test_accs.append((test_preds == y_test.long()).sum().item())
    test_accuracy = torch.sum(torch.Tensor(test_accs)) / len(test_data_loader.sampler)
    test_ll= torch.sum(torch.Tensor(test_lls)) / len(test_data_loader.sampler)
    print((f"Epoch {i}", f"Objective: {mean_train_obj}", f"Train Acc: {train_accuracy}", f"Test Acc: {test_accuracy}", f"Train LL: {train_ll}", f"Test LL: {test_ll}", f"Epoch Time (s): {end-start}"), flush=True)

    previous_obj=mean_train_obj
    scheduler.step(mean_train_obj)

    if opt.param_groups[0]['lr'] != previous_lr:
        print(f"LEARNING RATE HAS CHANGED TO {opt.param_groups[0]['lr']}")
        previous_lr = opt.param_groups[0]['lr']

#%% Final Test Accuracy and LL
train_accs = []
train_lls = []
train_objs = []
for x_train, y_train in train_data_loader:
    x_train, y_train = x_train.to(device, dtype), y_train.to(device, dtype)

    pred = model(x_train)
    if args.likelihood == "gaussian":
        obj = dkm.gaussian_expectedloglikelihood(pred, torch.nn.functional.one_hot(y_train.flatten().long(), num_classes=no_classes)) + dkm.norm_kl_reg(model, len(train_data_loader.sampler))
    elif args.likelihood == "categorical":
        obj = dkm.categorical_expectedloglikelihood(pred, y_train.long()) + dkm.norm_kl_reg(model, len(train_data_loader.sampler))

    if args.likelihood == "gaussian":
        train_preds = torch.argmax(dkm.gaussian_prediction(pred).loc, dim=1)
        train_lls.append(dkm.gaussian_prediction(pred).log_prob(torch.nn.functional.one_hot(y_train.long(), num_classes=no_classes)).sum().item())
    elif args.likelihood == "categorical":
        train_preds = torch.argmax(dkm.categorical_prediction(pred).probs, dim=1)
        train_lls.append(dkm.categorical_prediction(pred).log_prob(y_train).sum().item())
    train_accs.append((train_preds == y_train.long()).sum().item())
    train_objs.append(obj.item()*x_train.size(0))
train_accuracy = torch.sum(torch.Tensor(train_accs)) / len(train_data_loader.sampler)
train_ll = torch.sum(torch.Tensor(train_lls)) / len(train_data_loader.sampler)
train_obj = torch.sum(torch.Tensor(train_objs)) / len(train_data_loader.sampler)

test_accs = []
test_lls = []
for x_test, y_test in test_data_loader:
    x_test, y_test = x_test.to(device, dtype), y_test.to(device, dtype)

    pred = model(x_test)

    if args.likelihood == "gaussian":
        test_preds = torch.argmax(dkm.gaussian_prediction(pred).loc, dim=1)
        test_lls.append(dkm.gaussian_prediction(pred).log_prob(torch.nn.functional.one_hot(y_test.long(), num_classes=no_classes)).sum().item())
    elif args.likelihood == "categorical":
        test_preds = torch.argmax(dkm.categorical_prediction(pred).probs, dim=1)
        test_lls.append(dkm.categorical_prediction(pred).log_prob(y_test).sum().item())
    test_accs.append((test_preds == y_test.long()).sum().item())
test_accuracy = torch.sum(torch.Tensor(test_accs)) / len(test_data_loader.sampler)
test_ll= torch.sum(torch.Tensor(test_lls)) / len(test_data_loader.sampler)

print("(Pound symbols for easy extraction of metrics)")
print(f"Final Train Objective: £{train_obj}")
print(f"Final Train Accuracy: £{train_accuracy}")
print(f"Final Train LL: £{train_ll}")
print(f"Final Test Accuracy: £{test_accuracy}")
print(f"Final Test LL: £{test_ll}")