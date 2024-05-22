#%% Imports
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal

from torch.profiler import profile, record_function, ProfilerActivity

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from dkm import dkm

from sklearn.svm import SVC

import argparse
from timeit import default_timer as timer

from matplotlib import pyplot as plt

import pandas as pd

import psutil

import numpy as np

import math

torch.set_num_threads(4)

#%% UCI Data
# uci = UCI("wine", 2, dtype=getattr(torch,"float64"), device=device)
# x_train = uci.X_train
# y_train = uci.Y_train
# x_test = uci.X_test_norm
# y_test = uci.Y_test_unnorm
# no_of_data_points = x_train.shape[0]
# input_features = x_train.shape[1]


#%% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_path', type=str, default="./data/")
parser.add_argument('--datasets_list', type=str, default="data/datasets.txt")
parser.add_argument('--device', type=str, nargs='?', default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--dtype', type=str, nargs='?', default='float64', choices=['float32', 'float64'])
parser.add_argument("--dof", type=float, default=0.)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--save_name", type=str)
parser.add_argument("--s2_learned", type=bool, default=False)
parser.add_argument("--likelihood", type=str, default="categorical", choices=["gaussian","categorical"])
args = parser.parse_args()


#%% Set PyTorch Device and Dtype
device = torch.device(args.device)
dtype = getattr(torch, args.dtype)
torch.set_default_dtype(dtype)
if args.dtype == "float64":
    torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
#batch_size=256
batch_size = 256

# n_epochs =10001
n_epochs = 2001

ncol_divs = [0,1, 10, 100]
runs_epochs_condition_numbers = []
for i in range(len(ncol_divs)):
    torch.manual_seed(i)
    #%% Toy Data Classification
    n_classes = 2
    no_of_inducing_points = 100
    no_of_data_points =356
    xs_0 = torch.linspace(-2,2,no_of_data_points//2) + torch.randn(no_of_data_points//2)*0.1
    xs_1 = torch.linspace(-2,2,no_of_data_points//2) + torch.randn(no_of_data_points//2)*0.1
    xs_0 = torch.stack([xs_0, torch.sin(xs_0*3)*6+torch.randn(no_of_data_points//2)], dim=1)
    xs_1 = torch.stack([xs_1, torch.sin(xs_1*3)*6+torch.randn(no_of_data_points//2)+5], dim=1)
    toy_classifier_xs_class_0 = xs_0
    toy_classifier_xs_class_1 = xs_1

    shuffle_perm = torch.randperm(no_of_data_points)
    toy_classif_xs = torch.cat([toy_classifier_xs_class_0, toy_classifier_xs_class_1])[shuffle_perm]
    toy_classif_ys = torch.cat([torch.zeros(no_of_data_points//2), torch.ones(no_of_data_points//2)])[shuffle_perm]#.resize(no_of_data_points,1)

    #plt.scatter(toy_classif_xs[:,0], toy_classif_xs[:,1],c=toy_classif_ys)

    xs_0_test = torch.linspace(-2,2,no_of_data_points//4) + torch.randn(no_of_data_points//4)*0.1
    xs_1_test = torch.linspace(-2,2,no_of_data_points//4) + torch.randn(no_of_data_points//4)*0.1
    xs_0_test = torch.stack([xs_0_test, torch.sin(xs_0_test*3)*6+torch.randn(no_of_data_points//4)], dim=1)
    xs_1_test = torch.stack([xs_1_test, torch.sin(xs_1_test*3)*6+torch.randn(no_of_data_points//4)+5], dim=1)
    toy_classifier_xs_class_0_test = xs_0_test
    toy_classifier_xs_class_1_test = xs_1_test
    shuffle_perm2 = torch.randperm(no_of_data_points//2)
    toy_classif_xs_test = torch.cat([toy_classifier_xs_class_0_test, toy_classifier_xs_class_1_test])[shuffle_perm2]
    toy_classif_ys_test = torch.cat([torch.zeros(no_of_data_points//4), torch.ones(no_of_data_points//4)])[shuffle_perm2]#.resize(no_of_data_points//2,1)

    x_full =  (toy_classif_xs - toy_classif_xs.mean(dim=0)) / toy_classif_xs.std(dim=0)
    x_ind = x_full[:no_of_inducing_points,:]
    x_train = x_full[no_of_inducing_points:,:]


    y_full = toy_classif_ys
    y_ind = y_full[:no_of_inducing_points]
    y_train = y_full[no_of_inducing_points:]

    x_test = (toy_classif_xs_test - toy_classif_xs.mean(dim=0)) / toy_classif_xs.std(dim=0)
    y_test = toy_classif_ys_test

    input_features = x_train.shape[1]

    torch.manual_seed(args.seed)


    #%% No_DKM New Output
    kwargs = {}
    layer_kwargs = {**kwargs, 'sqrt' : dkm.sym_sqrt, 'MAP' : False}
    #sum_kwargs = {**kwargs, 'noise_var_learned' : args.s2_learned, 'likelihood' : args.likelihood}
    sum_kwargs = {**kwargs}

    x_ind_copy = torch.clone(x_ind)

    ncol_div = ncol_divs[i]

    model = nn.Sequential(dkm.Input(x_ind, learned=True),
                          dkm.F2G(),
                          dkm.SqExpKernel(),
                          dkm.GramLayer(x_ind.shape[0], args.dof, ncol_div=ncol_div, taylor_objective=False),
                          dkm.SqExpKernel(),
                          dkm.Output(x_ind.shape[0], 2, mc_samples=100, init_mu=torch.nn.functional.one_hot(y_ind.long(), num_classes=2).to(dtype=dtype), **sum_kwargs),
                          )

    print(f"Model Architecture: {model}")
    print(f"s2 learned: {args.s2_learned}")
    print(f"Random Seed: {args.seed}")
    print(f"Initial learning rate: {args.lr}")
    print(f"dof: {args.dof}")

    # model[-1]._mu.data = torch.nn.functional.one_hot(y_ind.long(), num_classes=2).double()
    # y_ind_old = torch.clone(y_ind)
    # y_ind = model[-1].mu.data.argmax(dim=1).clone()

    model = model.to(device=device)

    from torch.utils import checkpoint

    print("CUDA Checks")
    print(f"Model: {next(model.parameters()).is_cuda}")
    x_train = x_train.to(device)
    y_train = y_train.to(device=device, dtype=dtype)
    x_test = x_test.to(device=device, dtype=dtype)
    y_test = y_test.to(device=device, dtype=dtype)
    print(f"x_train: {x_train.is_cuda}")
    print(f"y_train: {y_train.is_cuda}")
    print(f"x_test: {x_test.is_cuda}")
    print(f"y_test: {y_test.is_cuda}")

    times = []
    lls = []
    rmses = []
    objs = []
    # print(f"x_train shape: {x_train.shape}")
    # print(f"y_train shape: {y_train.shape}")


    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    if args.save_name:
        torch.save(model, "temp_prior_model_" + args.save_name)
    previous_obj = -float("inf")
    model.eval()
    model(next(iter(trainloader))[0].to(device, dtype))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.8,0.9))
    previous_lr = opt.param_groups[0]['lr']


    # Define milestones and learning rates
    milestones = [10000000]  # Epochs where learning rate will change
    gammas = [0.1]  # Factors by which the learning rate will be multiplied
    #gammas=[1,1]
    # Create the scheduler
    scheduler = MultiStepLR(opt, milestones=milestones, gamma=gammas[0])  # Assuming gamma is the same for each step for simplicity
    #use cosine annealing
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=1e-4)

    def print_grad_norms(model):
        print("Gradient norms:")
        for name, param in model.named_parameters():
            if 'V' in name:
                grad_norm = param.grad.norm().item()
                print(f"{name}: {grad_norm}")

    gradient_norms = {}
    def save_grad_norms(model):
        for name, param in model.named_parameters():
            if 'V' in name:
                grad_norm = param.grad.norm().item()
                if name in gradient_norms:
                    gradient_norms[name].append(grad_norm)
                else:
                    gradient_norms[name] = [grad_norm]

    def make_meshgrid(x, y, h=0.04):
        x_min, x_max = x.min() - 2, x.max() + 2
        y_min, y_max = y.min() - 2, y.max() + 2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, xx, yy, **params):
        if args.likelihood == "gaussian":
            grid_preds = dkm.gaussian_prediction(model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))).loc.detach().numpy()
        else:
            grid_preds = dkm.categorical_prediction(model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))).probs.detach().numpy()
        #Z = (grid_preds[:,0] - grid_preds[:,1]).reshape(xx.shape)
        Z = (np.argmax(grid_preds, axis=1)).reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out


    epochs_condition_numbers = []
    for i in range(n_epochs):

        # if i == 30:
        #     opt.param_groups[0]['lr'] = 1e-1
        #     scheduler = scheduler_plateau
        model.train()
        start = timer()
        train_obj = 0
        train_lls = []
        batch = 0
        for X, Y in trainloader:
            batch += 1
            model.train()
            X, Y = X.to(device, dtype), Y.to(device, dtype)
            pred = model(X)
            if args.likelihood == "gaussian":
                obj = dkm.gaussian_expectedloglikelihood(pred, torch.nn.functional.one_hot(Y.long(), num_classes=2)) + dkm.norm_kl_reg(model, no_of_data_points-no_of_inducing_points)
            elif args.likelihood == "categorical":
                obj = dkm.categorical_expectedloglikelihood(pred,Y.long()) + dkm.norm_kl_reg(model, no_of_data_points-no_of_inducing_points)
            opt.zero_grad()
            (-obj).backward()
            #print_grad_norms(model)
            #save_grad_norms(model)
            opt.step()

            train_obj += obj.item()*X.size(0)

            #do a breakpoint() if obj is nan
            if math.isnan(obj.item()):
                print("obj is nan")
                breakpoint()

        end = timer()

        mean_train_obj = train_obj / len(trainloader.sampler)
        objs.append(mean_train_obj)


        if args.save_name:
            torch.save(model, "temp_trained_model_"+args.save_name)

        if i%50==0:
            print((i, mean_train_obj, end-start), flush=True)

            # retrieve the maximum memory usage of the CPU
            max_cpu_mem_usage = psutil.Process().memory_info().rss

            # convert to GB
            max_cpu_mem_usage_gb = max_cpu_mem_usage / 1024 ** 3

            # print the result
            print(f"Maximum CPU memory usage: {max_cpu_mem_usage_gb:.2f} GB", flush=True)

        import numpy as np



        previous_obj=mean_train_obj
        # if i >= 30:
        #     scheduler.step(mean_train_obj)
        # else:
        #     opt.param_groups[0]['lr'] = lr_lambda(i)

        scheduler.step()

        # if i == 40:
        #     opt = NGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0, update_period=4)
        #     scheduler = MultiStepLR(opt, milestones=milestones[1:], gamma=[1])  # Assuming gamma is the same for each step for simplicity
        model.eval()
        if args.likelihood == "gaussian":
            test_preds = torch.argmax(dkm.gaussian_prediction(model(x_test)).loc, dim=1)
        else:
            test_preds = torch.argmax(dkm.categorical_prediction(model(x_test)).probs, dim=1)
        test_accuracy = (test_preds == y_test.long()).sum() / y_test.numel()
        if i%50==0:
            print(f"Test Accuracy: {test_accuracy}",flush=True)

        if opt.param_groups[0]['lr'] != previous_lr:
            #print(f"LEARNING RATE HAS CHANGED TO {opt.param_groups[0]['lr']}")
            previous_lr = opt.param_groups[0]['lr']
        gram_matrices = [mod.G for mod in model.modules() if isinstance(mod, dkm.GramLayer)]
        condition_numbers = [np.linalg.cond(G.cpu().detach().numpy()) for G in gram_matrices]
        epochs_condition_numbers.append(condition_numbers)
    runs_epochs_condition_numbers.append(epochs_condition_numbers)

#Since we only have one layer, we can flatten the list
for ncol_div_index in range(len(runs_epochs_condition_numbers)):
    for epoch in range(len(runs_epochs_condition_numbers[ncol_div_index])):
        runs_epochs_condition_numbers[ncol_div_index][epoch] = runs_epochs_condition_numbers[ncol_div_index][epoch][0]

#save the information in runs_epochs_condition_numbers for later, making note of the ncol_divs used
import pandas as pd
df = pd.DataFrame(runs_epochs_condition_numbers)
#use condition numbers as row names
df.index = ncol_divs
df.to_csv("gammas_condition_numbers.csv")





#%% Manual Classification
model.eval()
if args.likelihood == "gaussian":
    test_preds = torch.nn.functional.one_hot(torch.argmax(dkm.gaussian_prediction(model(x_test)).loc, dim=1), num_classes=y_test.unique().numel())
else:
    test_preds = torch.argmax(dkm.categorical_prediction(model(x_test)).probs, dim=1)
test_accuracy = (test_preds == y_test.long()).sum() / y_test.numel()
print(f"Test Accuracy: {test_accuracy}",flush=True)

import numpy as np

def make_meshgrid(x, y, h=0.04):
    x_min, x_max = x.min() - 2, x.max() + 2
    y_min, y_max = y.min() - 2, y.max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, xx, yy, **params):
    if args.likelihood == "gaussian":
        grid_preds = dkm.gaussian_prediction(model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))).loc.detach().numpy()
    else:
        grid_preds = dkm.categorical_prediction(model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))).probs.detach().numpy()
    #Z = (grid_preds[:,0] - grid_preds[:,1]).reshape(xx.shape)
    Z = (np.argmax(grid_preds, axis=1)).reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface')
# Set-up grid for plotting.
#X = x_test
X = x_train
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
#ax.scatter(model[0].X[:,0].detach(), model[0].X[:,1].detach(), c=(1-model[-1].mu.detach().argmax(dim=1)), cmap=plt.cm.spring, s=40, edgecolors='k')
#ax.scatter(model.state_dict()["0.X"][:,0], model.state_dict()["0.X"][:,1], c=1-y_ind, cmap=plt.cm.spring, s=40, edgecolors='k')
#ax.scatter(x_ind[:,0], x_ind[:,1], c=(1-y_ind), cmap=plt.cm.spring, s=40, edgecolors='k')
ax.set_ylabel('x1')
ax.set_xlabel('x0')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()

import imageio

# # Create a list of filenames for the images
# filenames = [f'../temp/epoch_{i+1}.png' for i in range(100)]
#
# # Create an animated gif from the images
# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('../temp/training_animation.gif', images, fps=4, loop=0)  # fps controls the speed of the animation


# Grad norm plotting
# for layer, norms in gradient_norms.items():
#     plt.plot(norms, label=f"{layer}")
#
# plt.xlabel('Iterations')
# plt.ylabel('Gradient Norm')
# plt.yscale('log')
# plt.legend()
# plt.show()
