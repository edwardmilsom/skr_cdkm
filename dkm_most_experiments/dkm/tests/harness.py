import torch as t
from dkm.model import FCDKM
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import dkm

default_args = dict(
    device='cpu',
    dtype='float64',
    lr=1e-2,
    seed=0,
    Pt=500,
)

default_model_params = dict(
    dof=1e0,
    Pi=32,
    num_layers=2,
    feat_to_gram_params=dict(),
    output_params=dict(mc_samples=100),
    kernel=dict(type='relu', params=dict())
)

class Sin2dHarness:
    """test harness for the simple sin2d problem"""
    def __init__(self, model_params=default_model_params, **kwargs):
        """params: override default args with [kwargs]"""
        self.model_params = model_params | kwargs
        kwargs = default_args | kwargs
        for name in kwargs:
            setattr(self, name, kwargs[name])

        self.dtype = dict(float64=t.float64,float32=t.float32)[self.dtype]
        t.set_default_dtype(self.dtype)
        t.manual_seed(self.seed)

        self.Nin = 2; self.Nout = 2
        self._init_dataset()
    def _init_dataset(self):
        Ptrain = 1000; Ptest = 1000
        Pi = self.model_params['Pi']
        Pfull = Ptrain + Ptest + Pi; assert Pfull % 2 == 0, "need even number of data points..."
        xs_0 = t.linspace(-2,2,Pfull//2) + t.randn(Pfull//2)*0.1
        xs_1 = t.linspace(-2,2,Pfull//2) + t.randn(Pfull//2)*0.1
        xs_0 = t.stack([xs_0, t.sin(xs_0*3)*6+t.randn(Pfull//2)], dim=1)
        xs_1 = t.stack([xs_1, t.sin(xs_1*3)*6+t.randn(Pfull//2)+5], dim=1)
        X = t.cat([xs_0, xs_1])
        shuffle_perm = t.randperm(Pfull)
        X = X[shuffle_perm]
        y = t.cat([t.zeros(Pfull//2), t.ones(Pfull//2)])[shuffle_perm]

        Xi = X[:Pi]; yi = y[:Pi]
        Xtrain = X[Pi:(Ptrain+Pi)]; Xtest = X[(Ptrain+Pi):]
        ytrain = y[Pi:(Ptrain+Pi)]; ytest = y[(Ptrain+Pi):]

        # standardize
        mu = Xtrain.mean(dim=0); std = Xtrain.std(dim=0)
        Xi = (Xi - mu)/std; Xtrain = (Xtrain - mu)/std; Xtest = (Xtest - mu)/std

        # set values
        self.Ptrain = Ptrain
        self.Xi = Xi
        self.ytest = ytest; self.ytrain = ytrain
        self.yi_oh = F.one_hot(yi.long(), num_classes=self.Nout).to(dtype=self.dtype)
        train_dataset = TensorDataset(Xtrain, ytrain)
        test_dataset = TensorDataset(Xtest, ytest)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.Pt, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.Pt, shuffle=False)
    def reset_model(self, **extra_model_args):
        self.model_params = self.model_params | extra_model_args
        if 'Pi' in extra_model_args: self._init_dataset() # have to reset dataset
        mps = self.model_params
        # add implicit params
        mps['feat_to_gram_params'] = mps['feat_to_gram_params'] | dict(Xi=self.Xi, do_learn_Xi=True)
        mps['Nin'] = self.Nin; mps['Nout'] = self.Nout; mps['output_params'] = mps['output_params'] | dict(init_mu=self.yi_oh)
        self.model = FCDKM(**mps).to(device=self.device, dtype=self.dtype)
        self.opt = Adam(self.model.parameters(), lr=self.lr)
    def train_epoch(self):
        """perform one epoch of training, and calculate performance metrics"""
        self.model.train()
        lls = []; preds = []; objs = []; trues = []
        for (X, y) in self.train_dataloader:
            X = X.to(device=self.device, dtype=self.dtype)
            y = y.to(device=self.device, dtype=self.dtype)
            self.opt.zero_grad()
            pred = self.model(X)
            ll = dkm.categorical_expectedloglikelihood(pred, y.long())
            dkm_obj = ll + dkm.norm_kl_reg(self.model, self.Ptrain)
            (-dkm_obj).backward()
            self.opt.step()

            approx_posterior = dkm.categorical_prediction(pred)
            batch_preds = approx_posterior.probs.argmax(-1)
            lls.append(ll.cpu().detach().item())
            objs.append(dkm_obj.cpu().detach().item())
            preds.extend(batch_preds.cpu().detach().tolist())
            trues.extend(y.cpu().detach().tolist())
        return dict(ll=t.tensor(lls).mean().item(),
                    acc=(t.tensor(preds) == t.tensor(trues)).double().mean().item(),
                    obj=t.tensor(objs).mean().item())
    def test_eval(self):
        """perform a pass of the test set, and calculate performance metrics"""
        self.model.eval()
        lls = []; preds = []; objs = []; trues = []
        for (X, y) in self.test_dataloader:
            X = X.to(device=self.device, dtype=self.dtype)
            y = y.to(device=self.device, dtype=self.dtype)
            pred = self.model(X)
            ll = dkm.categorical_expectedloglikelihood(pred, y.long())
            dkm_obj = ll + dkm.norm_kl_reg(self.model, self.Ptrain)
            lls.append(ll.cpu().detach().item())
            objs.append(dkm_obj.cpu().detach().item())
            preds.extend(pred.argmax(dim=-1).mode(dim=0)[0].cpu().detach().tolist())
            trues.extend(y.cpu().detach().tolist())
        return dict(ll=t.tensor(lls).mean().item(),
                    acc=(t.tensor(preds) == t.tensor(trues)).double().mean().item(),
                    obj=t.tensor(objs).mean().item())