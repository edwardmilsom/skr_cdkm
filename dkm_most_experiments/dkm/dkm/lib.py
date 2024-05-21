import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .util import eye_like

class Eye(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, G):
        return StructMat(eye_like(G.ii), t.zeros_like(G.ti), t.ones_like(G.t))


class Weights(nn.Module):
    def __init__(self, weights, learned, lr_scale):
        super().__init__()
        self.lr_scale = lr_scale
        lw = t.tensor(weights).log() / lr_scale
        if learned:
            self.lw = nn.Parameter(lw)
        else:
            self.register_buffer("lw", lw)

    def forward(self):
        return t.exp(self.lw * self.lr_scale)


class StructMat():
    def __init__(self, ii, ti, st):
        assert 2 == ii.dim()
        assert st.dim() + 1 == ti.dim()

        self.ii = ii  # Pi x Pi
        self.ti = ti  # Pt x Shape x Pi
        self.t = st  # Pt

    def __getitem__(self, i):
        if i==0:
            return self.ii
        elif i==1:
            return self.ti
        elif i==2:
            return self.t
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        return 3


class Input(nn.Module):
    def __init__(self, X, Xtrain=None, learned=False):
        super().__init__()
        # Rearrange to put channels as the last dimension.
        # Usually, inputs come as [N, channels, height, width] for images.
        X = X.permute(0, *range(2, X.dim()), 1)
        if not learned:
            self.register_buffer("X", t.Tensor(X))
        else:
            self.X = nn.Parameter(t.Tensor(X))
        if Xtrain is not None:
            Xtrain = Xtrain.permute(0, *range(2, Xtrain.dim()), 1)
            self.register_buffer("Xtrain", t.Tensor(Xtrain))
        else:
            self.Xtrain = None

    def forward(self, Xtest):
        Xtest = Xtest.permute(0, *range(2, Xtest.dim()), 1)
        if (self.Xtrain is not None) and (Xtest is not None):
            return (self.X, t.cat([self.Xtrain, Xtest], 0))
        elif (self.Xtrain is not None) and (Xtest is None):
            return (self.X, self.Xtrain)
        elif (self.Xtrain is None) and (Xtest is not None):
            return (self.X, Xtest)
        elif (self.Xtrain is None) and (Xtest is None):
            return (self.X, t.empty((0, self.X.shape[-1]), device=self.X.device, dtype=self.X.dtype))


class F2G(nn.Module):
    def forward(self, XXt):
        X, Xt = XXt
        N = X.shape[-1]
        Gii = X @ X.mT / N  # Pi x Pi
        Gti = Xt @ X.mT / N  # Pt x H x W x Pi
        Gt = t.sum(Xt ** 2, -1) / N
        return StructMat(Gii, Gti, Gt)

##############
#  Old Stuff #
##############

# This is only here because one of the UCI files uses it.
def ReluLinearF(Min, Mout):
    return NonlinFeature(Mout, [Min], t.relu, F.linear, 1 / math.sqrt(2))


class NonlinFeature(nn.Module):
    def __init__(self, Mout, Mins, nonlin, lin, norm, **kwargs):
        super().__init__()
        self.nonlin = nonlin
        self.lin = lin
        self.kwargs = kwargs
        self.register_buffer("w", t.randn([Mout, *Mins]) * norm / math.sqrt(math.prod(Mins)))

    def forward_indiv(self, x):
        return self.nonlin(self.lin(x, self.w, **self.kwargs))

    def forward(self, x):
        if isinstance(x, t.Tensor):
            return self.forward_indiv(x)
        else:
            xi, xt = x
            return (self.forward_indiv(xi), self.forward_indiv(xt))


class SqExpLinearF(nn.Module):
    def __init__(self, Min, Mout):
        super().__init__()
        self.register_buffer("w", t.randn(Min, Mout) / math.sqrt(Min))
        self.register_buffer("b", t.rand(Mout) * 2 * math.pi)

    def forward(self, x):
        return t.cos(x @ self.w + self.b)