import torch as t
from torch import nn
from .layer import GramLayer, FeatToGram
from .kernels import ReluKernel, PolarHeavisideKernel
from .output import Output
from .batchnorm import GramBatchNorm


class FCDKM(nn.Module):
    """A fully connected deep kernel machine

    Some high level details:
        - A 'block' is defined by: Block(G) = Kernel(Norm(Gram(G))),
          we form a full model by stacking [num_layers] blocks;
        - To convert the input features to kernels, we use a [FeatToGram]layer;
          and to ensure positive definiteness of the input, we use an extra kernel
          layer immediately after;
        - To convert the final kernel to an output, we use an [Output] layer.
    """
    def __init__(self, Pi=None, dof=None, num_layers=None,
                       Nin=None, Nout=None,
                       feat_to_gram_params=dict(),
                       kernel=dict(type='relu', params=dict()),
                       bn_params=dict(),
                       output_params=dict(),
                       gram_params=dict()):
        super().__init__()
        self.num_layers = num_layers
        self.is_nngp = dof == float('inf')
        self.f2g = FeatToGram(Pi=Pi, N=Nin, **feat_to_gram_params)
        k = {'relu': ReluKernel, 'polar-heaviside': PolarHeavisideKernel}[kernel['type']]
        self.norms = nn.ModuleList([GramBatchNorm(**bn_params) for _ in range(num_layers)])
        self.kernels = nn.ModuleList([k(**kernel['params']) for _ in range(num_layers+1)])
        if not self.is_nngp: self.grams = nn.ModuleList([GramLayer(Pi, dof, **gram_params) for _ in range(num_layers)])
        self.output = Output(Pi, Nout, **output_params)
    def forward(self, x):
        x = self.f2g(x)
        x = self.kernels[0](x)
        for i in range(self.num_layers):
            if not self.is_nngp: x = self.grams[i](x)
            x = self.norms[i](x)
            x = self.kernels[i+1](x)
        x = self.output(x)
        return x