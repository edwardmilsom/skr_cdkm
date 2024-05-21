import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .lib import StructMat
from .util import pos_inv
from .layer import Layer, GramLayer
from .kernels import SumKernel, ReluKernel
from .batchnorm import GramBatchNorm2PixelAverageScaled, GramBatchNorm
from .matrix_functions import chol_sqrt

# Convolution layer using the C inducing point parameterisation.
class ConvMixup(nn.Module):
    def __init__(self, in_ind_points, out_ind_points, filter_size=3, stride=1, do_checkpointing=True, noise_var=0):
        super().__init__()
        self.do_checkpointing = do_checkpointing
        self.fanin = in_ind_points
        self.filter_size = filter_size
        self.Kt_mixer = nn.AvgPool2d(filter_size, stride=stride, padding=(filter_size // 2), count_include_pad=False)

        # Initialize weights as latents
        self.latent_learned_weights = nn.Parameter(t.randn(out_ind_points, in_ind_points, filter_size, filter_size))
        self.stride= stride

        self.noise_var = noise_var

    def _forward(self, *K_tuple):
        # Actual weights should be order 1/Pi, so we scale by sqrt(Pi) to get the right variance.
        # The learned parameters are order 1.
        if self.training:
            scaled_weights = self.latent_learned_weights / self.fanin**0.5 + t.randn(self.latent_learned_weights.shape, device=self.latent_learned_weights.device, dtype=self.latent_learned_weights.dtype)*self.noise_var
        else:
            scaled_weights = self.latent_learned_weights / self.fanin**0.5

        K = StructMat(*K_tuple)

        Kii = K.ii
        Kti = K.ti.permute(0, -1, *range(1, K.ti.dim() - 1))  # Put spatial last to work with PyTorch Conv. We currently believe that this does not result in a call to contiguous when using conv2d.
        Kt = K.t

        Mt = self.Kt_mixer(Kt)
        Mti = F.conv2d(Kti, scaled_weights, padding=self.filter_size // 2, stride=self.stride)  / (self.filter_size ** 2)
        Mti_p = Mti.permute(0, *range(2, Mti.dim()), 1)
        C = scaled_weights.permute(-2, -1, 0, 1)  # *range(0,Kti.dim()-2)
        Mii = t.mean(C @ Kii @ C.mT, (0, 1))

        return Mii, Mti_p, Mt

    def forward(self, K):
        if self.do_checkpointing:
            return StructMat(*t.utils.checkpoint.checkpoint(self._forward, *K, use_reentrant=False))
        else:
            return StructMat(*self._forward(*K))


# Global average pooling layer using the Nystrom Approximation.
class NystromGAP(nn.Module):
    def forward(self, K):
        batch, *space, M = K.ti.shape
        spatial_inds = tuple(range(1, K.t.dim()))

        gKti = K.ti.mean(spatial_inds)
        assert gKti.shape == (batch, M)

        chol_Kii = t.linalg.cholesky(K.ii)
        Kii_inv_gKit = t.cholesky_solve(gKti.t(), chol_Kii)
        gKt = t.sum(Kii_inv_gKit.t() * gKti, -1)
        assert gKt.shape == (batch,)

        Kti_flat = K.ti.view(-1, M)
        Kt_flat = K.t.view(-1)
        Kii_inv_Kit = t.cholesky_solve(Kti_flat.t(), chol_Kii)
        D = Kt_flat - t.sum(Kii_inv_Kit.t() * Kti_flat, -1)
        D = D.view(batch, *space)
        gD = D.sum(tuple(range(1, D.dim()))) / math.prod(space)**2
        assert gD.shape == (batch,)

        corrected_gKt = gKt + gD

        return StructMat(K.ii, gKti, corrected_gKt)

class NystromGAPMixup(nn.Module):
    def __init__(self, in_ind_points, out_ind_points, noise_var=0):
        super().__init__()
        self.in_ind_points = in_ind_points
        self.out_ind_points = out_ind_points
        self.width = None
        self.height = None
        self.initialised = False
        self.noise_var = noise_var

        # Initialize weights as latents
        self.latent_learned_weights = nn.Parameter(t.randn(out_ind_points, in_ind_points))

    def forward(self, K):
        if self.training:
            scaled_weights = self.latent_learned_weights / self.in_ind_points**0.5 + t.randn(self.latent_learned_weights.shape, device=self.latent_learned_weights.device, dtype=self.latent_learned_weights.dtype)*self.noise_var
        else:
            scaled_weights = self.latent_learned_weights / self.in_ind_points**0.5

        batch, *space, M = K.ti.shape

        Mti = F.linear(K.ti.mean(tuple(range(1,K.t.dim()))), scaled_weights)

        C = scaled_weights
        Mii = (C @ K.ii @ C.T)

        gKti = K.ti.mean(tuple(range(1, K.t.dim())))
        assert gKti.shape == (batch, M)

        chol_Kii = t.linalg.cholesky(K.ii)
        Kii_inv_gKit = t.cholesky_solve(gKti.t(), chol_Kii)
        gKt = t.sum(Kii_inv_gKit.t() * gKti, -1)
        assert gKt.shape == (batch,)

        Kti_flat = K.ti.view(-1, M)
        Kt_flat = K.t.view(-1)
        Kii_inv_Kit = t.cholesky_solve(Kti_flat.t(), chol_Kii)
        D = Kt_flat - t.sum(Kii_inv_Kit.t() * Kti_flat, -1)
        D = D.view(batch, *space)
        gD = D.sum(tuple(range(1, D.dim()))) / (math.prod(space)**2)
        assert gD.shape == (batch,)

        corrected_gKt = gKt + gD

        return StructMat(Mii, Mti, corrected_gKt)

#Just means over the spatial dimensions in the style of the conv GP which is defined as an additive GP over patch response functions.
class BigFinalConv(nn.Module):
    def forward(self, K):
        spatial_inds = tuple(range(1, K.t.dim()))
        return StructMat(K.ii, K.ti.mean(spatial_inds), K.t.mean(spatial_inds))


# Flatten and linear is the same thing as one massive convolution, so we do the same thing as ConvMixup by introducing C for the inducing points.
#Requires forward pass before training loop to initialise the weights.
class BigFinalConvMixup(nn.Module):
    def __init__(self, in_ind_points, out_ind_points, noise_var=0):
        super().__init__()
        self.in_ind_points = in_ind_points
        self.out_ind_points = out_ind_points
        self.width = None
        self.height = None
        self.initialised = False
        self.noise_var = noise_var

    def forward(self, K):
        batch, *space, M = K.ti.shape
        spatial_inds = tuple(range(1, K.t.dim()))

        if not self.initialised:
            # Initialize weights as latents
            # Note that they have this shape because we interpret the ti operation as an out x in*space linear layer.
            self.latent_learned_weights = nn.Parameter(t.randn(self.out_ind_points, math.prod(space) * self.in_ind_points))

            self.initialised = True

        if self.training:
            scaled_weights = self.latent_learned_weights / (self.in_ind_points)**0.5 + t.randn(self.latent_learned_weights.shape, device=self.latent_learned_weights.device, dtype=self.latent_learned_weights.dtype)*self.noise_var
        else:
            scaled_weights = self.latent_learned_weights / (self.in_ind_points)**0.5

        Mt = K.t.mean(spatial_inds)
        Mti = F.linear(K.ti.flatten(spatial_inds[0], spatial_inds[-1] + 1), scaled_weights) / math.prod(space)
        C = scaled_weights.reshape(self.out_ind_points, -1, self.in_ind_points).permute(1,0,2)
        Mii = (C @ K.ii @ C.mT).mean(0)

        return StructMat(Mii, Mti, Mt)



#Emulates the structure of the pytorch-cifar github repo
def ResNetBlock(n, in_ind_points, out_ind_points, dof, filter_size=3, stride=1, bn_indnorm="local", bn_tnorm="local", bn_indscale="global", bn_tscale="global", Kernel=ReluKernel, mult_eps=1e-4, abs_eps=1e-4, ncol_div=0, c_noisevar = 0, taylor_objective=True):

    strides = [stride] + [1]*(n-1)
    ins = [in_ind_points] + [out_ind_points]*(n-1)
    outs = [out_ind_points]*n

    def reschunk(in_ind_points, out_ind_points, stride):
        if stride != 1:
            shortcut = nn.Sequential(
                                      ConvMixup(in_ind_points, out_ind_points, filter_size=1, stride=stride, noise_var=c_noisevar),
                                      GramBatchNorm(bn_indnorm,bn_tnorm,bn_indscale,bn_tscale)
                                      )
        else:
            shortcut = nn.Sequential()
        return SumKernel([shortcut, nn.Sequential(
                                                        #Conv 1
                                                        Kernel(mult_eps=mult_eps, abs_eps=abs_eps),
                                                        ConvMixup(in_ind_points,out_ind_points,filter_size=filter_size, stride=stride, noise_var=c_noisevar),
                                                        GramLayer(out_ind_points,dof, ncol_div=ncol_div, taylor_objective=taylor_objective),
                                                        GramBatchNorm(bn_indnorm,bn_tnorm,bn_indscale,bn_tscale),
                                                        #Conv 2
                                                        Kernel(mult_eps=mult_eps, abs_eps=abs_eps),
                                                        ConvMixup(out_ind_points,out_ind_points,filter_size=filter_size, stride=1, noise_var=c_noisevar),
                                                        GramLayer(out_ind_points,dof, ncol_div=ncol_div, taylor_objective=taylor_objective),
                                                        GramBatchNorm(bn_indnorm,bn_tnorm,bn_indscale,bn_tscale),
                                                        )
                            ] , [0.5,0.5])
    return nn.Sequential(*[reschunk(ins[i], outs[i], strides[i]) for i in range(n)])

#Gram matrices fixed at kernel, so equivalent to standard infinite DGP.
def noGram_ResNetBlock(n, in_ind_points, out_ind_points, dof, filter_size=3, stride=1, bn_indnorm="local", bn_tnorm="local", bn_indscale="global", bn_tscale="global", Kernel=ReluKernel, mult_eps=1e-4, abs_eps=1e-4):

    strides = [stride] + [1]*(n-1)
    ins = [in_ind_points] + [out_ind_points]*(n-1)
    outs = [out_ind_points]*n

    def reschunk(in_ind_points, out_ind_points, stride):
        if stride != 1:
            shortcut = nn.Sequential(
                                      ConvMixup(in_ind_points, out_ind_points, filter_size=1, stride=stride),
                                      GramBatchNorm(bn_indnorm,bn_tnorm,bn_indscale,bn_tscale)
                                      )
        else:
            shortcut = nn.Sequential()
        return SumKernel([shortcut, nn.Sequential(
                                                        #Conv 1
                                                        Kernel(mult_eps=mult_eps, abs_eps=abs_eps),
                                                        ConvMixup(in_ind_points,out_ind_points,filter_size=filter_size, stride=stride),
                                                        #GramLayer(out_ind_points,dof),
                                                        GramBatchNorm(bn_indnorm,bn_tnorm,bn_indscale,bn_tscale),
                                                        #Conv 2
                                                        Kernel(mult_eps=mult_eps, abs_eps=abs_eps),
                                                        ConvMixup(out_ind_points,out_ind_points,filter_size=filter_size, stride=1),
                                                        #GramLayer(out_ind_points,dof),
                                                        GramBatchNorm(bn_indnorm,bn_tnorm,bn_indscale,bn_tscale),
                                                        )
                            ] , [0.5,0.5])
    return nn.Sequential(*[reschunk(ins[i], outs[i], strides[i]) for i in range(n)])

# Form Gram matrix from data by dot-producting patches.
class ConvF2G(nn.Module):
    def forward(self, XXt):
        X, Xt = XXt
        N = X.shape[-1]
        assert X.shape[-2] == X.shape[-3]  # We require inducing patches to be square for now

        Gii = (X.reshape(X.shape[0], -1) @ X.reshape(X.shape[0], -1).t()) / (N * X.shape[-2] * X.shape[-3])  # Pi x Pi

        Xt_p = Xt.permute(0, -1, *range(1, Xt.dim() - 1))
        X_p = X.permute(0, -1, *range(1, X.dim() - 1))

        Gt = F.avg_pool2d(t.mean(Xt_p * Xt_p, 1), X_p.shape[-2], stride=1, padding=X_p.shape[-2] // 2,
                          count_include_pad=False)

        Gti = F.conv2d(Xt_p, X_p, stride=1, padding=(X_p.shape[-1] // 2)) / (
                    N * X_p.shape[-2] * X_p.shape[-1])  # Pt x Pi x H x W

        Gti_p = Gti.permute(0, *range(2, Gti.dim()), 1)  # Pt x H x W x Pi

        return StructMat(Gii, Gti_p, Gt)

class ConvF2GScaled(Layer):
    #To remind user that this needs N, not P
    def __init__(self, N, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__(N, dof, sqrt, MAP)

    def compute_obj(self):
        D = self.G
        trace = -0.5 * self.dof * (t.trace(D) - D.shape[-1])
        return trace

    def forward(self, XXt):
        X, Xt = XXt
        N = X.shape[-1]
        W_f = X.shape[-3]
        H_f = X.shape[-2]
        W = Xt.shape[-3]
        H = Xt.shape[-2]
        assert W_f == H_f  # We require inducing patches to be square

        D_filter = self.G.reshape(N * W_f * H_f, N, W_f, H_f)

        # It's important that we are consistent in how we multiply D with things.
        # If we do everything after the permutes, it should all work out consistently.

        Xt_p = Xt.permute(0, -1, *range(1, Xt.dim() - 1))
        X_p = X.permute(0, -1, *range(1, X.dim() - 1))

        Gii = (X_p.reshape(X_p.shape[0], -1) @ self.G @ X_p.reshape(X_p.shape[0], -1).t()) / (N * W_f * H_f)  # Pi x Pi

        unfold = nn.Unfold((W_f, H_f), padding=(W_f // 2))
        Xt_p_unfold = unfold(Xt_p).unflatten(-1, (W, H))

        Gt = (F.conv2d(Xt_p, D_filter, padding="same", stride=1) * Xt_p_unfold).mean(
            1)

        XD_filter = (X_p.reshape(X_p.shape[0], -1) @ self.G).reshape(*X_p.shape)
        Gti = F.conv2d(Xt_p, XD_filter, stride=1, padding=(X_p.shape[-1] // 2)) / (N * W_f * H_f)  # Pt x Pi x H x W

        Gti_p = Gti.permute(0, *range(2, Gti.dim()), 1)  # Pt x H x W x Pi

        self.computed_obj = self.compute_obj()

        return StructMat(Gii, Gti_p, Gt)