import math

import torch as t
import torch.nn as nn

from torch.utils import checkpoint

from .lib import StructMat
from .matrix_functions import chol_sqrt, logm
from .util import pos_inv
from .output import Output

#Gives
def kl_reg(net):
    total = 0
    reg_layers = [layer for layer in net.modules() if
                  isinstance(layer, Layer) or isinstance(layer, Output) or isinstance(layer, LayerCholesky)]
    for layer in reg_layers:
        total = total + layer.obj()
    return total

#
def norm_kl_reg(net, num_datapoints):
    return kl_reg(net) / num_datapoints


class Layer(nn.Module):
    def obj(self):
        return self.computed_obj

    def __init__(self, P, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__()
        self.do_checkpointing = do_checkpointing
        self.P = P
        self.dof = dof
        self.V = nn.Parameter(t.eye(self.P) * (
            math.sqrt(self.P)))  # In theory this should never be used since we change it in the first forward pass
        self.sqrt = sqrt
        self.MAP = MAP
        self.computed_obj = None
        self.inited = False
        self.latent_Gscale = nn.Parameter(t.zeros(()))

    # The "a" scaling here is pretty arbitrary. It could be changed.
    @property
    def G(self):
        G = self.V @ self.V.t() / self.V.shape[1]
        return G #* t.exp(self.latent_Gscale)


class GramLayerOld(Layer):

    def compute_obj(self, K_inv, G):
        trace_term = t.trace(K_inv @ G)
        log_det_term = t.logdet(G) + t.logdet(K_inv)
        return -0.5 * self.dof * (-log_det_term + trace_term - G.shape[0])

    def _forward(self, *K_tuple):

        K = StructMat(*K_tuple)

        K_flat_ii = K.ii
        K_flat_ti = K.ti.reshape(-1, K.ti.shape[-1])
        K_flat_t = K.t.view(-1)
        K_flat = StructMat(K_flat_ii, K_flat_ti, K_flat_t)

        Kii_inv = pos_inv(K_flat.ii)
        Kti_inv_Kii_flat = K_flat.ti @ Kii_inv
        # ktt_i = K.t - t.diag(Kti_inv_Kii @ K.ti.t())
        ktt_i_flat = K_flat.t - t.sum(Kti_inv_Kii_flat * K_flat.ti, -1)

        Gii_flat = self.G
        Gti_flat = Kti_inv_Kii_flat @ Gii_flat
        gt_flat = t.sum(Gti_flat * Kti_inv_Kii_flat, -1) + ktt_i_flat
        assert (-1E-10 < gt_flat).all()

        Gii = Gii_flat.view(K.ii.shape)
        Gti = Gti_flat.reshape(K.ti.shape)
        gt = gt_flat.view(K.t.shape)

        # Compute and store objective
        computed_obj = self.compute_obj(Kii_inv, Gii)
        return Gii, Gti, gt, computed_obj

    def forward(self, K):
        # Initialize V
        if not self.inited:
            self.V.data = self.sqrt(K.ii) * (K.ii.shape[0]**0.5)
            self.inited = True

        Gii, Gti, gt, computed_obj = t.utils.checkpoint.checkpoint(self._forward, *K, use_reentrant=False)

        self.computed_obj = computed_obj

        return StructMat(Gii, Gti, gt)

class LayerMatrixExp(nn.Module):
    def obj(self):
        return self.computed_obj

    def __init__(self, P, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__()
        self.P = P
        self.dof = dof
        self.V = nn.Parameter(t.eye(self.P) * (
            math.sqrt(self.P)))  # In theory this should never be used since we change it in the first forward pass
        self.sqrt = sqrt
        self.MAP = MAP
        self.computed_obj = None
        self.inited = False
        self.latent_Gscale = nn.Parameter(t.zeros(()))

    # The "a" scaling here is pretty arbitrary. It could be changed.
    @property
    def G(self):
        G = t.linalg.matrix_exp(self.V + self.V.t())
        return G

class GramLayerMatrixExp(LayerMatrixExp):

    def compute_obj(self, K_chol):
        K_inv_G = t.cholesky_solve(self.G, K_chol)
        trace_term = t.trace(K_inv_G)
        chol_G = t.linalg.cholesky(self.G)
        log_det_term = -2 * t.sum(t.log(t.diag(K_chol))) + 2 * t.sum(t.log(t.diag(chol_G)))
        return -0.5 * self.dof * (-log_det_term + trace_term - K_chol.shape[0])

    def _forward(self, *K_tuple):

        K = StructMat(*K_tuple)

        K_flat_ii = K.ii
        K_flat_ti = K.ti.reshape(-1, K.ti.shape[-1])
        K_flat_t = K.t.view(-1)
        K_flat = StructMat(K_flat_ii, K_flat_ti, K_flat_t)

        Kii_chol = t.linalg.cholesky(K_flat.ii)
        Kti_inv_Kii_flat = t.cholesky_solve(K_flat.ti.t(), Kii_chol).t()
        ktt_i_flat = K_flat.t - t.sum(Kti_inv_Kii_flat * K_flat.ti, -1)

        Gii_flat = self.G
        Gti_flat = Kti_inv_Kii_flat @ Gii_flat
        gt_flat = t.sum(Gti_flat * Kti_inv_Kii_flat, -1) + ktt_i_flat
        assert (-1E-10 < gt_flat).all()

        Gii = Gii_flat.view(K.ii.shape)
        Gti = Gti_flat.reshape(K.ti.shape)
        gt = gt_flat.view(K.t.shape)

        # Compute and store objective
        #computed_obj = self.compute_obj(Kii_chol)
        computed_obj = 0
        return Gii, Gti, gt, computed_obj

    def forward(self, K):
        # Initialize V
        if not self.inited:
            #self.V.data = self.sqrt(K.ii) * (K.ii.shape[0]**0.5)
            V_plus_Vt = logm(K.ii)
            V = V_plus_Vt.tril(diagonal=-1) + t.diag(V_plus_Vt.diag()/2)
            self.V.data = V.to(dtype=K.ii.dtype)
            self.inited = True

        if self.do_checkpointing:
            Gii, Gti, gt, computed_obj = t.utils.checkpoint.checkpoint(self._forward, *K, use_reentrant=False)
        else:
            Gii, Gti, gt, computed_obj = self._forward(*K)

        self.computed_obj = computed_obj

        return StructMat(Gii, Gti, gt)

class LayerCholesky(nn.Module):
    def obj(self):
        return self.computed_obj

    def __init__(self, P, dof, sqrt=chol_sqrt, MAP=False, do_checkpointing=True, ncol_div=0, taylor_objective=True):
        super().__init__()
        self.P = P
        self.dof = dof
        self.V = nn.Parameter(t.eye(self.P) * (
            math.sqrt(self.P)))  # In theory this should never be used since we change it in the first forward pass
        self.sqrt = sqrt
        self.MAP = MAP
        self.computed_obj = None
        self.inited = False
        self.latent_Gscale = nn.Parameter(t.zeros(()))
        self.do_checkpointing = do_checkpointing
        self.ncol_div = ncol_div
        self.taylor_objective = taylor_objective

    @property
    def chol_G(self):
        V = self.V.tril(diagonal=-1) + t.diag(self.V.diag().exp()) #Ensure positive diagonal
        return V / V.shape[1]**0.5 #Scale nicely for optimiser

    @property
    def chol_G_logdiag(self):
        return self.V.diag() - 0.5*math.log(self.V.shape[1]) #Cholesky (diag) is exp(V.diag)/sqrt(P), so log is just diag - 0.5*log(P)

    @property
    def G(self):
        V = self.chol_G
        return V @ V.t()

class GramLayerTricks1and2Cholesky(LayerCholesky):
    def __init__(self, P, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__(P, dof, sqrt, MAP)

        self.ild = InverseLogDet(P,iters=1)

    def compute_obj(self, K_inv, backward_logdetK, G):
        trace_term = t.trace(K_inv @ G)
        log_det_term = 2*self.V.diag().log().sum() - backward_logdetK
        return -0.5 * self.dof * (-log_det_term + trace_term - G.shape[0])

    def _forward(self, *K_tuple):

        K = StructMat(*K_tuple)

        K_flat_ii = K.ii
        K_flat_ti = K.ti.reshape(-1, K.ti.shape[-1])
        K_flat_t = K.t.view(-1)
        K_flat = StructMat(K_flat_ii, K_flat_ti, K_flat_t)

        Kii_inv, backward_logdetK, Kii_placeholder  = self.ild(K_flat_ii)
        #print(self.ild.error(K_flat_ii))
        #Kii_inv = pos_inv(K_flat.ii)

        Kti_inv_Kii_flat = K_flat.ti @ Kii_inv
        ktt_i_flat = K_flat.t - t.sum(Kti_inv_Kii_flat * K_flat.ti, -1)

        Gii_flat = self.G
        Gti_flat = Kti_inv_Kii_flat @ Gii_flat
        gt_flat = t.sum(Gti_flat * Kti_inv_Kii_flat, -1) + ktt_i_flat
        assert (-1E-10 < gt_flat).all()

        Gii = Gii_flat.view(K.ii.shape)
        Gti = Gti_flat.reshape(K.ti.shape)
        gt = gt_flat.view(K.t.shape)

        # Compute and store objective
        computed_obj = self.compute_obj(Kii_inv, backward_logdetK, Gii)
        return Gii, Gti, gt, computed_obj

    def forward(self, K):
        # Initialize V
        if not self.inited:
            with t.no_grad():
                self.V.data = t.cholesky(K.ii) * (K.ii.shape[0]**0.5)
                self.ild.reset(K.ii)
            self.inited = True

        if self.do_checkpointing:
            Gii, Gti, gt, computed_obj = t.utils.checkpoint.checkpoint(self._forward, *K, use_reentrant=False)
        else:
            Gii, Gti, gt, computed_obj = self._forward(*K)

        self.computed_obj = computed_obj

        return StructMat(Gii, Gti, gt)

class GramLayerTricks1and2(Layer):
    def __init__(self, P, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__(P, dof, sqrt, MAP)

        self.ild = InverseLogDet(P,iters=1)

    def compute_obj(self, K_inv, backward_logdetK, G):
        trace_term = t.trace(K_inv @ G)
        log_det_term = t.logdet(G) - backward_logdetK
        return -0.5 * self.dof * (-log_det_term + trace_term - G.shape[0])

    def _forward(self, *K_tuple):

        K = StructMat(*K_tuple)

        K_flat_ii = K.ii
        K_flat_ti = K.ti.reshape(-1, K.ti.shape[-1])
        K_flat_t = K.t.view(-1)
        K_flat = StructMat(K_flat_ii, K_flat_ti, K_flat_t)

        Kii_inv, backward_logdetK, Kii_placeholder  = self.ild(K_flat_ii)
        #print(self.ild.error(K_flat_ii))
        #Kii_inv = pos_inv(K_flat.ii)

        Kti_inv_Kii_flat = K_flat.ti @ Kii_inv
        ktt_i_flat = K_flat.t - t.sum(Kti_inv_Kii_flat * K_flat.ti, -1)

        Gii_flat = self.G
        Gti_flat = Kti_inv_Kii_flat @ Gii_flat
        gt_flat = t.sum(Gti_flat * Kti_inv_Kii_flat, -1) + ktt_i_flat
        assert (-1E-10 < gt_flat).all()

        Gii = Gii_flat.view(K.ii.shape)
        Gti = Gti_flat.reshape(K.ti.shape)
        gt = gt_flat.view(K.t.shape)

        # Compute and store objective
        computed_obj = self.compute_obj(Kii_inv, backward_logdetK, Gii)
        return Gii, Gti, gt, computed_obj

    def forward(self, K):
        # Initialize V
        if not self.inited:
            with t.no_grad():
                self.V.data = self.sqrt(K.ii) * (K.ii.shape[0]**0.5)
                self.ild.reset(K.ii)
            self.inited = True

        if self.do_checkpointing:
            Gii, Gti, gt, computed_obj = t.utils.checkpoint.checkpoint(self._forward, *K, use_reentrant=False)
        else:
            Gii, Gti, gt, computed_obj = self._forward(*K)

        self.computed_obj = computed_obj

        return StructMat(Gii, Gti, gt)

class GramLayerStableCholesky(LayerCholesky):

    def compute_obj(self, Kii, chol_G, Kii_chol):
        """
                        want to calculate the elbo, ∑_λ KL_λ := ∑_λ KL(N(0,G) || N(0,Kii))
                        (if tempering, we also divide by the number of output channels)

                        by definition,
                        KL_λ = 1/2 * (logdet(Kii @ G^-1) - P + tr(Kii^-1 @ G))

                        For stability, we use a taylor approximation to the logdet and trace terms.

                        Specifically, we represent the logdet and trace in terms of the eigenvalues, and use a second order taylor approximation to the eigenvalues.

                        This results in Trace(Kii^-1 @ G) - logdet(Kii @ G^-1) + P ≈ 0.5 * FrobeniusNorm(G^-1 @ Kii - I)^2
                """
        if self.taylor_objective:
            GinvK_minusI = t.cholesky_solve(Kii,chol_G) - t.eye(Kii.shape[0], device=Kii.device)
            taylor_approx = 0.5*t.linalg.matrix_norm(GinvK_minusI)**2
            return -0.5 * self.dof * (taylor_approx)
        else:
            Kinv_G = t.cholesky_solve(self.G, Kii_chol)
            trace_term = t.trace(Kinv_G)
            log_det_term = 2 * (t.sum(self.chol_G_logdiag) - t.sum(t.log(t.diagonal(Kii_chol))))
            return -0.5 * self.dof * (-log_det_term + trace_term - Kii_chol.shape[0])

    def _forward(self, *K_tuple):
        K = StructMat(*K_tuple)

        K_flat_ii = K.ii
        K_flat_ti = K.ti.reshape(-1, K.ti.shape[-1])
        K_flat_t = K.t.view(-1)
        K_flat = StructMat(K_flat_ii, K_flat_ti, K_flat_t)

        Gjit = 0.1

        L = t.linalg.cholesky(K_flat_ii)
        cholG = self.chol_G
        if self.training and self.ncol_div>0:
            n_cols = math.ceil(self.P/self.ncol_div)
            V = (cholG/(n_cols**0.5)) @ t.randn(self.P, n_cols, device=cholG.device, dtype=cholG.dtype)
            sample_WG = V @ V.t() + Gjit*t.eye(self.P, device=cholG.device, dtype=cholG.dtype)
            V = t.linalg.cholesky(sample_WG)

        else:
            V = cholG
            sample_WG = V @ V.t() + Gjit * t.eye(self.P, device=cholG.device, dtype=cholG.dtype)
            V = t.linalg.cholesky(sample_WG)

        # Concatenate K_flat_ti (transposed) and V
        combined = t.cat([K_flat_ti.t(), V], dim=1)
        combined_solution = t.linalg.solve_triangular(L, combined, upper=False)

        # K = LL^T
        # G = VV^T
        # X = L^-1 Kti^T
        # Y = L^-1 V

        # Separate the solutions
        X = combined_solution[:, :K_flat_ti.size(0)]
        Y = combined_solution[:, K_flat_ti.size(0):]

        XtY = X.t() @ Y
        Gti_flat = XtY @ V.t()
        Kttdoti_diag = K_flat_t - t.sum(X.t() **2, -1)
        gt_flat = t.sum(XtY **2, -1) + Kttdoti_diag

        assert (-1E-10 < gt_flat).all()
        if self.training and self.ncol_div>0:
            G = sample_WG
        else:
            G = self.G + Gjit * t.eye(self.P, device=self.G.device, dtype=self.G.dtype)
        Gii = G.reshape(K.ii.shape)
        Gti = Gti_flat.reshape(K.ti.shape)
        gt = gt_flat.view(K.t.shape)

        # Compute and store objective
        computed_obj = self.compute_obj(K.ii, cholG, L)
        return Gii, Gti, gt, computed_obj

    def forward(self, K):
        # Initialize V
        if not self.inited:
            V = t.linalg.cholesky(K.ii) * (K.ii.shape[0]**0.5)
            V = t.tril(V,diagonal=-1) + t.diag(V.diag().log())
            self.V.data = V
            self.inited = True

        if self.do_checkpointing:
            Gii, Gti, gt, computed_obj = t.utils.checkpoint.checkpoint(self._forward, *K, use_reentrant=False)
        else:
            Gii, Gti, gt, computed_obj = self._forward(*K)

        self.computed_obj = computed_obj

        return StructMat(Gii, Gti, gt)

GramLayer = GramLayerStableCholesky

import warnings

class InverseLogDet(nn.Module):

    # hook(module, grad_input, grad_output) -> tuple(Tensor) or None
    # The grad_input and grad_output are tuples that contain the gradients with respect to the inputs and outputs respectively. The hook should not modify its arguments, but it can optionally return a new gradient with respect to the input that will be used in place of grad_input in subsequent computations. grad_input will only correspond to the inputs given as positional arguments and all kwarg arguments are ignored. Entries in grad_input and grad_output will be None for all non-Tensor arguments.
    @staticmethod
    def hook(mod, grad_input, grad_output):
        Delta_invA, Delta_logdet, Delta_A = grad_output
        # Delta_invA, Delta_logdet = grad_output#, Delta_A = grad_output

        grad_input = grad_input[0] + 0.  # copies grad_input.
        if Delta_invA is not None:
            X_Delta_invA = mod.X @ Delta_invA
            # input.addmm(mat1, mat2, *, beta=1, alpha=1, out=None)
            # input <- beta*input + alpha*mat1@mat2

            # grad_input <- grad_input - X @ Delta_invA @ X
            grad_input.addmm_(X_Delta_invA, mod.X, alpha=-1)
        if Delta_logdet is not None:
            grad_input.add_(mod.X, alpha=Delta_logdet)

        return (grad_input,)

    def __init__(self, N, iters=1, power_iters=1, omega=0.5, phi=0.8):
        super().__init__()
        self.iters = iters
        self.power_iters = power_iters

        assert 0 < omega
        assert omega <= 1
        assert 0 <= phi
        assert phi <= 1

        if omega < 1 - phi:
            warnings.warn("omega is too small, or phi too large, such that the learning rate doesn't converge to 1 as estimate converges")

        self.omega = omega
        self.phi = phi

        self.register_buffer("X", t.eye(N))
        self.register_buffer("I", t.eye(N))

        # warm-restarted unit vector; uses power iterations to align along max eigenvector
        self.register_buffer("v", t.randn(N))
        # scratch space for power iterations.
        self.register_buffer("AXv", t.randn(N))

        # Scratch space for Newton iterations.
        self.register_buffer("AX", t.zeros(N, N))  # Also used in error
        self.register_buffer("XAX", t.zeros(N, N))

        self.register_full_backward_hook(InverseLogDet.hook)

    def reset(self, A):
        t.linalg.inv(A.detach(), out=self.X)

    def forward(self, A):
        # Note that we're returning A to ensure that the backward hook runs.

        # A is the matrix to invert.
        # X is the current estimate of the inverse.
        assert A.shape == self.X.shape
        assert A.device == self.X.device
        assert A.dtype == self.X.dtype

        with t.no_grad():
            for _ in range(self.iters):
                use_newtons = False
                if use_newtons:
                    # X_new = X + lr*(X - X A X)

                    t.mm(A, self.X, out=self.AX)  # Sets AX
                    t.mm(self.X, self.AX, out=self.XAX)  # Sets XAX

                    # For stability, we need the magnitudes of the eigenvalues of lr*(AX - I) to all be smaller than 1.
                    # Negative eigenvalues can't go out-of-bounds, because A and X are postive definite.
                    # But positive eigenvalues can be too large.
                    # In particular, we need the eigenvalues of lr*AX to be smaller than 2.
                    # To find the maximum eigenvalue, we use a power-iteration.
                    # If max_eig is small, we can just use lr=1.
                    # If max_eig is big, we set max_eig * lr = 1; or lr = 1/max_eig (to ensure we're well below the bound).

                    for _ in range(self.power_iters):
                        t.mv(self.AX, self.v, out=self.AXv)
                        norm = t.dot(self.AXv, self.AXv).sqrt()
                        t.div(self.AXv, norm, out=self.v)

                    lr = 1 / max((norm - self.phi) / self.omega, 1.)
                    self.X.mul_(1 + lr).sub_(self.XAX, alpha=lr)
                else:
                    t.linalg.cholesky(A, out=self.X)
                    t.cholesky_inverse(self.X, out=self.X)

        # Need to explicitly ensure that requires_grad is set to true.
        return self.X.clone().requires_grad_(True), t.zeros((), requires_grad=True, device=A.device, dtype=A.dtype), A

    def error(self, A):
        # Average Frobenius norm of I - A@X
        return ((self.I - self.AX) ** 2).mean().sqrt()


# Scale features by a matrix D, inheriting from Layer for convenience.
#TODO: Objective
class FeatureScaling(Layer):

    #Just put this here to make it clearer that this class needs N, the number of features, rather than P, the number of data points.
    def __init__(self, N, dof, sqrt=chol_sqrt, MAP=False):
        super().__init__(N, dof, sqrt, MAP)

    #In theory should be equivalent to the other compute_obj. This is the formula from the old "D Notation".
    def compute_obj(self):
        D = self.G
        trace = -0.5 * self.dof * (t.trace(D) - D.shape[-1])
        return trace


    def indiv_forward(self, X):

        Yp = X @ self.sqrt(self.G)
        return Yp

    def forward(self, Xs):
        if isinstance(Xs, tuple):
            X, Xt = Xs
            X_scaled, Xt_scaled = self.indiv_forward(X), self.indiv_forward(Xt)
            self.compute_obj()
            return (X_scaled, Xt_scaled)
        #TODO: Why is this else case here?
        else:
            assert isinstance(Xs, t.Tensor)
            return self.indiv_forward(Xs)

class FeatToGram(nn.Module):
    def __init__(self, Pi=None, N=None, Xi=None, do_learn_Xi=True):
        """
        params: Pi          - number of inducing points,
                N           - number of features,
                Xi          - (optional) initialization for the inducing points. if [Xi]
                              provided, it must have shape (Pi, N). if no inducing points are provided,
                              they will be initialized as std. normal random vectors
                do_learn_Xi - if True, then inducing points will be treated as a pytorch parameter,
                              otherwise inducing points will remain fixed
        """
        super().__init__()
        if N is None: raise ValueError("N must be provided")
        if Pi is None: raise ValueError("Pi must be provided")
        assert Xi is None or Xi.size() == (Pi, N), "Xi must have shape (Pi, N)"

        self.N = N
        Xi = t.randn((Pi, N)) if Xi is None else Xi
        if do_learn_Xi:
            self.Xi = nn.Parameter(Xi)
        else:
            self.register_buffer("Xi", Xi)
    def forward(self, Xt):
        """params: Xt - should have shape (Pt, *batch_dims, N)"""
        Gii = self.Xi @ self.Xi.mT / self.N
        Gti = Xt @ self.Xi.mT / self.N
        Gt = t.sum(Xt * Xt, -1) / self.N
        return StructMat(Gii, Gti, Gt)
