import torch as t
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

import torch.nn.functional as F

from .util import pos_inv

import math

def gaussian_expectedloglikelihood(f_samples, y, obj_type="mean", noise_var=1):
    #If noise_var isn't a tensor, make one
    try:
        if not isinstance(noise_var, t.Tensor):
            noise_var = t.tensor([noise_var], device=f_samples.device)
    except:
        raise TypeError("noise_var must be a torch.Tensor / Parameter or a scalar")

    if obj_type == "sum":
        return Normal(f_samples.flatten(0, 1), noise_var.sqrt()).log_prob(y.repeat(f_samples.shape[0],1)).sum() / f_samples.shape[0]
    elif obj_type == "mean":
        #breakpoint()
        return Normal(f_samples.flatten(0,1), noise_var.sqrt()).log_prob(y.repeat(f_samples.shape[0],1)).mean(0).sum()
    else:
        raise ValueError("obj_type must be either 'sum' or 'mean'")


def gaussian_prediction(f_samples, noise_var=1):
    #If noise_var isn't a tensor, make one
    try:
        if not isinstance(noise_var, t.Tensor):
            noise_var = t.tensor([noise_var], device=f_samples.device)
    except:
        raise TypeError("noise_var must be a torch.Tensor / Parameter or a scalar")

    averaged_mean = f_samples.mean(0)
    averaged_var = f_samples.var(0) + noise_var
    return Normal(averaged_mean, averaged_var.sqrt())


def categorical_expectedloglikelihood(f_samples, y, obj_type="mean"):
    if obj_type == "sum":
        return -F.cross_entropy(f_samples.flatten(0, 1), y.repeat(f_samples.shape[0])) / f_samples.shape[0]
    elif obj_type == "mean":
        return -F.cross_entropy(f_samples.flatten(0, 1), y.repeat(f_samples.shape[0]))
    else:
        raise ValueError("obj_type must be either 'sum' or 'mean'")

def categorical_prediction(f_samples):
    averaged_log_prob = t.logsumexp(F.log_softmax(f_samples, dim=2), dim=0) - t.log(t.tensor([f_samples.shape[0]], device=f_samples.device))  # More numerically stable method of averaging probabilities over samples.
    return Categorical(t.exp(averaged_log_prob))


class Output(nn.Module):
    def __init__(self, P, out_features, init_mu=None, mc_samples=1000, do_tempering=False, returns='samples', taylor_objective=True):
        """
        params:
          - returns:
                if returns == 'samples', returns [mc_samples] samples from approx posterior
                else if returns == 'distribution', returns the approx posterior itself
        """
        super().__init__()
        self.do_tempering = do_tempering
        self.out_features = out_features
        self.Pi = P
        if init_mu is None:
            self.mu = nn.Parameter(t.randn(P, out_features))
        else:
            self.mu = nn.Parameter(init_mu)
            assert self.mu.shape == (P, out_features)
        #self.V  = nn.Parameter(t.randn(P, P))
        V = t.randn(P, P)
        L = t.linalg.cholesky(V @ V.t() / P + 0.0001*t.eye(P)) * (P ** 0.5)
        L = t.tril(L, diagonal=-1) + t.diag(L.diag().log())
        self.V = nn.Parameter(L)
        self.mc_samples = mc_samples
        self.taylor_objective = taylor_objective
        self.returns = returns
        assert self.returns in ['samples', 'distribution']

    #Shared A for all classes
    @property
    def chol_A(self):
        V = self.V.tril(diagonal=-1) + t.diag(self.V.diag().exp())
        return V / (V.shape[1] ** 0.5)

    @property
    def chol_A_logdiag(self):
        return self.V.diag() - 0.5 * math.log(self.V.shape[1])

    @property
    def A(self):
        V = self.chol_A
        return V @ V.t()

    def compute_objs(self, Kii_chol, Kii, chol_A):
        """
                want to calculate the elbo, ∑_λ KL_λ := ∑_λ KL(N(mu_λ,A) || N(0,Kii))
                (if tempering, we also divide by the number of output channels)

                by definition,
                KL_λ = 1/2 * (logdet(Kii @ A^-1) - P + tr(Kii^-1 @ A) + mu_λ^T @ Kii^-1 @ mu_λ)

                For stability, we use a taylor approximation to the logdet and trace terms.

                Specifically, we represent the logdet and trace in terms of the eigenvalues, and use a second order taylor approximation to the eigenvalues.

                This results in Trace(Kii^-1 @ A) - logdet(Kii @ A^-1) + P ≈ 0.5 * FrobeniusNorm(A^-1 @ Kii - I)^2
        """
        if self.taylor_objective:
            AinvK_minusI = t.cholesky_solve(Kii, chol_A) - t.eye(Kii.shape[0], device=Kii.device)
            taylor_approx = 0.5 * t.linalg.matrix_norm(AinvK_minusI) ** 2
            muT_Kinv = t.cholesky_solve(self.mu, Kii_chol).t()
            muSigmaQmu_term = t.sum((muT_Kinv) * self.mu.T)  # Only term that differs in sum. Compute all at once for efficiency
            if self.do_tempering:
                tempering_term = 1 / self.out_features
            else:
                tempering_term = 1
            return -0.5 * tempering_term * (
                    (taylor_approx) * self.mu.shape[1] + muSigmaQmu_term)  # Multiply non muSigmaQmu terms by out_features for "sum".

        else:
            Kinv_A = t.cholesky_solve(self.A, Kii_chol)
            muT_Kinv = t.cholesky_solve(self.mu, Kii_chol).t()
            trace_term = t.trace(Kinv_A)
            log_det_term = 2 * (t.sum(self.chol_A_logdiag) - t.sum(t.log(t.diagonal(Kii_chol))))  # Logdet using choleskies
            muSigmaQmu_term = t.sum((muT_Kinv) * self.mu.T)  # Only term that differs in sum. Compute all at once for efficiency
            if self.do_tempering:
                tempering_term = 1 / self.out_features
            else:
                tempering_term = 1
            return -0.5 * tempering_term * (
                    (trace_term - log_det_term - Kii_chol.shape[0]) * self.mu.shape[1] + muSigmaQmu_term)  # Multiply non muSigmaQmu terms by out_features for "sum".


    def obj(self):
        return self.computed_obj

    def forward(self, K):
        Kii = K.ii
        Kti = K.ti
        Kt = K.t

        L = t.linalg.cholesky(Kii)
        V = self.chol_A

        # Concatenate Kti (transposed), V, and self.mu
        combined = t.cat([Kti.t(), V, self.mu], dim=1)
        combined_solution = t.linalg.solve_triangular(L, combined, upper=False)

        # K = LL^T
        # A = VV^T
        # X = L^-1 Kti^T
        # Y = L^-1 V
        # Z = L^-1 mu

        # Separate the solutions
        X = combined_solution[:, :Kti.size(0)]
        Y = combined_solution[:, Kti.size(0):Kti.size(0) + V.size(1)]
        Z = combined_solution[:, Kti.size(0) + V.size(1):]

        mean_f = X.t() @ Z
        var_f = t.sum((X.t() @ Y) ** 2, -1) + Kt - t.sum(X.t() * X.t(), -1)

        # Compute final regularisation term
        self.computed_obj = self.compute_objs(L, Kii, V)

        if self.returns == 'samples':
            var_f = var_f[..., None]  # Since we share A for all output features, make sure var_f broadcasts correctly
            std_samples = t.randn((self.mc_samples, *mean_f.shape), device=mean_f.device)
            f_samples = std_samples * var_f.sqrt() + mean_f
            return f_samples
        else:
            return Normal(mean_f, var_f.sqrt())
