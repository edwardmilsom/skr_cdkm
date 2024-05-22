import torch as t

def abs_eigvals(X):
    values, vectors = t.linalg.eigh(X)
    return (vectors * t.abs(values)) @ vectors.t()

def eye_like(A):
    assert A.shape[-1] == A.shape[-2]
    return t.eye(A.shape[-1], dtype=A.dtype, device=A.device)

def pos_def(A, tol=0):
    return (-tol < t.linalg.eigvalsh(A)).all()

def pos_inv(A):
    return t.cholesky_inverse(t.linalg.cholesky(A))

def pos_logdet(A):
    L = t.linalg.cholesky(A)
    return 2*L.diagonal(dim1=-1, dim2=-2).log().sum(-1)
