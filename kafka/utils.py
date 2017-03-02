#!/usr/bin/env python
"""Some utility functions used by the main code base."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def spsolve2(a, b):
    a_lu = spl.splu(a.tocsc()) # LU decomposition for sparse a
    out = sp.lil_matrix((a.shape[1], b.shape[1]), dtype=np.float32)
    b_csc = b.tocsc()
    for j in xrange(b.shape[1]):
        bb = np.array(b_csc[j, :].todense()).squeeze()
        out[j,j] = a_lu.solve(bb)[j]
    return out.tocsr()

def matrix_squeeze (A, mask=None, n_params=1):
    """The matrix A is squeezed out of 0-filled submatrices. We return a
    squeezed version of the original matrix, as well as the locations of the
    non-zero elements that enable a reconstruction from the squeezed version.

    Parameters
    ----------
    A: array
        An N*N matrix, ideally sparse (hence the "!=" comparison). The matrix
        is assumed to be block diagonal, with some blocks along the diagonal
        set to 0.

    Returns
    -------
    The squeezed matrix, as well as a Boolean array that allows to reconstruct
    the original matrix with zero padding.
    """
    n = mask.sum()
    A_squeeze = sp.csr_matrix((n,n))
    if mask is None:
        # A needs to be CSR or something...
        rows, columns, values = sp.find(A)
        A_squeeze = A[rows, :][:, cols]
    else:

        m = np.array([], dtype=np.bool)
        if n_params > 1:
            for i in xrange(n_params):
                m = np.r_[m, mask]
        else:
            m = mask
        if A.ndim == 2:
            A_squeeze = A[m, :][:, m]
        elif A.ndim == 1: # vector
            A_squeeze = np.zeros(n_params*n)
            A_squeeze = A[m]

    return A_squeeze


def reconstruct_array(A, B, mask, n_params=1):
    """We have an array A which is a subset of B. We want to update B with the
    elements of A, which are given where mask is true"""

    mask = mask.ravel()
    if A.ndim == 1:
        n = mask.shape[0] # big dimension
        n_good = mask.sum()
        ilocs = mask.nonzero()[0]
        for i in xrange(n_params):
            B[ilocs +i*n] = A[(i*n_good):((i+1)*n_good)]
        return B
    elif A.ndim == 2:
        n = mask.shape[0] # big dimension
        n_good = mask.sum()
        ilocs = mask.nonzero()[0]
        for i in xrange(n_params):
            ii = 0
            for j in xrange(n):
                if mask[j]:
                    B[j + i*n, i*n + ilocs] = A[ii, (i*n_good):((i+1)*n_good)]
                    ii = ii+1
        return B


def matrix_reconstruct (A, mask, n_params=1):
    """Reconstruct an "squeezed" matrix to full size, where we have the
    locations of the non-zero elements in the original matrix.

    Parameters
    ----------
    A: array
        A N_squeeze*N_squeeze array or sparse matrix
    no_zeros: array
        An array of booleans of size N (where N >= N_squeeze) that has the
        locations of the data points in the full matrix representation

    Returns
    --------
    A sparse matrix
    """
    if n_params==1:
        # Square matrix
        n = mask.shape[0]
        locs = mask.nonzero()[0]
        rows_squeeze, cols_squeeze, values = sp.find(A)
        reconstruct = sp.csr_matrix( (values, (locs, locs)), shape=(n,n),dtype=np.float32 )
        return reconstruct
    else:
        # A.shape[1]*3 == A.shape[0]
        n = mask.shape[0]
        locs = mask.nonzero()[0]
        rows_squeeze, cols_squeeze, values = sp.find(A)
        reconstruct = sp.csr_matrix( (values, (locs, locs)), shape=(n_params*n,n),dtype=np.float32 )
        return reconstruct


    #return sp.lil_matrix(reconstruct)
