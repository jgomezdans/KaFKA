#!/usr/bin/env python
"""Some utility functions used by the main code base."""
# KaFKA A fast Kalman filter implementation for raster based datasets.
# Copyright (c) 2017 J Gomez-Dans. All rights reserved.
#
# This file is part of KaFKA.
#
# KaFKA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KaFKA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KaFKA.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


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


def matrix_squeeze (a_matrix, mask=None, n_params=1):
    """The matrix A is squeezed out of 0-filled submatrices, or rows/cols where
     a 1D mask indicates False. We return a squeezed version of the original
     matrix.
    Parameters
    ----------
    a_matrix: array
        An N,N array, or an N array. We assume that if no mask is given, we want
        to squeeze out all the zero locations
    mask: array
        An N boolean array, indicating where to squeeze the original array
    n_params: integer
        The number of parameters in the state vector. So for an identity o
        operator this will be 1, for the kernels 3, for TIP, 7, ...
    Returns
    -------
    The squeezed matrix.
    """
    if mask is None:
        # a needs to be a sparse matrix, otherwise find doesn't work!
        rows, columns, values = sp.find(a_matrix)
        # We can now squeeze easily using slicing of the original matrix
        a_matrix_squeezed = a_matrix[rows, :][:, columns]
    else:
        # Calculate the size of the output array from the non-zero mask elements
        n = mask.sum()
        a_matrix_squeezed = sp.csr_matrix((n, n))
        m = np.array([], dtype=np.bool)
        # the next if statement is there to cope with problems where the size of
        # the state has more than one parameter
        if n_params > 1:
            # We need to stack the masks in this case
            for i in xrange(n_params):
                m = np.r_[m, mask]
        else:
            # We don't stack the masks, as there's only one parameter
            m = mask
        # This is different for vector and matrix
        if a_matrix.ndim == 2:
            # Just subset by mask location in rows/cols
            a_matrix_squeezed = a_matrix[m, :][:, m]
        elif a_matrix.ndim == 1: # vector
            # Same, but just in one dimension
            a_matrix_squeezed = np.zeros(n_params*n)
            a_matrix_squeezed = a_matrix[m]
    return a_matrix_squeezed


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
