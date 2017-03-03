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
        if mask.ndim == 2: 
            mask = mask.ravel()
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


def reconstruct_array(a_matrix, b_matrix, mask, n_params=1):
    """A function to fill in a squeezed array (a_matrix) with elements from a
    complete array (b_matrix). In effect, the elements of the b_matrix where the
    mask is True will be updated with the elements of a_matrix that correspond.
    The function works both on vectors and matrices, and they need to be
    ordered.
    Parameters
    -----------
    a_matrix: array
        The squeezed matrix with the updated elements
    b_matrix: array
        The full matrix that needs updating
    mask: array
        The location of the elements that need to be updated
    n_params: integer
        The number of parameters in the state
    Returns
    --------
    The updated `b_matrix`"""

    mask = mask.ravel()
    if a_matrix.ndim == 1:
        n = mask.shape[0] # big dimension
        n_good = mask.sum()
        ilocs = mask.nonzero()[0]
        for i in xrange(n_params):
            b_matrix[ilocs +i*n] = a_matrix[(i*n_good):((i+1)*n_good)]
    elif a_matrix.ndim == 2:
        n = mask.shape[0] # big dimension
        n_good = mask.sum()
        ilocs = mask.nonzero()[0]
        for i in xrange(n_params):
            ii = 0
            for j in xrange(n):
                if mask[j]:
                    b_matrix[j + i*n, i*n + ilocs] = a_matrix[ii,
                                                     (i*n_good):((i+1)*n_good)]
                    ii += 1
    return b_matrix
