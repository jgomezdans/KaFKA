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
import datetime as dt
import os
import gdal

import logging
LOG = logging.getLogger(__name__)


def iterate_time_grid(time_grid, the_dates):
    is_first = True
    istart_date = time_grid[0]
    for ii, timestep in enumerate(time_grid[1:]):
        # First locate all available observations for time step of interest.
        # Note that there could be more than one...
        locate_times_idx = np.where(np.logical_and(
            np.array(the_dates) >= istart_date,
            np.array(the_dates) < timestep), True, False)
        locate_times = np.array(the_dates)[locate_times_idx]
        LOG.info("Doing timestep from {} -> {}".format(
                istart_date.strftime("%Y-%m-%d"),
                timestep.strftime("%Y-%m-%d")))
        LOG.info("# of Observations: %d" % len(locate_times))
        for iobs in locate_times:
                LOG.info("\t->{}".format(iobs.strftime("%Y-%m-%d")))
        istart_date = timestep
        if is_first:
            yield timestep, locate_times, True
            is_first = False
        else:
            yield timestep, locate_times, False


def run_emulator_NEW(gp, x, tol=None):
    # We select the unique values in vector x
    # Note that we could have done this using e.g. a histogram
    # or some other method to select solutions "close enough"
    unique_vectors, unique_indices, unique_inverse = np.unique(
        x, axis=0, return_index=True, return_inverse=True)
    if len(unique_vectors) == 1:  # Prior!
        cluster_labels = np.zeros(x.shape[0], dtype=np.int16)
    elif len(unique_vectors) > 1e6:

        LOG.info("Clustering parameter space")
        mean = np.mean(x, axis=0)  # 7 dimensions
        cov = np.cov(x, rowvar=0)  # 4 x 4 dimensions
        # Draw a 300 element LUT
        unique_vectors = np.random.multivariate_normal(mean, cov,
                                                       5000)
        # Assign each element of x to a LUT/cluster entry
        cluster_labels = locate_in_lut(unique_vectors, x)
    # Runs emulator for emulation subset
    prediction = gp.predict(unique_vectors, do_unc=False)
    if len(prediction) == 2:
        H_, dH_ = prediction
    else:
        H_, _, dH_ = prediction

    H = np.zeros(x.shape[0])
    dH = np.zeros_like(x)
    if 'cluster_labels' in locals():
        nclust = cluster_labels.shape
        for label in np.unique(cluster_labels):
            H[cluster_labels == label] = H_[label]
            dH[cluster_labels == label, :] = dH_[label, :]
        return H, dH
    H[unique_indices] = H_
    H = H[unique_inverse]
    dH[unique_indices] = dH_
    dH = dH[unique_inverse]
    return H, dH


def run_emulator(gp, x, tol=None):
    # We select the unique values in vector x
    # Note that we could have done this using e.g. a histogram
    # or some other method to select solutions "close enough"
    unique_vectors = np.vstack({tuple(row) for row in x})
    if len(unique_vectors) == 1:  # Prior!
        cluster_labels = np.zeros(x.shape[0], dtype=np.int16)
    elif len(unique_vectors) > 1e6:

        LOG.info("Clustering parameter space")
        mean = np.mean(x, axis=0)  # 7 dimensions
        cov = np.cov(x, rowvar=0)  # 4 x 4 dimensions
        # Draw a 300 element LUT
        unique_vectors = np.random.multivariate_normal(mean, cov,
                                                       5000)
        # Assign each element of x to a LUT/cluster entry
        cluster_labels = locate_in_lut(unique_vectors, x)
    # Runs emulator for emulation subset
    try:
        H_, dH_ = gp.predict(unique_vectors, do_unc=False)
    except ValueError:
        # Needed for newer gp version
        H_, _, dH_ = gp.predict(unique_vectors, do_unc=False)

    H = np.zeros(x.shape[0])
    dH = np.zeros_like(x)
    try:
        nclust = cluster_labels.shape
    except NameError:
        for i, uniq in enumerate(unique_vectors):
            passer = np.all(x == uniq, axis=1)
            H[passer] = H_[i]
            dH[passer, :] = dH_[i, :]
        return H, dH

    for label in np.unique(cluster_labels):
        H[cluster_labels == label] = H_[label]
        dH[cluster_labels == label, :] = dH_[label, :]
    return H, dH    

def create_uncertainty(uncertainty, mask):
    """Creates the observational uncertainty matrix. We assume that
    uncertainty is a single value and we return a diagonal matrix back.
    We present this diagonal **ONLY** for pixels that have observations
    (i.e. not masked)."""
    good_obs = mask.sum()
    R_mat = np.ones(good_obs)*uncertainty*uncertainty
    return sp.dia_matrix((R_mat, 0), shape=(R_mat.shape[0], R_mat.shape[0]))


def create_linear_observation_operator(obs_op, n_params, metadata,
                                       mask, state_mask,
                                       x_forecast, band=None):
    """A simple **identity** observation opeartor. It is expected that you
    subclass and redefine things...."""
    good_obs = mask.sum()  # size of H_matrix
    H_matrix = sp.dia_matrix(np.eye(good_obs))
    return H_matrix



def create_nonlinear_observation_operator(n_params, emulator, metadata,
                                          mask, state_mask,  x_forecast, band,
                                          band_mapper=None):
    """Using an emulator of the nonlinear model around `x_forecast`.
    This case is quite special, as I'm focusing on a BHR SAIL
    version (or the JRC TIP), which have spectral parameters
    (e.g. leaf single scattering albedo in two bands, etc.). This
    is achieved by using the `state_mapper` to select which bits
    of the state vector (and model Jacobian) are used."""
    LOG.info("Creating the ObsOp for band %d" % band)
    n_times = int( x_forecast.shape[0] / n_params )
    good_obs = mask.sum()

    H_matrix = sp.lil_matrix((n_times, n_params * n_times),
                             dtype=np.float32)
    H0 = np.zeros(n_times, dtype=np.float32)


    if band_mapper is not None:
        state_mapper = band_mapper[band]
    else:
        state_mapper = np.arange(n_params, dtype=np.int16)


    # This loop can be JIT'ed
    x0 = np.zeros((n_times, n_params))
    for i, m in enumerate(mask[state_mask].flatten()):
        if m:
            x0[i, :] = x_forecast[(n_params*i) + state_mapper]
    LOG.info("Running emulators")
    # Calls the run_emulator method that only does different vectors
    # It might be here that we do some sort of clustering

    H0_, dH = run_emulator(emulator, x0[mask[state_mask]])

    LOG.info("Storing emulators in H matrix")
    # This loop can be JIT'ed too
    n = 0
    for i, m in enumerate(mask[state_mask].flatten()):
        if m:
            H_matrix[i, state_mapper + n_params * i] = dH[n]
            H0[i] = H0_[n]
            n += 1

    LOG.info("\tDone!")
    if hasattr(emulator, "hessian"):
        calc_hess = True
    else:
        calc_hess = False

    if calc_hess:
        ddH = emulator.hessian(x0[mask[state_mask]])
        hess = np.zeros((n_times, n_params, n_params))
        for n, (lil_hess, m) in enumerate(zip(ddH, 
                                              mask[state_mask].flatten())):
            if m:
                big_hess = np.zeros((n_params, n_params))
                for i, ii in enumerate(state_mapper):
                    for j, jj in enumerate(state_mapper):
                        big_hess[ii, jj] = lil_hess.squeeze()[i, j]
                hess[n,...] = big_hess

    return (H0, H_matrix.tocsr(), hess) if calc_hess else (H0, H_matrix.tocsr())


def create_prosail_observation_operator(n_params, emulator, metadata,
                                          mask, state_mask,  x_forecast,
                                          band, calc_hess=False, 
                                          band_mapper=None):
    """Using an emulator of the nonlinear model around `x_forecast`.
    This case is quite special, as I'm focusing on a BHR SAIL
    version (or the JRC TIP), which have spectral parameters
    (e.g. leaf single scattering albedo in two bands, etc.). This
    is achieved by using the `state_mapper` to select which bits
    of the state vector (and model Jacobian) are used."""
    LOG.info("Creating the ObsOp for band %d" % band)
    n_times = int( x_forecast.shape[0] / n_params )
    good_obs = mask.sum()

    H_matrix = sp.lil_matrix((n_times, n_params * n_times),
                             dtype=np.float32)
    H0 = np.zeros(n_times, dtype=np.float32)

    # This loop can be JIT'ed
    x0 = np.zeros((n_times, n_params))
    for i, m in enumerate(mask[state_mask].flatten()):
        if m:
            x0[i, :] = x_forecast[(n_params*i):(n_params*(i+1))]
    LOG.info("Running emulators")
    # Calls the run_emulator method that only does different vectors
    # It might be here that we do some sort of clustering

    H0_, dH = run_emulator(emulator, x0[mask[state_mask]])

    LOG.info("Storing emulators in H matrix")
    # This loop can be JIT'ed too
    n = 0
    for i, m in enumerate(mask[state_mask].flatten()):
        if m:
            H_matrix[i,  (n_params*i):(n_params*(i+1))] = dH[n]
            H0[i] = H0_[n]
            n += 1

    LOG.info("\tDone!")

    if calc_hess:
        hess = np.zeros((n_times, n_params, n_params),
                             dtype=np.float32)
        hess_ = emulator.hessian(x0[mask[state_mask]])

        n = 0
        for i, m in enumerate(mask[state_mask].flatten()):
            if m:
                hess[i, ...] = hess_[n]
                n += 1

    return (H0, H_matrix.tocsr(), hess) if calc_hess \
                                        else (H0, H_matrix.tocsr())





def locate_in_lut(lut, im):
    """This function locates a samples nearest neighbour in another dataset.
    We assume that `lut` is `[m, np]` and `im` is `[n, np]`, where `n >> m`
    and `np` is not too big. We will look for the location of the row of
    `lut` that is closest to each row in `im`.
    It returns `idx`, an array with an integer index to the first dimension
    of lut."""
    assert (lut.shape[1] == im.shape[1])
    idx = np.linalg.norm(lut[:, None, :] - im, axis=2).argmin(axis=0)
    return idx


# This is a faster version for equally-sized blocks.
# Currently, open PR on scipy's github
# (https://github.com/scipy/scipy/pull/5619)
def block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix from provided matrices.

    Parameters
    ----------
    mats : sequence of matrices
        Input matrices. Can be any combination of lists, numpy.array,
         numpy.matrix or sparse matrix ("csr', 'coo"...)
    format : str, optional
        The sparse format of the result (e.g. "csr").  If not given, the matrix
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix.  If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Providing a sequence of equally shaped matrices
     will provide marginally faster results

    .. versionadded:: 0.18.0

    See Also
    --------
    bmat, diags, block_diag

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, block_diag
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5, 6], [7, 8]])
    >>> C = coo_matrix([[9, 10], [11,12]])
    >>> block_diag((A, B, C)).toarray()
    array([[ 1,  2,  0,  0,  0,  0],
           [ 3,  4,  0,  0,  0,  0],
           [ 0,  0,  5,  6,  0,  0],
           [ 0,  0,  7,  8,  0,  0],
           [ 0,  0,  0,  0,  9, 10],
           [ 0,  0,  0,  0, 11, 12]])
    """
    import scipy.sparse as sp
    import scipy.sparse.sputils as spu
    from scipy.sparse.sputils import upcast, get_index_dtype

    from scipy.sparse.csr import csr_matrix
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.bsr import bsr_matrix
    from scipy.sparse.coo import coo_matrix
    from scipy.sparse.dia import dia_matrix

    from scipy.sparse import issparse

    n = len(mats)
    mats_ = [None] * n
    for ia, a in enumerate(mats):
        if hasattr(a, 'shape'):
            mats_[ia] = a
        else:
            mats_[ia] = coo_matrix(a)

    if any(mat.shape != mats_[-1].shape for mat in mats_) or (
            any(issparse(mat) for mat in mats_)):
        data = []
        col = []
        row = []
        origin = np.array([0, 0], dtype=np.int)
        for mat in mats_:
            if issparse(mat):
                data.append(mat.data)
                row.append(mat.row + origin[0])
                col.append(mat.col + origin[1])

            else:
                data.append(mat.ravel())
                row_, col_ = np.indices(mat.shape)
                row.append(row_.ravel() + origin[0])
                col.append(col_.ravel() + origin[1])

            origin += mat.shape

        data = np.hstack(data)
        col = np.hstack(col)
        row = np.hstack(row)
        total_shape = origin
    else:
        shape = mats_[0].shape
        data = np.array(mats_, dtype).ravel()
        row_, col_ = np.indices(shape)
        row = (np.tile(row_.ravel(), n) +
               np.arange(n).repeat(shape[0] * shape[1]) * shape[0]).ravel()
        col = (np.tile(col_.ravel(), n) +
               np.arange(n).repeat(shape[0] * shape[1]) * shape[1]).ravel()
        total_shape = (shape[0] * n, shape[1] * n)

    return coo_matrix((data, (row, col)), shape=total_shape).asformat(format)


def spsolve2(a, b):
    a_lu = spl.splu(a.tocsc())   # LU decomposition for sparse a
    out = sp.lil_matrix((a.shape[1], b.shape[1]), dtype=np.float32)
    b_csc = b.tocsc()
    for j in range(b.shape[1]):
        bb = np.array(b_csc[j, :].todense()).squeeze()
        out[j, j] = a_lu.solve(bb)[j]
    return out.tocsr()

def robust_inflation(n_bands, innovations, observations, uncertainty,
                        state_mask, tolerance=6.1):
        innovations = np.split(innovations, n_bands)
        nn, mm = uncertainty[0].shape
        M = np.zeros((n_bands,*observations[0].shape))
        unc = np.zeros((n_bands, *observations[0].shape))
        for i in range(n_bands):
            M[i, state_mask] = innovations[i]
            unc[i, state_mask] = np.sqrt(1./uncertainty[i].diagonal()[
                state_mask.flatten()])
        passer = unc == 0
        unc[passer] = np.nan
        z_score = M*0.
        z_score[~passer] = M[~passer]/unc[~passer]
        outliers = z_score > tolerance
        outliers *= state_mask    
        new_mask = []             

        for i in range(n_bands):
            LOG.info(f"Searching outliers band{i:d}. Found " +
                f"{(outliers[i]).sum():d}")
            #the_fxxing_mx = sp.lil_matrix((nn, mm))
            #diag = uncertainty[i].diagonal()
            #diag[(outliers[i, :, :]).ravel()] /= (100.)**2
            #the_fxxing_mx.setdiag(diag)
            #new_unc.append(the_fxxing_mx)
            new_mask.append(outliers[i])
            
        
        return new_mask
