"""
Extracted the state propagation bits to individual functions
"""
import logging

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl

#from .utils import block_diag

class IdentityPropagator(object):
    def __init__(self, q_diag, n_params, mask):
        self.n_params = n_params
        self.mask = mask
        self.n_elements = mask.sum()
        assert q_diag.shape[0] == self.n_params
        self.q_diag = q_diag
        
    def get_matrices(self, x, P, P_inv, timestep):
        M_matrix = sp.eye(self.n_params*self.n_elements,
                          self.n_params*self.n_elements,
                          format="csr")
        Q_matrix = sp.eye(self.n_params*self.n_elements,
                          self.n_params*self.n_elements,
                          format="csr")
        Q_diag = np.tile(self.q_diag, self.n_elements)
        Q_matrix.setdiag(Q_diag)
        return M_matrix, Q_matrix


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




class NoHessianMethod(Exception):
    """An exception triggered when the forward model isn't able to provide an
    estimation of the Hessian"""
    def __init__(self, message):
        self.message = message
def band_selecta(band):
    if band == 0:
        return np.array([0, 1, 6, 2])
    else:
        return np.array([3, 4, 6, 5])


def hessian_correction_pixel(gp, x0, C_obs_inv, innovation, band, nparams):
    selecta = band_selecta(band)
    ddH = gp.hessian(np.atleast_2d(x0[selecta]))
    big_ddH = np.zeros((nparams, nparams))
    for i, ii in enumerate(selecta):
        for j, jj in enumerate(selecta):
            big_ddH[ii, jj] = ddH.squeeze()[i, j]
    big_hessian_corr = big_ddH*C_obs_inv*innovation
    return big_hessian_corr


def hessian_correction(gp, x0, R_mat, innovation, mask, state_mask, band,
                       nparams):
    """Calculates higher order Hessian correction for the likelihood term.
    Needs the GP, the Observational uncertainty, the mask...."""
    if not hasattr(gp, "hessian"):
        # The observation operator does not provide a Hessian method. We just
        # return 0, meaning no Hessian correction.
        return 0.
    C_obs_inv = R_mat.diagonal()[state_mask.flatten()]
    mask = mask[state_mask].flatten()
    little_hess = []
    for i, (innov, C, m) in enumerate(zip(innovation, C_obs_inv, mask)):
        if not m:
            # Pixel is masked
            hessian_corr = np.zeros((nparams, nparams))
        else:
            # Get state for current pixel
            x0_pixel = x0.squeeze()[(nparams*i):(nparams*(i + 1))]
            # Calculate the Hessian correction for this pixel
            hessian_corr = m * hessian_correction_pixel(gp, x0_pixel, C,
                                                        innov, band, nparams)
        little_hess.append(hessian_corr)
    hessian_corr = block_diag(little_hess)
    return hessian_corr


def hessian_correction_multiband(gp, x0, R_mats, innovations, masks, state_mask, n_bands,
                       nparams):
    """ Non linear correction for the Hessian of the cost function. This handles
    multiple bands. """
    little_hess_cor = []
    for R, innovation, mask, band in zip(R_mats, innovations, masks, range(n_bands)):
        little_hess_cor.append(hessian_correction(gp, x0, R, innovation, mask, state_mask, band,
                       nparams))
    hessian_corr = sum(little_hess_cor) #block_diag(little_hess_cor)
    return hessian_corr



#_advance(x_analysis, P_analysis, P_analysis_inverse,
#                          prior=self.prior, date=self.current_timestep,
#                          state_propagator=self._state_propagator)
def forward_state_propagation(x_analysis, P_analysis, P_analysis_inv,
                              M_matrix, Q_matrix, 
                              time_step, prior_obj):
    # The next few lines set up propagting the state
    # given a linear(ised) model M and the input
    # state and associated inverse covariance matrix
    A_inv = M_matrix @ P_analysis_inv @ M_matrix.T
    # Set up the approximate inflation matrix
    C_matrix_diagonal = 1./(1 + A_inv.diagonal()*Q.diagonal())
    C_matrix = sp.eye(*P_analysis_inv.shape)
    C_matrix.setdiag(C_matrix_diagonal)
    # Do propagate the state now
    P_forecast_inv = C_matrix @ A_inv
    x_forecast = M_matrix @ x_analysis
    
    # Retrieve the prior distribution from the prior object
    mu_prior, c_prior_inv = prior_obj.process_prior(time_step)
    B = c_prior_inv + P_forecast_inv
    y = c_prior_inv @ mu_prior + P_forecast_inv @ x_forecast
    BI = sp.linalg.splu(B)
    x_merged = BI.solve(y)
    return x_merged, B



class JRCPrior(object):
    """Dummpy 2.7/3.6 prior class following the same interface as 3.6 only
    version."""

    def __init__ (self, parameter_list, state_mask):
        """It makes sense to have the list of parameters and state mask
        defined at this point, as they won't change during processing."""
        self.mean, self.covar, self.inv_covar = self._tip_prior() 
        self.parameter_list = parameter_list
        if isinstance(state_mask, (np.ndarray, np.generic) ):
            self.state_mask = state_mask
        else:
            self.state_mask = self._read_mask(state_mask)
            
    def _read_mask(self, fname):
        """Tries to read the mask as a GDAL dataset"""
        if not os.path.exists(fname):
            raise IOError("State mask is neither an array or a file that exists!")
        g = gdal.Open(fname)
        if g is None:
            raise IOError("{:s} can't be opened with GDAL!".format(fname))
        mask = g.ReadAsArray()
        return mask

    def _tip_prior(self):
        """The JRC-TIP prior in a convenient function which is fun for the whole
        family. Note that the effective LAI is here defined in transformed space
        where TLAI = exp(-0.5*LAIe).

        Returns
        -------
        The mean prior vector, covariance and inverse covariance matrices."""
        # broadly TLAI 0->7 for 1sigma
        sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
        x0 = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*2.)])
        # The individual covariance matrix
        little_p = np.diag(sigma**2).astype(np.float32)
        little_p[5, 2] = 0.8862*0.0959*0.2
        little_p[2, 5] = 0.8862*0.0959*0.2

        inv_p = np.linalg.inv(little_p)
        return x0, little_p, inv_p

    def process_prior ( self, time, inv_cov=True):
        # Presumably, self._inference_prior has some method to retrieve 
        # a bunch of files for a given date...
        n_pixels = self.state_mask.sum()
        x0 = np.array([self.mean for i in range(n_pixels)]).flatten()
        if inv_cov:
            inv_covar_list = [self.inv_covar for m in range(n_pixels)]
            inv_covar = block_diag(inv_covar_list, dtype=np.float32)
            return x0, inv_covar
        else:
            covar_list = [self.covar for m in range(n_pixels)]
            covar = block_diag(covar_list, dtype=np.float32)
            return x0, covar
        
if __name__ == "__main__":
    
    parameter_list = ["w_vis", "x_vis", "a_vis",
                     "w_nir", "x_nir", "a_nir", "TeLAI"]
    mask = np.ones(1, dtype=np.bool)
    the_prior = JRCPrior(parameter_list, mask)
    
    x_analysis = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*2)])
    sigma = 2*np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
    # The individual covariance matrix
    little_p = np.diag(sigma**2).astype(np.float32)
    little_p[5, 2] = 0.8862*0.0959*0.2
    little_p[2, 5] = 0.8862*0.0959*0.2
    P_analysis_inv = np.linalg.inv(little_p)
    M_matrix = np.zeros((7,7))
    #M_matrix[-1, -1] = 1.
    
    Q_matrix = np.diag(np.zeros(7))
    Q_matrix[-1, -1] = 0.1 # Say
    xf, Pf_i = forward_state_propagation(M_matrix, Q_matrix,
                              x_analysis, None, P_analysis_inv,
                              1, the_prior)
    
