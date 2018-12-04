import os                                                                                                
import sys                                                                                               
sys.path.insert(0, '../')                                                                                
                                                                                                         
import numpy as np                       
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pytest import fixture                                                                               
from distutils import dir_util      

from kafka import NoPropagator, IdentityPropagator, block_diag
from kafka.state_propagation import forward_state_propagation


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


def test_nopropagation_m():
    mask = np.random.rand(64, 64)
    mask = np.where (mask > 0.7, False, True)
    propagator = NoPropagator(np.array([0.1, 0.5]), 2, mask)
    M_true = np.zeros((2*mask.sum(), 2*mask.sum()))
    Q_true = np.zeros((2*mask.sum(), 2*mask.sum()))
    M, Q = propagator.get_matrices(np.zeros(2*mask.sum()),
                                None, None, None)
    assert np.allclose(np.linalg.norm(M_true), 
                       np.linalg.norm(M.todense())) 
    
def test_nopropagation_q():
    mask = np.random.rand(64, 64)
    mask = np.where (mask > 0.7, False, True)
    propagator = NoPropagator(np.array([0.1, 0.5]), 2, mask)
    M_true = np.zeros((2*mask.sum(), 2*mask.sum()))
    Q_true = np.zeros((2*mask.sum(), 2*mask.sum()))
    M, Q = propagator.get_matrices(np.zeros(2*mask.sum()),
                                None, None, None)
    assert np.allclose(np.linalg.norm(Q_true),
                       spl.norm(Q)) 
    
    

def test_identitypropagation_m():
    mask = np.random.rand(64, 64)
    mask = np.where (mask > 0.7, False, True)
    propagator = IdentityPropagator(np.array([0.1, 0.5]), 2, mask)
    M_true = np.eye(2*mask.sum())
    Q_true = np.zeros((2*mask.sum(), 2*mask.sum()))
    M, Q = propagator.get_matrices(np.zeros(2*mask.sum()),
                                None, None, None)
    assert np.allclose(np.linalg.norm(M_true), 
                       np.linalg.norm(M.todense())) 

def test_identitypropagation_q():
    mask = np.random.rand(64, 64)
    mask = np.where (mask > 0.7, False, True)
    propagator = IdentityPropagator(np.array([0.1, 0.5]), 2, mask)
    Q_true = sp.eye(2*mask.sum())
    q_diag = np.tile(np.array([0.1, 0.5]), mask.sum())
    Q_true.setdiag(q_diag)
    
    M, Q = propagator.get_matrices(np.zeros(2*mask.sum()),
                                None, None, None)
    assert np.allclose(spl.norm(Q_true), 
                       spl.norm(Q)) 

def test_forward_no_model_mean():
        parameter_list = ["w_vis", "x_vis", "a_vis",
                     "w_nir", "x_nir", "a_nir", "TeLAI"]
        mask = np.ones(1, dtype=np.bool)
        the_prior = JRCPrior(parameter_list, mask)
    
        x_analysis = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*2)])
        sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
        # The individual covariance matrix
        little_p = np.diag(sigma**2).astype(np.float32)
        little_p[5, 2] = 0.8862*0.0959*0.2
        little_p[2, 5] = 0.8862*0.0959*0.2
        P_analysis_inv = np.linalg.inv(little_p)
        M_matrix = np.zeros((7,7))
        #M_matrix[-1, -1] = 1.
    
        Q_matrix = np.diag(np.zeros(7))
        #Q_matrix[-1, -1] = 0.1 # Say
        xf, _, B = forward_state_propagation(x_analysis, None, P_analysis_inv,
                              M_matrix, Q_matrix, 
                              1, the_prior)
        
        assert np.allclose(xf, x_analysis)


def test_forward_no_model_inv_cov():
        parameter_list = ["w_vis", "x_vis", "a_vis",
                     "w_nir", "x_nir", "a_nir", "TeLAI"]
        mask = np.ones(1, dtype=np.bool)
        the_prior = JRCPrior(parameter_list, mask)
    
        x_analysis = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*2)])
        sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
        # The individual covariance matrix
        little_p = np.diag(sigma**2).astype(np.float32)
        little_p[5, 2] = 0.8862*0.0959*0.2
        little_p[2, 5] = 0.8862*0.0959*0.2
        P_analysis_inv = np.linalg.inv(little_p)
        M_matrix = np.zeros((7,7))
        #M_matrix[-1, -1] = 1.
    
        Q_matrix = np.diag(np.zeros(7))
        #Q_matrix[-1, -1] = 0.1 # Say
        xf, _, B = forward_state_propagation(x_analysis, None, P_analysis_inv,
                              M_matrix, Q_matrix, 
                              1, the_prior)
        
        assert np.allclose(B, P_analysis_inv)
