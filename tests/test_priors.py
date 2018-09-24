import os                                                                                                
import sys                                                                                               
sys.path.insert(0, '../')                                                                                
                                                                                                         
import numpy as np                       
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pytest import fixture                                                                               
from distutils import dir_util      

from kafka import JRCPrior




def test_jrc_mean():
    mask = np.random.rand(64, 64)
    mask = np.where (mask > 0.7, False, True)
    jrc_prior = JRCPrior(["w_vis", "x_vis", "a_vis",
                           "w_nir", "x_nir", "a_nir", "tlai"],
                            mask)
    n_pixels = mask.sum()
    x0, inv_covar = jrc_prior.process_prior (None, inv_cov=True)
    mu = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*2.)])
    x0_test =  np.array([mu for i in range(n_pixels)]).flatten()
    assert np.allclose( x0, x0_test)
    
def test_jrc_inv_covar():
    mask = np.array([True], dtype=np.bool)
    n_pixels = 1
    jrc_prior = JRCPrior(["w_vis", "x_vis", "a_vis",
                           "w_nir", "x_nir", "a_nir", "tlai"],
                            mask)
    x0, inv_covar = jrc_prior.process_prior (None, inv_cov=True)
    sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.6])
    # The individual covariance matrix
    little_p = np.diag(sigma**2).astype(np.float32)
    little_p[5, 2] = 0.8862*0.0959*0.2
    little_p[2, 5] = 0.8862*0.0959*0.2
    inv_p = np.linalg.inv(little_p)
    inv_covar_test =  np.array([inv_p for i in range(n_pixels)]).flatten()
    assert np.allclose(spl.norm(inv_covar), np.linalg.norm(inv_covar_test))
