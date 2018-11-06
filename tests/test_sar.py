import os                                                                                                
import sys                                                                                               
sys.path.insert(0, '../')                                                                                
                                                                                                         
import numpy as np                       
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pytest import fixture                                                                               
from distutils import dir_util      

from scipy.optimize import approx_fprime

from kafka.observation_operators import sar_observation_operator



def test_sar_vv():
    x = np.array([4.0, 0.1])
    sigma_vv, dsigma_vv = sar_observation_operator(x, np.array([23.0]), "VV")
    assert np.allclose( 0.15665851, sigma_vv)
    

def test_dsar_vv():
    x = np.array([4.0, 0.1])
    fwd_model = lambda p: sar_observation_operator(p, np.array([23.0]), "VV")[0]
    approx_grad = approx_fprime(x, fwd_model, 1e-6)
    sigma_vv, dsigma_vv = sar_observation_operator(x, np.array([23.0]), "VV")
    
    assert np.allclose( approx_grad, dsigma_vv)





def test_sar_vh():
    x = np.array([4.0, 0.1])
    sigma_vh, dsigma_vh = sar_observation_operator(x, np.array([23.0]), "VH")
    print(sigma_vh)
    assert np.allclose( 0.06595635, sigma_vh)
    

def test_dsar_vh():
    x = np.array([4.0, 0.1])
    fwd_model = lambda p: sar_observation_operator(p, np.array([23.0]), "VH")[0]
    approx_grad = approx_fprime(x, fwd_model, 1e-6)
    sigma_vv, dsigma_vh = sar_observation_operator(x, np.array([23.0]), "VH")
    
    assert np.allclose( approx_grad, dsigma_vh)

