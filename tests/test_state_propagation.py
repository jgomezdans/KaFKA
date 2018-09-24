import os                                                                                                
import sys                                                                                               
sys.path.insert(0, '../')                                                                                
                                                                                                         
import numpy as np                       
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pytest import fixture                                                                               
from distutils import dir_util      

from kafka import NoPropagator, IdentityPropagator




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
