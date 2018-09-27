import os                                                                                                
import sys                                                                                               
sys.path.insert(0, '../')                                                                                
                                                                                                         
import numpy as np                       
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pytest import fixture                                                                               
from distutils import dir_util      

from kafka import NoPropagator, IdentityPropagator
from kafka import forward_state_propagation

class TestPrior(object):
    
    def __init__(self, n_params, n_pxls):
        self.n_params = n_params
        self.n_pxls = n_pxls
    
    def process_prior(self, time_step):
        x = np.ones(self.n_params*self.n_pxls)*5.
        inv_covar = sp.eye(self.n_params*self.n_pxls)*50.
        return x, inv_covar
        


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

def test_no_propagation_mean():
    #forward_state_propagation(x_analysis, P_analysis, P_analysis_inv,
                              #M_matrix, Q_matrix, 
                              #time_step, prior_obj):
    prior_obj = TestPrior(7, 1)
    xa = np.zeros(7) # Say
    P_analysis = None
    P_analysis_inv = sp.eye(7)
    P_analysis_inv.setdiag(np.ones(7)*100.)
    M_matrix = sp.eye(7)*0.
    Q_matrix = sp.eye(7)*0.
    time_step = None 
    xf, Pa, Pa_inv = forward_state_propagation(xa, P_analysis, P_analysis_inv,
                              M_matrix, Q_matrix, 
                              time_step, prior_obj)
    assert np.allclose(xf, np.ones(7)*5.)
    
def test_no_propagation_invcov():
    #forward_state_propagation(x_analysis, P_analysis, P_analysis_inv,
                              #M_matrix, Q_matrix, 
                              #time_step, prior_obj):
    prior_obj = TestPrior(7, 1)
    xa = np.zeros(7) # Say
    P_analysis = None
    P_analysis_inv = sp.eye(7)
    P_analysis_inv.setdiag(np.ones(7)*100.)
    M_matrix = sp.eye(7)*0.
    Q_matrix = sp.eye(7)*0.
    time_step = None 
    xf, Pa, Pa_inv = forward_state_propagation(xa, P_analysis, P_analysis_inv,
                              M_matrix, Q_matrix, 
                              time_step, prior_obj)
    assert np.allclose(np.array( (sp.eye(7)*50.).todense()),
                       np.array( (Pa_inv).todense()))
