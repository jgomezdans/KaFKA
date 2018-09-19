"""
Extracted the state propagation bits to individual functions
"""
import logging

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl


class IdentityPropagator(object):
    """An identity observation operator propagator class."""
    def __init__(self, q_diag, n_params, mask):
        """Takes a vector with the Q factors associated with the 
        model uncertainty per parameter, and we assume we have
        `n_params` and the boolean mask to figure out the number
        of pixels."""
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



def forward_state_propagation(x_analysis, P_analysis, P_analysis_inv,
                              M_matrix, Q_matrix, 
                              time_step, prior_obj):
    """Forward state propagation. Spun out to its own function for 
    testability. Takes the state vector (x_analysis), the associated
    covariance (although this will always be `None` here ;D) and its
    inverse (which gets used). It also takes the model matrix and its
    associated covariance matrix, as well as the time step, and the 
    prior object. The function propagates the state through the 
    linear model given by `M_matrix`, inflates the uncertainty of
    the propagation by using `Q_matrix`, and it then combines this
    result with the Gaussian prior for the current time step.
    
    It returns the predicted state vector, and associated covariances."""
    
    ## TODO Needs to check sizes of matrices and vectors
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
    
