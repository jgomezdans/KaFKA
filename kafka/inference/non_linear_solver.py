import numpy as np
import scipy.optimize

import scipy.sparse as sp

def non_linear_solver(xforecast, Pinv, current_data, state_mask,
                        n_params, band_mapper):
    Y = []
    MASK = []
    UNC = []
    META = []

    for band, data in enumerate(current_data):
        n_good_obs = np.sum(data.mask * state_mask)
        if n_good_obs > 0:
                # Calculate H matrix & H0 vector around the linearisation
                # point given by x_prev
            #                H_matrix_= self._create_observation_operator(n_params,
            #                                                            data.emulator,
            #                                                            data.metadata,
            #                                                            data.mask,
            #                                                            state_mask,
            #                                                            x_prev,
            #                                                            band,
            #                                                band_mapper = band_mapper)
            Y.append(data.observations)
            MASK.append(data.mask)
            UNC.append(data.uncertainty)
            META.append(data.metadata)
        else:
            print(f"Band {band+1:d} didn't have unmasked pixels")
    
    def cost_band(x0):
        cost = 0.
        dcost = np.zeros_like(x0)
        total_residual = 0.
        for band in range(len(Y)):
            maski = current_data[band].mask * state_mask
            maskf = maski.astype(np.float32)
            x_vect = x0.reshape([-1, n_params]) # array of single pixel parameters
            # from this x_vect, one needs to remove the `data.mask` pixels
            H, _, dH = current_data[band].emulator.predict(x_vect, do_unc=False)
            pred = np.zeros_like(Y[band])
            pred[state_mask] = H
            C_obs_inv = sp.diags(
                UNC[band].diagonal()[state_mask.flatten()])

            residual = ((pred-Y[0])*maskf)[state_mask]
            log_like = 0.5*(residual@C_obs_inv)@residual
            dlog_like = (dH*(C_obs_inv@residual)[:, None]).flatten() 
            log_prior = 0.5*(x0-xforecast).T@Pinv@(x0 - xforecast)
            dlog_prior = Pinv@(x0 - xforecast)
            total_residual += (residual**2).sum()
            cost += log_like + log_prior
            dcost += dlog_like + dlog_prior
        print(total_residual)
        return cost, dcost
    
    retval = scipy.optimize.minimize(cost_band, xforecast, method='L-BFGS-B',
                                     jac=True,options={"disp":True})
    return retval.x
    
