#!/usr/bin/env python
import logging
logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="the_log.log")
import os
from datetime import datetime, timedelta

import numpy as np

import kafka
from kafka.input_output import BHRObservations, KafkaOutput, get_chunks
from kafka import LinearKalman
from kafka.inference import block_diag
from kafka.inference import create_nonlinear_observation_operator
from kafka.state_propagation import IdentityPropagator, NoPropagator
from kafka.priors import JRCPrior

from InferenceInterface import kafka_inference

if __name__ == "__main__":
    parameter_list = ["w_vis", "x_vis", "a_vis",
                     "w_nir", "x_nir", "a_nir", "TeLAI"]
    tile = "h17v05"
    start_time = "2017001"
    band_mapper = [np.array([0, 1, 6, 2]),
                   np.array([3, 4, 6, 5])]
    emulator = "./SAIL_emulator_both_500trainingsamples.pkl"
    
    if os.path.exists("/storage/ucfajlg/Ujia/MCD43/"):
        mcd43a1_dir = "/storage/ucfajlg/Ujia/MCD43/"
    else:
        mcd43a1_dir="/data/selene/ucfajlg/Ujia/MCD43"
    mask = np.zeros((2400,2400),dtype=np.bool8)
    mask[900:940, 1300:1340] = True # Alcornocales
#    mask[640:700, 1400:1500] = True # Campinha
    mask[650:730, 1180:1280] = True # Arros
    #mask[ 2200:2395, 450:700 ] = True # Bondville, h11v04
    #mask = province_mask(provinces=["Córdoba"])

    bhr_data =  BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                                end_time=None, mcd43a2_dir=None)
    base = datetime(2017,5,1)
    num_days = 180
    time_grid = []
    for x in range( 0, num_days, 16):
        time_grid.append( base + timedelta(days = x) )
        
    
    Q = np.array([100., 1e5, 1e2, 100., 1e5, 1e2, 100.])
    state_propagator = IdentityPropagator(Q, 7, mask)
    prior = JRCPrior
    kafka_inference(mask, time_grid, parameter_list,
                    bhr_data, prior, state_propagator,
                    "/data/selene/ucfajlg/tmp/", 
                    band_mapper, None)





