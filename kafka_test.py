import logging
logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="/tmp/stuff2.log")
import os
import numpy as np

import datetime
import kafka
from kafka.input_output import BHRObservations, KafkaOutput
from kafka import LinearKalman
from kafka.inference import block_diag
from kafka.inference import propagate_information_filter_LAI
from kafka.inference import no_propagation
#from kafka.inference import propagate_information_filter

emulator = "./SAIL_emulator_both_500trainingsamples.pkl"
tile = "h17v05"
start_time = "2017001"
if os.path.exists("/storage/ucfajlg/Ujia/MCD43/"):
    mcd43a1_dir = "/storage/ucfajlg/Ujia/MCD43/"
else:
    mcd43a1_dir="/data/selene/ucfajlg/Ujia/MCD43/"
####tilewidth = 75
###n_pixels = tilewidth*tilewidth
mask = np.zeros((2400,2400),dtype=np.bool8)
mask[900:940, 1300:1340] = True # Alcornocales
mask[640:700, 1400:1500] = True # Campinha
mask[650:730, 1180:1280] = True # Arros
mask[700:705, 1200] = True

bhr_data =  BHRObservations(emulator, tile, mcd43a1_dir, start_time,
###                            ulx=1200, uly=650, dx=tilewidth, dy=tilewidth,
                            end_time=None, mcd43a2_dir=None)

projection, geotransform = bhr_data.define_output()

output = KafkaOutput( geotransform, projection, "/tmp/")

#vis=bhr_data.get_band_data(datetime.datetime(2017,8,1), "vis")
#nir=bhr_data.get_band_data(datetime.datetime(2017,8,1), "nir")

kf = LinearKalman(bhr_data, output, mask,
                  state_propagation=propagate_information_filter_LAI,
                  n_params=7, 
                  bands_per_observation=2, linear=False)

# Need to set up prior
# Defining the prior
n_pixels = mask.sum()
sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5]) # broadly TLAI 0->7 for 1sigma
x0 = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*1.5)])
x0 = np.array([x0 for i in xrange(n_pixels)]).flatten()
# The individual covariance matrix
little_p = np.diag ( sigma**2).astype(np.float32)
little_p[5,2] = 0.8862*0.0959*0.2
little_p[2,5] = 0.8862*0.0959*0.2

inv_p = np.linalg.inv(little_p)
xlist = [inv_p for m in xrange(n_pixels)]


P_forecast_inv=block_diag(xlist, dtype=np.float32)


Q = np.tile(sigma*1., n_pixels)
#Q[6::7] = np.exp(-1./2.)**2 # TLAI


kf.set_trajectory_model()
kf.set_trajectory_uncertainty(Q)
# Also time grid, let's try to do this with datetime objects and see 
# how far we get?
base = datetime.datetime(2017,7,1)
num_days = 60
time_grid = list((base + datetime.timedelta(days=x) 
                  for x in range(0, num_days, 8)))
kf.run(time_grid, x0, None, P_forecast_inv, iter_obs_op=True)




