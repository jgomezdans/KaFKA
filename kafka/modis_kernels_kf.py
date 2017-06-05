#!/usr/bin/env python
"""A fast Kalman filter implementation designed with raster data in mind. This
implementation basically performs a very fast update of the filter."""

# KaFKA A fast Kalman filter implementation for raster based datasets.
# Copyright (c) 2017 J Gomez-Dans. All rights reserved.
#
# This file is part of KaFKA.
#
# KaFKA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KaFKA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KaFKA.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "J Gomez-Dans, N Pounder"
__copyright__ = "Copyright 2017 J Gomez-Dans, N Pounder"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

from collections import namedtuple
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"

import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

import gdal

from linear_kernels_kf import KernelLinearKalman

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG, 
                        format=
                        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
LOG = logging.getLogger(__name__)


# metadata is now different as it has angles innit
Metadata = namedtuple('Metadata', 'mask uncertainty vza sza raa')
MODIS_observations = namedtuple('MODIS_observations',
                        'doys qa vza sza saa vaa b01 b02 b03 b04 b05 b06 b07')


class MODISKernelLinearKalman (KernelLinearKalman):
    """
    A class to specifically deal with MODIS M?D09GA data sets pre-processed
    by my VRT software. Metadata now comes from the actual files, so needs to be
    created there.
    *NOTE* `observation_times` should refer to VRT bands, I think...
    """
    def __init__(self, observations, observation_times,
                 output_array, output_unc, the_band=1):
        observation_metadata=None
        KernelLinearKalman.__init__(self, observations, observation_times,
                              observation_metadata, output_array, output_unc)
        self.the_band = the_band
        
    def _dump_output(self, step, timestep, x_analysis, P_analysis, 
                     P_analysis_inverse):

        outputs = {}
        outputs['iso'] =  x_analysis[:(2400*2400)].reshape((2400,2400))
        outputs['vol'] =  x_analysis[(2400*2400):2*(2400*2400)].reshape((2400,2400))
        outputs['geo'] =  x_analysis[2*(2400*2400):3*(2400*2400)].reshape((2400,2400))

        outputs['iso_unc'] =  P_analysis.diagonal()[:(2400*2400)].reshape((2400,2400))
        outputs['vol_unc'] =  P_analysis.diagonal()[
            (2400*2400):2*(2400*2400)].reshape((2400,2400))
        outputs['geo_unc'] =  P_analysis.diagonal()[
            2*(2400*2400):3*(2400*2400)].reshape((2400,2400))

        outputs['bhr'] = outputs['iso']+outputs['vol']*0.189184 \
            - outputs['geo']*1.377622
        outputs['bhr_unc'] = outputs['iso_unc']+outputs['vol_unc']*0.189184 \
            - outputs['geo_unc']*1.377622


        LOG.info("saving. Timestep %d, step %d" % (timestep, step))
        for key, the_array in outputs.iteritems():
            self.output[key].GetRasterBand(timestep+1).WriteArray(the_array)
            self.output[key].GetRasterBand(timestep+1).SetMetadata(
                {'DoY':"%d"%(timestep)})


        if timestep % 10 == 0:
            for key, val in self.output.iteritems():
                val.FlushCache()
        #self.output['bhr'].FlushCache()
        #plt.imshow(x, interpolation='nearest', vmin=-0.1, vmax=0.5)
        #plt.title("%d-%d" % (step, timestep))
        #plt.show()
        #a=raw_input("Next?")
        #plt.close()
        

    def _get_observations_timestep(self, timestep, band=None):
        """A method that reads in MODIS reflectance data from VRT files produced
        from MOD09GA/MYD09GA data sets.

        Parameters
        ----------
        timestep : This should be time step
        band : what MODIS band we're using (e.g. 1 to 7)

        Returns
        -------
        rho, R_mat, mask, metadata
        """
        LOG.info("Reading timestep >>> %d" % timestep)
        QA_OK = np.array([8, 72, 136, 200, 1032, 1288, 2056, 2120, 2184, 2248])
        qa = self.observations.qa.GetRasterBand(timestep+1).ReadAsArray()
        mask = np.in1d(qa, QA_OK).reshape((2400,2400))
        sza = self.observations.sza.GetRasterBand(timestep + 1).ReadAsArray()
        vza = self.observations.vza.GetRasterBand(timestep + 1).ReadAsArray()
        saa = self.observations.saa.GetRasterBand(timestep + 1).ReadAsArray()
        vaa = self.observations.vaa.GetRasterBand(timestep + 1).ReadAsArray()
        raa = vaa - saa
        rho_pntr = self.observations[5 + self.the_band]
        rho = rho_pntr.GetRasterBand(timestep + 1).ReadAsArray()/10000.
        # Taken from http://modis-sr.ltdri.org/pages/validation.html
        modis_uncertainty=np.array([0.005, 0.014, 0.008, 0.005, 0.012,
                                    0.006, 0.003])[self.the_band]
        R_mat = self.create_uncertainty(modis_uncertainty, mask)
        metadata = Metadata(mask, modis_uncertainty, 
                            vza/100., sza/100., raa/100.)
        return rho, R_mat, mask, metadata


def create_output_file(template_file, location, suffix,
                        n_bands=366,
                        elements=['iso', 'vol','geo',
                                    'iso_unc', 'vol_unc','geo_unc',
                                    'bhr', 'bhr_unc']):
    plethora = {}
    drv = gdal.GetDriverByName("GTiff")
    for el in elements:
        fname = os.path.join(location, "{}_{}.tif".format(el, suffix))
    
        dst_ds = drv.Create (fname, 2400, 2400, n_bands, gdal.GDT_Float32,
                            ['COMPRESS=DEFLATE', 'BIGTIFF=YES', 
                            'PREDICTOR=1', 'TILED=YES'])
        dst_ds.SetProjection(template_file.GetProjection())
        dst_ds.SetGeoTransform(template_file.GetGeoTransform())
        plethora[el] = dst_ds
    return plethora



def get_mcd43_prior(mcd43_fstring, band):
    g = gdal.Open(mcd43_fstring % band)
    # Deal with means...
    iso = g.GetRasterBand(1).ReadAsArray().flatten()
    vol = g.GetRasterBand(2).ReadAsArray().flatten()
    geo = g.GetRasterBand(3).ReadAsArray().flatten()
    
    iso = np.where (np.isnan(iso), 0.5, iso)
    vol = np.where (np.isnan(vol), 0.5, vol)
    geo = np.where (np.isnan(geo), 0.5, geo)
    
    n = 2400*2400
    
    #x_forecast = np.empty(3*n)
    #x_forecast[0::3] = iso
    #x_forecast[1::3] = vol
    #x_forecast[2::3] = geo
    x_forecast = np.r_[iso, vol, geo]

    
    # Deal with sigmas...
    iso = g.GetRasterBand(4).ReadAsArray().flatten()
    vol = g.GetRasterBand(5).ReadAsArray().flatten()
    geo = g.GetRasterBand(6).ReadAsArray().flatten()
    
    iso = np.where (np.isnan(iso), 0.5, iso)
    vol = np.where (np.isnan(vol), 0.5, vol)
    geo = np.where (np.isnan(geo), 0.5, geo)
    
    
    #sigma = np.empty(3*n)
    #sigma[0::3] = 1./(iso*iso)
    #sigma[1::3] = 1./(vol*vol)
    #sigma[2::3] = 1./(geo*geo)
    sigma= np.r_[iso, vol, geo]
    

    
    P_forecast = sp.eye(3*n, 3*n, format="csc", dtype=np.float32)
    P_forecast.setdiag(sigma**2)
    
    return x_forecast, P_forecast



if __name__ == "__main__":

    if os.path.exists("/storage/ucfajlg/Ujia/"):
        the_dir="/storage/ucfajlg/Ujia/"
    else:
        the_dir="/data/selene/ucfajlg/Ujia/"

    g = gdal.Open(the_dir + "brdf_2016_b01.vrt")
    days = np.array([int(g.GetRasterBand(i+1).GetMetadata()['DoY'])
                            for i in xrange(g.RasterCount)])
    band = 2
    time_offset = np.nonzero(days == 46)[0][0] # First time DoY 46 is mentioned
    days = days[np.logical_and(days >= 45, days <= 365)]
    
    modis_obs = MODIS_observations( days,
                gdal.Open(os.path.join(the_dir, "statekm_2016.vrt")),
                gdal.Open(os.path.join(the_dir, "SensorZenith_2016.vrt")),
                gdal.Open(os.path.join(the_dir, "SolarZenith_2016.vrt")),
                gdal.Open(os.path.join(the_dir, "SolarAzimuth_2016.vrt")),
                gdal.Open(os.path.join(the_dir, "SensorAzimuth_2016.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2016_b01.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2016_b02.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2016_b03.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2016_b04.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2016_b05.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2016_b06.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2016_b07.vrt")))
    output_plethora = create_output_file(g, "tmp/", "band{}_testSolver_lowinflate".format(band))
    
    kf = MODISKernelLinearKalman(modis_obs, days, output_plethora, [], the_band=band )
    kf.time_offset = time_offset
    n = 2400
    mcd43_fstring = "/data/selene/ucfajlg/" + \
                "Ujia/MCD43/MCD43_average_2016_001_030_b%d.tif"
    x_forecast, P_forecast = get_mcd43_prior(mcd43_fstring, band)
    kf.set_trajectory_model(n, n)
    q = np.ones(3*n*n, dtype=np.float32)*0.00001
    #q[:(n*n)] = 0.0001
    kf.set_trajectory_uncertainty(q, n, n)
    # The following runs the filter over time, selecting band 2 (NIR)
    # In order to calcualte BB albedos, you need to run the filter over
    # all bands, but you can do this in parallel

    kf.run(x_forecast, P_forecast, None, band=band, refine_diag=False)
    for key, val in output_plethora.iteritems():
        val = None
    LOG.info("FINISHED")
