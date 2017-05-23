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

__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

from collections import namedtuple
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"

import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

#from geoh5 import kea
import gdal

from linear_kernels_kf import KernelLinearKalman
from utils import OutputFile # The netCDF writer

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
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
        x_iso =  x_analysis[:(2400*2400)].reshape((2400,2400))
        x_vol =  x_analysis[(2400*2400):2*(2400*2400)].reshape((2400,2400))
        x_geo =  x_analysis[2*(2400*2400):3*(2400*2400)].reshape((2400,2400))

        x_iso_unc =  P_analysis.diagonal()[:(2400*2400)].reshape((2400,2400))
        x_vol_unc =  P_analysis.diagonal()[(2400*2400):2*(2400*2400)].reshape((2400,2400))
        x_geo_unc =  P_analysis.diagonal()[2*(2400*2400):3*(2400*2400)].reshape((2400,2400))

        BHR = x_iso+x_vol*0.189184 + x_geo*1.377622
        BHR_unc = x_iso_unc + x_vol_unc*0.189184 + x_geo_unc*1.377622

        LOG.info("saving. Timestep %d, step %d" % (timestep, step))
        self.output['bhr'].GetRasterBand(timestep+1).WriteArray(BHR)
        self.output['bhr'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})
        
        self.output['iso'].GetRasterBand(timestep+1).WriteArray(x_iso)
        self.output['iso'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})
        
        self.output['vol'].GetRasterBand(timestep+1).WriteArray(x_vol)
        self.output['vol'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})
        
        self.output['geo'].GetRasterBand(timestep+1).WriteArray(x_geo)
        self.output['geo'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})

        self.output['bhr_unc'].GetRasterBand(timestep+1).WriteArray(BHR_unc)
        self.output['bhr_unc'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})
        
        self.output['iso_unc'].GetRasterBand(timestep+1).WriteArray(x_iso_unc)
        self.output['iso_unc'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})
        
        self.output['vol_unc'].GetRasterBand(timestep+1).WriteArray(x_vol_unc)
        self.output['vol_unc'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})
        
        self.output['geo_unc'].GetRasterBand(timestep+1).WriteArray(x_geo_unc)
        self.output['geo_unc'].GetRasterBand(timestep+1).SetMetadata({'DoY':"%d"%(timestep)})

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
        rho = rho_pntr.GetRasterBand(timestep+1).ReadAsArray()/10000.
        # Taken from http://modis-sr.ltdri.org/pages/validation.html
        modis_uncertainty=np.array([0.005, 0.014, 0.008, 0.005, 0.012,
                                    0.006, 0.003])[self.the_band]
        R_mat = self.create_uncertainty(modis_uncertainty, mask)
        metadata = Metadata(mask, modis_uncertainty, 
                            vza/100., sza/100., raa/100.)
        return rho, R_mat, mask, metadata




if __name__ == "__main__":
    the_dir="/data/selene/ucfajlg/Aurade_MODIS/"
    band = 1
    g = gdal.Open(the_dir + "brdf_2010_b01.vrt")
    days = np.array([int(g.GetRasterBand(i+1).GetMetadata()['DoY'])
                            for i in xrange(g.RasterCount)])
    modis_obs = MODIS_observations( days,
                gdal.Open(os.path.join(the_dir, "statekm_2010.vrt")),
                gdal.Open(os.path.join(the_dir, "SensorZenith_2010.vrt")),
                gdal.Open(os.path.join(the_dir, "SolarZenith_2010.vrt")),
                gdal.Open(os.path.join(the_dir, "SolarAzimuth_2010.vrt")),
                gdal.Open(os.path.join(the_dir, "SensorAzimuth_2010.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2010_b01.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2010_b02.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2010_b03.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2010_b04.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2010_b05.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2010_b06.vrt")),
                gdal.Open(os.path.join(the_dir, "brdf_2010_b07.vrt")))

    # We can now create the output
    # stuff stuff stuff    
    def create_output_file(template, filename):
        drv = gdal.GetDriverByName("GTiff")
        dst_ds = drv.Create (filename, 2400, 2400, 366, gdal.GDT_Float32,
                              ['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'PREDICTOR=1',
                               'TILED=YES'])
        dst_ds.SetProjection(template.GetProjection())
        dst_ds.SetGeoTransform(template.GetGeoTransform())
        return dst_ds

    fname_add = '_testing'# ''
    dst_ds={}
    dst_ds['bhr'] = create_output_file(modis_obs[1], "tmp/bhr_band{}{}.tif".format(band, fname_add))
    dst_ds['iso'] = create_output_file(modis_obs[1], "tmp/iso_band{}{}.tif".format(band, fname_add))
    dst_ds['geo'] = create_output_file(modis_obs[1], "tmp/geo_band{}{}.tif".format(band, fname_add))
    dst_ds['vol'] = create_output_file(modis_obs[1], "tmp/vol_band{}{}.tif".format(band, fname_add))
    dst_ds['bhr_unc'] = create_output_file(modis_obs[1], "tmp/bhr_unc_band{}{}.tif".format(band, fname_add))
    dst_ds['iso_unc'] = create_output_file(modis_obs[1], "tmp/iso_unc_band{}{}.tif".format(band, fname_add))
    dst_ds['geo_unc'] = create_output_file(modis_obs[1], "tmp/geo_unc_band{}{}.tif".format(band, fname_add))
    dst_ds['vol_unc'] = create_output_file(modis_obs[1], "tmp/vol_unc_band{}{}.tif".format(band, fname_add))
 

    #output = OutputFile("/tmp/testme.nc", times=None, x=np.arange(2400),
    #                    y=np.arange(2400))
    kf = MODISKernelLinearKalman(modis_obs, days, dst_ds, [] )
    n = 2400
    x_forecast = np.ones(3*2400*2400)*0.5
    P_forecast = sp.eye(3*n*n, 3*n*n, format="csc", dtype=np.float32)
    kf.set_trajectory_model(2400, 2400)
    kf.set_trajectory_uncertainty(0.005, 2400, 2400)
    # The following runs the filter over time, selecting band 2 (NIR)
    # In order to calcualte BB albedos, you need to run the filter over
    # all bands, but you can do this in parallel
    kf.run(x_forecast, P_forecast, None, band=band, refine_diag=False)
    for key, val in dst_ds.iteritems():
        val = None
    LOG.info("FINISHED")
