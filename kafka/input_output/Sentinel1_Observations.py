#!/usr/bin/env python
"""

"""
import datetime
import glob
import os

from functools import partial

from collections import namedtuple

import gdal

import numpy as np

import osr

import scipy.sparse as sp

from ..observation_operators import sar_observation_operator

WRONG_VALUE = -999.0  # TODO tentative missing value

SARdata = namedtuple('SARdata',
                     'observations uncertainty mask metadata emulator')

class Gp_Wrapper(object):
    def __init__ (self, function):
        self._function = function
    
    def predict(self, x, do_deriv=True, do_unc=False):
        return self._function(self._function(x))
        
        
def reproject_image(source_img, target_img, dstSRSs=None):
    """Reprojects/Warps an image to fit exactly another image.
    Additionally, you can set the destination SRS if you want
    to or if it isn't defined in the source image."""
    try:
        g = gdal.Open(target_img)
    except RuntimeError:
        if type(target_img) == gdal.Dataset:
            g = target_img
    
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
    if dstSRSs is None:
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
    else:
        dstSRS = dstSRSs
    g = gdal.Warp('', source_img, format='MEM',
                  outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                  dstSRS=dstSRS)
    if g is None:
        raise ValueError("Something failed with GDAL!")
    return g


class S1Observations(object):
    """
    """

    def __init__(self, data_folder, state_mask,
                 emulators={'VV': partial(sar_observation_operator, polarisation="VV"),
                            'VH': partial(sar_observation_operator, polarisation="VH")},
                 chunk=None):

        """

        """
        # 1. Find the files
        files = glob.glob(os.path.join(data_folder, '*.nc'))
        files.sort()
        self.original_mask = state_mask
        self.state_mask = state_mask
        self.dates = []
        self.date_data = {}
        for fich in files:
            fname = os.path.basename(fich)
            # TODO Maybe filter files by metadata
            # (e.g. select ascending/descending passes)
            splitter = fname.split('_')
            this_date = datetime.datetime.strptime(splitter[5],
                                                   '%Y%m%dT%H%M%S')
            this_date = this_date.replace(
                minute=0, hour=0, second=0, microsecond=0)
            self.dates.append(this_date)
            self.date_data[this_date] = fich
        # 2. Store the emulator(s)
        self.emulators = emulators
        self.bands_per_observation = {}
        self.chunk = chunk
        for the_date in self.dates:
            self.bands_per_observation[the_date] = 2 # 2 bands

       
        
    def apply_roi(self, ulx, uly, lrx, lry):
        self.ulx = ulx
        self.uly = uly
        self.lrx = lrx
        self.lry = lry
        width = lrx - ulx
        height = uly - lry
        
        self.state_mask = gdal.Translate("", self.original_mask,
                                         srcWin=[ulx, uly, width, abs(height)],
                                         format="MEM")
        
    def define_output(self):
        try:
            g = gdal.Open(self.state_mask)
            proj = g.GetProjection()
            geoT = np.array(g.GetGeoTransform())

        except:
            proj = self.state_mask.GetProjection()
            geoT = np.array(self.state_mask.GetGeoTransform())

        #new_geoT = geoT*1.
        #new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        #new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geoT.tolist() #new_geoT.tolist()
    
    def _read_backscatter(self, obs_ptr):
        """
        Read backscatter from NetCDF4 File
        Should return a 2D array that should overlap with the "state grid"

        Input
        ------
        fname (string, filename of the NetCDF4 file, path included)
        polarisation (string, used polarisation)

        Output
        ------
        backscatter values stored within NetCDF4 file for given polarisation

        """
        backscatter = obs_ptr.ReadAsArray()
        return backscatter

    def _calculate_uncertainty(self, backscatter):
        """
        Calculation of the uncertainty of Sentinel-1 input data

        Radiometric uncertainty of Sentinel-1 Sensors are within 1 and 0.5 dB

        Calculate Equivalent Number of Looks (ENL) of input dataset leads to
        uncertainty of scene caused by speckle-filtering/multi-looking

        Input
        ------
        backscatter (backscatter values)

        Output
        ------
        unc (uncertainty in dB)

        """

        # first approximation of uncertainty (1 dB)
        unc = backscatter*0.1


        # need to find a good way to calculate ENL
        # self.ENL = (backscatter.mean()**2) / (backscatter.std()**2)

        return unc

    def _get_mask(self, backscatter):
        """
        Mask for selection of pixels

        Get a True/False array with the selected/unselected pixels


        Input
        ------
        this_file ?????

        Output
        ------

        """

        mask = np.ones_like(backscatter, dtype=np.bool)
        mask[backscatter == WRONG_VALUE] = False
        return mask

    def get_band_data(self, timestep, band):
        """
        get all relevant S1 data information for one timestep to get processing
        done


        Input
        ------
        timestep
        band

        Output
        ------
        sardata (namedtuple with information on observations, uncertainty,
                mask, metadata, emulator/used model)

        """
        if band == 0:
            polarisation = 'VV'
        elif band == 1:
            polarisation = 'VH'
        this_file = self.date_data[timestep]
        fname = 'NETCDF:"{:s}":sigma0_{:s}_multi'.format(this_file,
                                                         polarisation.lower())
        obs_ptr = reproject_image(fname, self.state_mask)
        observations = self._read_backscatter(obs_ptr)
        uncertainty = self._calculate_uncertainty(observations)
        mask = self._get_mask(observations)
        R_mat = np.zeros_like(observations)
        R_mat = uncertainty
        R_mat[np.logical_not(mask)] = 0.
        N = mask.ravel().shape[0]
        R_mat_sp = sp.lil_matrix((N, N))
        R_mat_sp.setdiag(1./(R_mat.ravel())**2)
        R_mat_sp = R_mat_sp.tocsr()

        
        # TODO read in angle of incidence from netcdf file
        # metadata['incidence_angle_deg'] =
        fname = 'NETCDF:"{:s}":{:s}'.format(this_file, "theta")
        obs_ptr = reproject_image(fname, self.state_mask)
        metadata = {'incidence_angle': obs_ptr.ReadAsArray()}
        #emulator = partial(self.emulators[polarisation],
        #                   theta=metadata["incidence_angle"],
        #                   polarisation=polarisation)
        #emu_wrapper = Gp_Wrapper(emulator)
        sardata = SARdata(observations, R_mat_sp, mask, metadata,
                          sar_observation_operator)
        return sardata


if __name__ == "__main__":
    data_folder = "/data/selene/ucfajlg/ELBARA_LMU/LMU_SAR_testcase"
    state_mask = "/home/ucfajlg/Data/python/LMU_testcase/data/ESU.tif"
    s1_observations = S1Observations(data_folder, state_mask)
    for timestep in s1_observations.dates:
        retval = s1_observations.get_band_data(timestep, 0)
