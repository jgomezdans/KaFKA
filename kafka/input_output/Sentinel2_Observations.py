#!/usr/bin/env python
import cPickle
import datetime
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp # Required for unc
import gdal
import osr

import xml.etree.ElementTree as ET
from collections import namedtuple

def parse_xml(filename):
    """Parses the XML metadata file to extract view/incidence 
    angles. The file has grids and all sorts of stuff, but
    here we just average everything, and you get 
    1. SZA
    2. SAA 
    3. VZA
    4. VAA.
    """
    with open(filename, 'r') as f:
        tree = ET.parse(filename)
        root = tree.getroot()

        vza = []
        vaa = []
        for child in root:
            for x in child.findall("Tile_Angles"):
                for y in x.find("Mean_Sun_Angle"):
                    if y.tag == "ZENITH_ANGLE":
                        sza = float(y.text)
                    elif y.tag == "AZIMUTH_ANGLE":
                        saa = float(y.text)
                for s in x.find("Mean_Viewing_Incidence_Angle_List"):
                    for r in s:
                        if r.tag == "ZENITH_ANGLE":
                            vza.append(float(r.text))
                            
                        elif r.tag == "AZIMUTH_ANGLE":
                            vaa.append(float(r.text))
                            
    return sza, saa, np.mean(vza), np.mean(vaa)


def reproject_image(source_img, target_img, dstSRSs=None):
    """Reprojects/Warps an image to fit exactly another image.
    Additionally, you can set the destination SRS if you want
    to or if it isn't defined in the source image."""
    g = gdal.Open(target_img)
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


S2MSIdata = namedtuple('S2MSIdata',
                     'observations uncertainty mask metadata emulator')

class Sentinel2Observations(object):
    def __init__(self, parent_folder, emulator_folder, state_mask):
        if not os.path.exists(parent_folder):
            raise IOError("S2 data folder doesn't exist")
        self.parent = parent_folder
        self.emulator_folder = emulator_folder
        self.state_mask = state_mask
        self._find_granules(self.parent)
        self.band_map = ['02', '03', '04', '05', '06', '07',
                         '08', '8A', '09', '12']
        emulators = glob.glob(os.path.join(self.emulator_folder, "*.pkl"))
        emulators.sort()
        self.emulator_files = emulators

    def define_output(self):
        g = gdal.Open(self.state_mask)
        proj = g.GetProjection()
        geoT = np.array(g.GetGeoTransform())
        #new_geoT = geoT*1.
        #new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        #new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geoT.tolist() #new_geoT.tolist()


    def _find_granules(self, parent_folder):
        """Finds granules. Currently does so by checking for
        Feng's AOT file."""
        self.dates = []
        self.date_data = {}
        for root, dirs, files in os.walk(parent_folder):
            for fich in files:
                if fich.find("aot.tif") >= 0:
                    this_date = datetime.datetime(*[int(i) 
                                for i in root.split("/")[-4:-1]])
                    self.dates.append(this_date)
                    self.date_data[this_date] = root
        self.bands_per_observation = {}
        for the_date in self.dates:
            self.bands_per_observation[the_date] = 10 # 10 bands


    def _find_emulator(self, sza, saa, vza, vaa):
        raa = vaa - saa
        vzas = np.array([float(s.split("_")[-3]) 
                         for s in self.emulator_files])
        szas = np.array([float(s.split("_")[-2]) 
                         for s in self.emulator_files])
        raas = np.array([float(s.split("_")[-1].split(".")[0]) 
                         for s in self.emulator_files])        
        e1 = szas == szas[np.argmin(np.abs(szas - sza))]
        e2 = vzas == vzas[np.argmin(np.abs(vzas - vza))]
        e3 = raas == raas[np.argmin(np.abs(raas - raa))]
        iloc = np.where(e1*e2*e3)[0][0]
        return self.emulator_files[iloc]


    def get_band_data(self, timestep, band):
        
        current_folder = self.date_data[timestep]

        meta_file = os.path.join(current_folder, "metadata.xml")
        sza, saa, vza, vaa = parse_xml(meta_file)
        metadata = dict (zip(["sza", "saa", "vza", "vaa"],
                            [sza, saa, vza, vaa]))
        # This should be really using EmulatorEngine...
        emulator_file = self._find_emulator(sza, saa, vza, vaa)
        emulator = cPickle.load(open (emulator_file, 'rb'))
        
        # Read and reproject S2 surface reflectance
        the_band = self.band_map[band]
        original_s2_file = os.path.join ( current_folder, 
                                         "B{}_sur.tif".format(the_band))
        print(original_s2_file)
        g = reproject_image( original_s2_file, self.state_mask)
        rho_surface = g.ReadAsArray()
        mask = rho_surface > 0
        rho_surface = np.where(mask, rho_surface/10000., 0)
        # Read and reproject S2 angles
        emulator_band_map = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
        
        
        R_mat = rho_surface*0.05
        R_mat[np.logical_not(mask)] = 0.
        N = mask.ravel().shape[0]
        R_mat_sp = sp.lil_matrix((N, N))
        R_mat_sp.setdiag(1./(R_mat.ravel())**2)
        R_mat_sp = R_mat_sp.tocsr()
        
        s2data = S2MSIdata (rho_surface, R_mat_sp, mask, metadata, 
                            emulator["S2A_MSI_{:02d}".format(emulator_band_map[band])])
        return s2data
