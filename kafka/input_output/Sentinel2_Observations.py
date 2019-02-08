#!/usr/bin/env python
import _pickle as cPickle
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

from pathlib import Path

import logging
LOG = logging.getLogger(__name__+".Sentinel2_Observations")


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


S2MSIdata = namedtuple('S2MSIdata',
                     'observations uncertainty mask metadata emulator')

class Sentinel2Observations(object):
    def __init__(self, parent_folder, emulator_folder, state_mask, chunk=None):
        parent_folder = Path(parent_folder)
        emulator_folder = Path(emulator_folder)
        if not parent_folder.exists():
            LOG.info(f"S2 data folder: {parent_folder}")
            raise IOError("S2 data folder doesn't exist")

        if not emulator_folder.exists():
            LOG.info(f"Emulator folder: {emulator_folder}")
            raise IOError("Emulator folder doesn't exist")
        
        self.parent = parent_folder
        self.emulator_folder = emulator_folder
        self.original_mask = state_mask
        self.state_mask = state_mask
        self._find_granules(self.parent)
        self.band_map = ['02', '03', '04', '05', '06', '07',
                              '08', '12']
        emulators = glob.glob(os.path.join(self.emulator_folder, "*.pkl"))
        emulators.sort()
        self.emulator_files = emulators
        self.chunk = chunk
        
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


    def _find_granules(self, parent_folder):
        """Finds granules. Currently does so by checking for
        Feng's AOT file."""
        # this is needed to follow symlinks
        test_files = [x for f in parent_folder.iterdir() 
                      for x in f.rglob("**/*_aot.tif") ]
        try:
            self.dates = [ datetime.datetime(*(list(map(int, f.parts[-5:-2]))))
                    for f in test_files]
        except ValueError:
            self.dates = [datetime.datetime.strptime(f.parts[-1].split(
                "_")[1], "%Y%m%dT%H%M%S") for f in test_files]
        self.date_data = dict(zip(self.dates, [f.parent for f in test_files]))
        self.bands_per_observation = {}
        LOG.info(f"Found {len(test_files):d} S2 granules")
        LOG.info(f"First granule: {sorted(self.dates)[0].strftime('%Y-%m-%d'):s}")
        LOG.info(f"Last granule: {sorted(self.dates)[-1].strftime('%Y-%m-%d'):s}")
                              
        for the_date in self.dates:
            self.bands_per_observation[the_date] = 8 # 10m +redEdgebands


    def _find_emulator(self, sza, saa, vza, vaa):
        LOG.info("Emulator library code needs updating....")
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
        
        meta_file = current_folder.parent / "MTD_TL.xml"
        if not meta_file.exists():
            "Cant find metadat file!"
        sza, saa, vza, vaa = parse_xml(meta_file)
        metadata = dict (zip(["sza", "saa", "vza", "vaa"],
                            [sza, saa, vza, vaa]))
        # This should be really using EmulatorEngine...
        emulator_file = self._find_emulator(sza, saa, vza, vaa)
        emulator = cPickle.load(open(emulator_file, 'rb'),
                                 encoding='latin1')
        
        # emulator is a dictionary with keys 
        # [b'S2A_MSI_09', b'S2A_MSI_08', b'S2A_MSI_05', 
        # b'S2A_MSI_04', b'S2A_MSI_07', b'S2A_MSI_06', 
        # b'S2A_MSI_12', b'S2A_MSI_13', b'S2A_MSI_03', 
        # b'S2A_MSI_02']

        # Read and reproject S2 surface reflectance
        the_band = self.band_map[band]
        fname_prefix = [f.name.split("B02")[0]
                        for f in current_folder.glob("*B02_sur.tif")][0]
        original_s2_file = current_folder/f"{fname_prefix:s}B{the_band:s}_sur.tif"
        LOG.info(f"Original file {str(original_s2_file):s}")
        g = reproject_image(str(original_s2_file),
                            self.state_mask)
        
        rho_surface = g.ReadAsArray()
        cloud_mask = current_folder.parent/f"cloud.tif"
        g = reproject_image(str(cloud_mask),
                            self.state_mask)
        cloud_mask = g.ReadAsArray()
        mask = rho_surface > 0
        mask = mask*(cloud_mask <= 20)
        LOG.info(f"Total of {mask.sum():d} clear pixels " + 
                 f"({100.*mask.sum()/np.prod(mask.shape):f}%)")
        rho_surface = np.where(mask, rho_surface/10000., 0)
        # Read and reproject S2 angles
        
        #emulator_band_map = [2, 3, 4, 5, 6, 7, 8, 9, 12]
        #emulator_band_map = [2, 3, 4, 5, 6, 7, 8]
        
        band_dictionary = {'02':2, '03': 3, '04': 4, '05': 5, '06': 6, '07':7, '08': 8, '8A': 9, '09': 10, '11': 12, '12': 13}
        
        emulator_band_map = []
        for i in self.band_map:
            emulator_band_map.append(band_dictionary[i])
        
        R_mat = rho_surface*0.05# + self.band_unc[band]
        R_mat[np.logical_not(mask)] = 0.
        N = mask.ravel().shape[0]
        R_mat_sp = sp.lil_matrix((N, N))
        R_mat_sp.setdiag(1./(R_mat.ravel())**2)
        R_mat_sp = R_mat_sp.tocsr()

        s2_band = bytes("S2A_MSI_{:02d}".format(
                        emulator_band_map[band]), 'latin1' )
        print(">>>", original_s2_file, emulator_file, s2_band)
        s2data = S2MSIdata (rho_surface, R_mat_sp, mask, metadata, emulator[s2_band] )

        return s2data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    obs = Sentinel2Observations("/home/ucfafyi/public_html/S2_data/32/U/PU/",
           "/home/ucfafyi/DATA/Multiply/emus/sail/",
           "./ESU.tif")
    for timestep in obs.dates:
        retval = obs.get_band_data(timestep, 0)
