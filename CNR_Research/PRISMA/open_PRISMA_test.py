#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Feb 27 17:43:33 2020

@author: javier.concha
"""
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import h5py

#%% Open file.
path_to_image = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Images/PRISMA'
file_name = 'PRS_L1_STD_OFFL_20200208101029_20200208101033_0001.he5'
path_to_file = os.path.join(path_to_image,file_name)



f = h5py.File(path_to_file, 'r')
# reading name and value for root attributes (metadata contained in HDF5 root)
for attribute in f.attrs:
    print(attribute,f.attrs[attribute])
# reading names for all attributes (metadata) contained in HDF5 Groups
# specific method for reading the values shall be built depending by the specific metadata type (a single value, an array, a matrix, etc)
def printname(name):
    print(name)
f.visit(printname)

 # reading SWIR & VNIR datacubes
swir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube']
vnir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube']

# get geolocation info ----
lat_swir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_SWIR']
lon_swir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR']

lat_vnir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR']
lon_vnir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR']

# list the structure of SWIR data
swir.shape
# list the structure of VNIR data
vnir.shape
# print portions of the SWIR and VNIR bands
# band 0
swir[0:9,0,0:9]
vnir[0:9,0,0:9]
# band 170
swir[990:999,170,990:999]
# band 60
vnir[990:999,60,990:999]    

# from prismaread/R/convert_prisma.R
#  # Get wavelengths and fwhms ----
#   wl_vnir    <- hdf5r::h5attr(f, "List_Cw_Vnir")
#   order_vnir <- order(wl_vnir)
#   wl_vnir <- wl_vnir[order_vnir]