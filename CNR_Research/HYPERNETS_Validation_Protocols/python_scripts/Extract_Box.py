#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:45:45 2019
Description: Extract box around  and save it as netCDF file
@author: javier
"""
from netCDF4 import Dataset
import netCDF4 as nc
import os
import numpy as np
def convert_DMS_to_decimal(DD,MM,SS,cardinal):
    D = DD + (MM/60) + (SS/3600)
    if (cardinal == 'S') or (cardinal == 'W'):
        D = -D
    return D

def find_row_column_from_lat_lon(lat0,lon0):
    #% closest squared distance
    #% lat and lon are arrays of MxN
    #% lat0 and lon0 is the coordinates of one point
    dist_squared = (lat-lat0)**2 + (lon-lon0)**2
    r, c = np.unravel_index(np.argmin(dist_squared),lon.shape) # index to the closest in the latitude and longitude arrays
    return r, c
    
#%%
#year_str = '2016'
#doy_str =     
path_data = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/data/source/'
path_prod_folder = 'S3A_OL_2_WFR____20160517T085515_20160517T085715_20171031T182843_0119_004_164______MR1_R_NT_002.SEN3'
coordinates_filename = 'geo_coordinates.nc'

Rrs_0412p50_filename = 'Oa02_reflectance.nc'
Rrs_0442p50_filename = 'Oa03_reflectance.nc'
Rrs_0490p00_filename = 'Oa04_reflectance.nc'
Rrs_0510p00_filename = 'Oa05_reflectance.nc'
Rrs_0560p00_filename = 'Oa06_reflectance.nc'
Rrs_0665p00_filename = 'Oa08_reflectance.nc'
Rrs_0673p75_filename = 'Oa09_reflectance.nc'
Rrs_0865p00_filename = 'Oa17_reflectance.nc'
Rrs_1020p50_filename = 'Oa21_reflectance.nc'

filepah = os.path.join(path_data,path_prod_folder,coordinates_filename)
nc_f0 = Dataset(filepah,'r')

lat = nc_f0.variables['latitude'][:,:]
lon = nc_f0.variables['longitude'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0412p50_filename)
nc_f1 = Dataset(filepah,'r')
Rrs_0412p50 = nc_f1.variables['Oa02_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0442p50_filename)
nc_f2 = Dataset(filepah,'r')
Rrs_0442p50 = nc_f2.variables['Oa03_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0490p00_filename)
nc_f3 = Dataset(filepah,'r')
Rrs_0490p00 = nc_f3.variables['Oa04_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0510p00_filename)
nc_f4 = Dataset(filepah,'r')
Rrs_0510p00 = nc_f4.variables['Oa05_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0560p00_filename)
nc_f5 = Dataset(filepah,'r')
Rrs_0560p00 = nc_f5.variables['Oa06_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0665p00_filename)
nc_f6 = Dataset(filepah,'r')
Rrs_0665p00 = nc_f6.variables['Oa08_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0673p75_filename)
nc_f7 = Dataset(filepah,'r')
Rrs_0673p75 = nc_f7.variables['Oa09_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_0865p00_filename)
nc_f8 = Dataset(filepah,'r')
Rrs_0865p00 = nc_f8.variables['Oa17_reflectance'][:,:]

filepah = os.path.join(path_data,path_prod_folder,Rrs_1020p50_filename)
nc_f9 = Dataset(filepah,'r')
Rrs_1020p50 = nc_f9.variables['Oa21_reflectance'][:,:]

#%%
#Venise location: N 45o 18' 50", E 12o 30' 29"
lat_Venise = convert_DMS_to_decimal(45,18,50,'N')
lon_Venise = convert_DMS_to_decimal(12,30,29,'E')

r, c = find_row_column_from_lat_lon(lat_Venise,lon_Venise)

size_box = 11



#chl=chl[latind-1:latind+2,lonind-1:lonind+2] 

#%% Save netCDF4 file
path_out = os.path.join(path_data,path_prod_folder)
ofname = 'extract.nc'
ofname = os.path.join(path_out,ofname)

if os.path.exists(ofname):
  os.remove(ofname)

fmb = Dataset(ofname, 'w', format='NETCDF4')
fmb.description = 'OLCI NxN extract'
fmb.start_time = nc_f0.start_time
fmb.stop_time = nc_f0.stop_time    

fmb.createDimension('size_box_x', size_box)
fmb.createDimension('size_box_y', size_box)

latitude = fmb.createVariable('latitude',  'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6) 
latitude[:] = lat[r-int(size_box/2):r+(int(size_box/2)+1),\
                c-int(size_box/2):c+(int(size_box/2)+1)]

longitude = fmb.createVariable('longitude',  'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
longitude[:] = lon[r-int(size_box/2):r+(int(size_box/2)+1),\
                c-int(size_box/2):c+(int(size_box/2)+1)]

Rrs_0412p50_box=fmb.createVariable('Rrs_0412p50_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0412p50_box[:] = np.array(Rrs_0412p50[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_0442p50_box=fmb.createVariable('Rrs_0442p50_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0442p50_box[:] = np.array(Rrs_0442p50[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_0490p00_box=fmb.createVariable('Rrs_0490p00_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0490p00_box[:] = np.array(Rrs_0490p00[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_0510p00_box=fmb.createVariable('Rrs_0510p00_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0510p00_box[:] = np.array(Rrs_0510p00[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_0560p00_box=fmb.createVariable('Rrs_0560p00_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0560p00_box[:] = np.array(Rrs_0560p00[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_0665p00_box=fmb.createVariable('Rrs_0665p00_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0665p00_box[:] = np.array(Rrs_0665p00[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_0673p75_box=fmb.createVariable('Rrs_0673p75_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0673p75_box[:] = np.array(Rrs_0673p75[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_0865p00_box=fmb.createVariable('Rrs_0865p00_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_0865p00_box[:] = np.array(Rrs_0865p00[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

Rrs_1020p50_box=fmb.createVariable('Rrs_1020p50_box', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
Rrs_1020p50_box[:] = np.array(Rrs_1020p50[r-int(size_box/2):r+(int(size_box/2)+1),\
                    c-int(size_box/2):c+(int(size_box/2)+1)],dtype=np.float)

fmb.close()


