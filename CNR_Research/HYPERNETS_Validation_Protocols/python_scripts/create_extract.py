#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:41:12 2019

@author: javier
"""
import os
import shutil
import sys
import zipfile
import subprocess

from netCDF4 import Dataset
import numpy as np

def convert_DMS_to_decimal(DD,MM,SS,cardinal):
    D = DD + (MM/60) + (SS/3600)
    if (cardinal == 'S') or (cardinal == 'W'):
        D = -D
    return D

def find_row_column_from_lat_lon(lat,lon,lat0,lon0):
    #% closest squared distance
    #% lat and lon are arrays of MxN
    #% lat0 and lon0 is the coordinates of one point
    dist_squared = (lat-lat0)**2 + (lon-lon0)**2
    r, c = np.unravel_index(np.argmin(dist_squared),lon.shape) # index to the closest in the latitude and longitude arrays
    return r, c

def extract_box(path_source,path_output,in_situ_lat,in_situ_lon):
    #%%
  
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
    
    filepah = os.path.join(path_source,coordinates_filename)
    nc_f0 = Dataset(filepah,'r')
    
    lat = nc_f0.variables['latitude'][:,:]
    lon = nc_f0.variables['longitude'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0412p50_filename)
    nc_f1 = Dataset(filepah,'r')
    Rrs_0412p50 = nc_f1.variables['Oa02_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0442p50_filename)
    nc_f2 = Dataset(filepah,'r')
    Rrs_0442p50 = nc_f2.variables['Oa03_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0490p00_filename)
    nc_f3 = Dataset(filepah,'r')
    Rrs_0490p00 = nc_f3.variables['Oa04_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0510p00_filename)
    nc_f4 = Dataset(filepah,'r')
    Rrs_0510p00 = nc_f4.variables['Oa05_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0560p00_filename)
    nc_f5 = Dataset(filepah,'r')
    Rrs_0560p00 = nc_f5.variables['Oa06_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0665p00_filename)
    nc_f6 = Dataset(filepah,'r')
    Rrs_0665p00 = nc_f6.variables['Oa08_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0673p75_filename)
    nc_f7 = Dataset(filepah,'r')
    Rrs_0673p75 = nc_f7.variables['Oa09_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_0865p00_filename)
    nc_f8 = Dataset(filepah,'r')
    Rrs_0865p00 = nc_f8.variables['Oa17_reflectance'][:,:]
    
    filepah = os.path.join(path_source,Rrs_1020p50_filename)
    nc_f9 = Dataset(filepah,'r')
    Rrs_1020p50 = nc_f9.variables['Oa21_reflectance'][:,:]
    
    r, c = find_row_column_from_lat_lon(lat,lon,lat_Venise,lon_Venise)
    
    size_box = 11
    
    #%% Save netCDF4 file
    path_out = os.path.join(path_output)
    ofname = 'extract.nc'
    ofname = os.path.join(path_out,ofname)
    
    if os.path.exists(ofname):
      os.remove(ofname)
    
    fmb = Dataset(ofname, 'w', format='NETCDF4')
    fmb.description = 'OLCI NxN extract'
    fmb.start_time = nc_f0.start_time
    fmb.stop_time = nc_f0.stop_time    
    fmb.input_file = path_source
    
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
#%%
host = 'mac'

if host == 'vm':
    path_source = '/DataArchive/OC/OLCI/sources_baseline_2.23/'
    
elif host == 'mac':
    path_main = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts'
    path_source = os.path.join(path_main,'data/source')
    listname = 'OLCI_list_test.txt' #'OLCI_list_uniq.txt '
    path_to_list = os.path.join(path_main,listname)

#Venise location: N 45o 18' 50", E 12o 30' 29"
lat_Venise = convert_DMS_to_decimal(45,18,50,'N')
lon_Venise = convert_DMS_to_decimal(12,30,29,'E')

with open(path_to_list,'r') as file:
    for cnt, line in enumerate(file):
        print('------------------')
        source_dir = os.path.join(path_source,line.split('/')[1],line.split('/')[2])
        source = os.path.join(source_dir,line.split('/')[3][:-1])
        if os.path.exists(source+'.SEN3.zip'):
            source_file_path = source+'.SEN3.zip'
            zipfilename = line.split('/')[3][:-1]+'.SEN3.zip'
            print('file '+source_file_path+' exists!')
            
        elif os.path.exists(source+'.zip'):
            source_file_path = source+'.zip'
            zipfilename = line.split('/')[3][:-1]+'.zip'
            print('file '+source_file_path+' exists!')
            
        else:
            print('file '+source+' does NOT exists!')
            continue
        
        output_dir = os.path.join(path_main,'data/output',line.split('/')[1],line.split('/')[2])
        # create the folders if not already exists
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        
        # adding exception handling
        try:
            shutil.copy(source_file_path, output_dir)
            zipfilePath = os.path.join(output_dir,zipfilename)
            zip = zipfile.ZipFile(zipfilePath)
            zip.extractall(output_dir)
            zip.close()
        except IOError as e:
            print("Unable to copy file. %s" % e)
        except:
            print("Unexpected error:", sys.exc_info())
        path_to_unzip = os.path.join(output_dir,zipfilename[:-4])   # without the .zip ending
        print(path_to_unzip)
        extract_box(path_source=path_to_unzip,path_output=output_dir,in_situ_lat=lat_Venise,in_situ_lon=lon_Venise)
        
        (ls_status, ls_output) = subprocess.getstatusoutput('rm -r '+path_to_unzip+'*')