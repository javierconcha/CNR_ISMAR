#!/usr/bin/env python3
# coding: utf-8
"""
Created on Mon Jul  8 16:41:12 2019

@author: javier
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
import shutil
import sys
import zipfile
import subprocess

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma

host = 'mac'

# to import brdf_mario.py
if host == 'mac':
    sys.path.insert(0,'/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts')
elif host == 'vm':
    sys.path.insert(0,'/home/Javier.Concha/Val_Prot/codes')
import brdf_mario as brdf

os.environ['QT_QPA_PLATFORM']='offscreen' # to avoid error "QXcbConnection: Could not connect to display"

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

def extract_wind_and_angles(path_source,in_situ_lat,in_situ_lon):
    # from Tie-Points grid (a coarser grid)
    filepah = os.path.join(path_source,'tie_geo_coordinates.nc')
    nc_f0 = Dataset(filepah,'r')
    tie_lon = nc_f0.variables['longitude'][:]
    tie_lat = nc_f0.variables['latitude'][:]
    
    filepah = os.path.join(path_source,'tie_meteo.nc')
    nc_f0 = Dataset(filepah,'r')
    horizontal_wind = nc_f0.variables['horizontal_wind'][:]
    nc_f0.close()
    
    filepah = os.path.join(path_source,'tie_geometries.nc')
    nc_f1 = Dataset(filepah,'r')
    SZA = nc_f1.variables['SZA'][:]    
    SAA = nc_f1.variables['SAA'][:]  
    OZA = nc_f1.variables['OZA'][:]  
    OAA = nc_f1.variables['OAA'][:] 
    nc_f1.close()        
      
    r, c = find_row_column_from_lat_lon(tie_lat,tie_lon,in_situ_lat,in_situ_lon)
    
    ws0 = horizontal_wind[r,c,0]
    ws1 = horizontal_wind[r,c,1]  
    sza = SZA[r,c]
    saa = SAA[r,c]
    vza = OZA[r,c]
    vaa = OAA[r,c]

    return ws0, ws1, sza, saa, vza, vaa

def extract_box(path_source,path_output,in_situ_lat,in_situ_lon):
    #%
  
    coordinates_filename = 'geo_coordinates.nc'
    
    rhow_0412p50_filename = 'Oa02_reflectance.nc'
    rhow_0442p50_filename = 'Oa03_reflectance.nc'
    rhow_0490p00_filename = 'Oa04_reflectance.nc'
    rhow_0510p00_filename = 'Oa05_reflectance.nc'
    rhow_0560p00_filename = 'Oa06_reflectance.nc'
    rhow_0620p00_filename = 'Oa07_reflectance.nc'
    rhow_0665p00_filename = 'Oa08_reflectance.nc'
    rhow_0673p75_filename = 'Oa09_reflectance.nc'
    rhow_0865p00_filename = 'Oa17_reflectance.nc'
    rhow_1020p50_filename = 'Oa21_reflectance.nc'
    
    AOT_0865p50_filename = 'w_aer.nc'
    WQSF_filename = 'wqsf.nc'
    
    filepah = os.path.join(path_source,coordinates_filename)
    nc_f0 = Dataset(filepah,'r')
    
    lat = nc_f0.variables['latitude'][:,:]
    lon = nc_f0.variables['longitude'][:,:]

    r, c = find_row_column_from_lat_lon(lat,lon,in_situ_lat,in_situ_lon)
    
    size_box = 11
    
    start_idx_x = (r-int(size_box/2))
    stop_idx_x = (r+int(size_box/2)+1)
    start_idx_y = (c-int(size_box/2))
    stop_idx_y = (c+int(size_box/2)+1)

    
    if r>=0 and r+1<lat.shape[0] and c>=0 and c+1<lat.shape[1]:
        
        filepah = os.path.join(path_source,rhow_0412p50_filename)
        nc_f1 = Dataset(filepah,'r')
        rhow_0412p50 = nc_f1.variables['Oa02_reflectance'][:]
        nc_f1.close()
        
        filepah = os.path.join(path_source,rhow_0442p50_filename)
        nc_f2 = Dataset(filepah,'r')
        rhow_0442p50 = nc_f2.variables['Oa03_reflectance'][:]
        nc_f2.close()
        
        filepah = os.path.join(path_source,rhow_0490p00_filename)
        nc_f3 = Dataset(filepah,'r')
        rhow_0490p00 = nc_f3.variables['Oa04_reflectance'][:]
        nc_f3.close()
        
        filepah = os.path.join(path_source,rhow_0510p00_filename)
        nc_f4 = Dataset(filepah,'r')
        rhow_0510p00 = nc_f4.variables['Oa05_reflectance'][:]
        nc_f4.close()
        
        filepah = os.path.join(path_source,rhow_0560p00_filename)
        nc_f5 = Dataset(filepah,'r')
        rhow_0560p00 = nc_f5.variables['Oa06_reflectance'][:]
        nc_f5.close()

        filepah = os.path.join(path_source,rhow_0620p00_filename)
        nc_f6 = Dataset(filepah,'r')
        rhow_0620p00 = nc_f6.variables['Oa07_reflectance'][:]
        nc_f6.close()
        
        filepah = os.path.join(path_source,rhow_0665p00_filename)
        nc_f7 = Dataset(filepah,'r')
        rhow_0665p00 = nc_f7.variables['Oa08_reflectance'][:]
        nc_f7.close()
        
        filepah = os.path.join(path_source,rhow_0673p75_filename)
        nc_f8 = Dataset(filepah,'r')
        rhow_0673p75 = nc_f8.variables['Oa09_reflectance'][:]
        nc_f8.close()
        
        filepah = os.path.join(path_source,rhow_0865p00_filename)
        nc_f9 = Dataset(filepah,'r')
        rhow_0865p00 = nc_f9.variables['Oa17_reflectance'][:]
        nc_f9.close()
        
        filepah = os.path.join(path_source,rhow_1020p50_filename)
        nc_f10 = Dataset(filepah,'r')
        rhow_1020p50 = nc_f10.variables['Oa21_reflectance'][:]
        nc_f10.close()
        
        filepah = os.path.join(path_source,AOT_0865p50_filename)
        nc_f11 = Dataset(filepah,'r')
        AOT_0865p50 = nc_f11.variables['T865'][:]
        nc_f11.close()

        filepah = os.path.join(path_source,WQSF_filename)
        nc_f12 = Dataset(filepah,'r')
        WQSF = nc_f12.variables['WQSF'][:]
        nc_f12.close()
        #%% Calculate BRDF
        ws0, ws1, sza, saa, vza, vaa = extract_wind_and_angles(path_source,in_situ_lat,in_situ_lon)
        
        filepah = os.path.join(path_source,'chl_oc4me.nc')
        nc_f11 = Dataset(filepah,'r')
        CHL_OC4ME = nc_f11.variables['CHL_OC4ME'][:]
        nc_f11.close()
        CHL_OC4ME_extract = ma.array(CHL_OC4ME[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        BRDF0 = np.full(CHL_OC4ME_extract.shape,np.nan)
        BRDF1 = np.full(CHL_OC4ME_extract.shape,np.nan)
        BRDF2 = np.full(CHL_OC4ME_extract.shape,np.nan)
        BRDF3 = np.full(CHL_OC4ME_extract.shape,np.nan)
        BRDF4 = np.full(CHL_OC4ME_extract.shape,np.nan)
        BRDF5 = np.full(CHL_OC4ME_extract.shape,np.nan)
        BRDF6 = np.full(CHL_OC4ME_extract.shape,np.nan)
        for ind0 in range(CHL_OC4ME_extract.shape[0]):
            for ind1 in range(CHL_OC4ME_extract.shape[1]):
                chl = CHL_OC4ME_extract[ind0,ind1]
                # 412.5, 442.5, 490, 510, 560, 620, 660 bands
                # 0      1      2    3    4    5    6   brdf index
                # 412.5  442.5  490  510  560  620  665 OLCI bands
                # 02     03     04   05   06   07   08  OLCI band names in L2
                brdf_coeffs = brdf.brdf(ws0, ws1, chl, sza, saa, vza, vaa)
                BRDF0[ind0,ind1] = brdf_coeffs[0,0]
                BRDF1[ind0,ind1] = brdf_coeffs[0,1]
                BRDF2[ind0,ind1] = brdf_coeffs[0,2]
                BRDF3[ind0,ind1] = brdf_coeffs[0,3]
                BRDF4[ind0,ind1] = brdf_coeffs[0,4]
                BRDF5[ind0,ind1] = brdf_coeffs[0,5]
                BRDF6[ind0,ind1] = brdf_coeffs[0,6]
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
        
        fmb.createDimension('index',None)
        row_center = fmb.createVariable('row_center',  'f4', ('index'), fill_value=-999, zlib=True, complevel=6) 
        row_center[:] = r
        row_center.description = 'row index to the original L2 file'
        col_center = fmb.createVariable('col_center',  'f4', ('index'), fill_value=-999, zlib=True, complevel=6) 
        col_center[:] = c
        col_center.description = 'column index to the original L2 file'
        
        lat_insitu = fmb.createVariable('lat_insitu',  'f4', ('index'), fill_value=-999, zlib=True, complevel=6) 
        lat_insitu[:] = in_situ_lat
        lat_insitu.description = 'latitude of in situ measurement'
        
        lon_insitu = fmb.createVariable('lon_insitu',  'f4', ('index'), fill_value=-999, zlib=True, complevel=6) 
        lon_insitu[:] = in_situ_lon
        lon_insitu.description = 'longitude of in situ measurement'
        
        fmb.createDimension('angles_and_wind',None)
        ws0_value = fmb.createVariable('ws0_value',  'f4', ('angles_and_wind'), fill_value=-999, zlib=True, complevel=6) 
        ws0_value[:] = ws0
        ws1_value = fmb.createVariable('ws1_value',  'f4', ('angles_and_wind'), fill_value=-999, zlib=True, complevel=6) 
        ws1_value[:] = ws1
        sza_value = fmb.createVariable('sza_value',  'f4', ('angles_and_wind'), fill_value=-999, zlib=True, complevel=6) 
        sza_value[:] = sza
        saa_value = fmb.createVariable('saa_value',  'f4', ('angles_and_wind'), fill_value=-999, zlib=True, complevel=6) 
        saa_value[:] = saa
        vza_value = fmb.createVariable('vza_value',  'f4', ('angles_and_wind'), fill_value=-999, zlib=True, complevel=6) 
        vza_value[:] = vza
        vaa_value = fmb.createVariable('vaa_value',  'f4', ('angles_and_wind'), fill_value=-999, zlib=True, complevel=6) 
        vaa_value[:] = vaa
        
        latitude = fmb.createVariable('latitude',  'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6) 
        latitude[:] = lat[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
        
        longitude = fmb.createVariable('longitude',  'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        longitude[:] = lon[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]

        # NOT BRDF-corrected
        rhow_0412p50_box=fmb.createVariable('rhow_0412p50', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0412p50_box[:] = ma.array(rhow_0412p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0412p50_box.description = 'rhow(0412.50) NOT brdf-corrected'
        
        rhow_0442p50_box=fmb.createVariable('rhow_0442p50', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0442p50_box[:] = ma.array(rhow_0442p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0442p50_box.description = 'rhow(0442.50) NOT brdf-corrected'
        
        rhow_0490p00_box=fmb.createVariable('rhow_0490p00', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0490p00_box[:] = ma.array(rhow_0490p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0490p00_box.description = 'rhow(0490.00) NOT brdf-corrected'
        
        rhow_0510p00_box=fmb.createVariable('rhow_0510p00', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0510p00_box[:] = ma.array(rhow_0510p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0510p00_box.description = 'rhow(0510.00) NOT brdf-corrected'
        
        rhow_0560p00_box=fmb.createVariable('rhow_0560p00', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0560p00_box[:] = ma.array(rhow_0560p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0560p00_box.description = 'rhow(0560.00) NOT brdf-corrected'

        rhow_0620p00_box=fmb.createVariable('rhow_0620p00', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0620p00_box[:] = ma.array(rhow_0620p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0620p00_box.description = 'rhow(0620.00) NOT brdf-corrected'
        
        rhow_0665p00_box=fmb.createVariable('rhow_0665p00', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0665p00_box[:] = ma.array(rhow_0665p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0665p00_box.description = 'rhow(0665.00) NOT brdf-corrected'

        rhow_0673p75_box=fmb.createVariable('rhow_0673p75', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0673p75_box[:] = ma.array(rhow_0673p75[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0673p75_box.description = 'rhow(0673.75) NOT brdf-corrected'
        
        rhow_0865p00_box=fmb.createVariable('rhow_0865p00', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0865p00_box[:] = ma.array(rhow_0865p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_0865p00_box.description = 'rhow(0865.00) NOT brdf-corrected'
        
        rhow_1020p50_box=fmb.createVariable('rhow_1020p50', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_1020p50_box[:] = ma.array(rhow_1020p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        rhow_1020p50_box.description = 'rhow(1020.50) NOT brdf-corrected'
        
        # BRDF-corrected
        rhow_0412p50_fq=fmb.createVariable('rhow_0412p50_fq', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0412p50_fq[:] = ma.array(rhow_0412p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]*BRDF0)
        rhow_0412p50_fq.description = 'rhow(0412.50) brdf-corrected'
        
        rhow_0442p50_fq=fmb.createVariable('rhow_0442p50_fq', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0442p50_fq[:] = ma.array(rhow_0442p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]*BRDF1)
        rhow_0442p50_fq.description = 'rhow(0442.50) brdf-corrected'
        
        rhow_0490p00_fq=fmb.createVariable('rhow_0490p00_fq', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0490p00_fq[:] = ma.array(rhow_0490p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]*BRDF2)
        rhow_0490p00_fq.description = 'rhow(0490.00) brdf-corrected'
        
        rhow_0510p00_fq=fmb.createVariable('rhow_0510p00_fq', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0510p00_fq[:] = ma.array(rhow_0510p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]*BRDF3)
        rhow_0510p00_fq.description = 'rhow(0510.00) brdf-corrected'
        
        rhow_0560p00_fq=fmb.createVariable('rhow_0560p00_fq', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0560p00_fq[:] = ma.array(rhow_0560p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]*BRDF4)
        rhow_0560p00_fq.description = 'rhow(0560.00) brdf-corrected'

        rhow_0620p00_fq=fmb.createVariable('rhow_0620p00_fq', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0620p00_fq[:] = ma.array(rhow_0620p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]*BRDF5)
        rhow_0620p00_fq.description = 'rhow(0620.00) brdf-corrected'
        
        rhow_0665p00_fq=fmb.createVariable('rhow_0665p00_fq', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        rhow_0665p00_fq[:] = ma.array(rhow_0665p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]*BRDF6)
        rhow_0665p00_fq.description = 'rhow(0665.00) brdf-corrected'
        
        AOT_0865p50_box=fmb.createVariable('AOT_0865p50', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        AOT_0865p50_box[:] = ma.array(AOT_0865p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        AOT_0865p50_box.description = 'Aerosol optical thickness'

        WQSF_box=fmb.createVariable('WQSF', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        WQSF_box[:] = ma.array(WQSF[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
        WQSF_box.description = 'OLCI Level 2 WATER Product, Classification, Quality and Science Flags Data Set'
        
        fq_0 = fmb.createVariable('fq_0', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        fq_0[:] = ma.array(BRDF0)

        fq_1 = fmb.createVariable('fq_1', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        fq_1[:] = ma.array(BRDF1)

        fq_2 = fmb.createVariable('fq_2', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        fq_2[:] = ma.array(BRDF2)

        fq_3 = fmb.createVariable('fq_3', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        fq_3[:] = ma.array(BRDF3)

        fq_4 = fmb.createVariable('fq_4', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        fq_4[:] = ma.array(BRDF4)

        fq_5 = fmb.createVariable('fq_5', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        fq_5[:] = ma.array(BRDF5)

        fq_6 = fmb.createVariable('fq_6', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        fq_6[:] = ma.array(BRDF6)

        chl_oc4me = fmb.createVariable('chl_oc4me', 'f4', ('size_box_x','size_box_y'), fill_value=-999, zlib=True, complevel=6)
        chl_oc4me[:] = ma.array(CHL_OC4ME_extract)
        
        fmb.close()
        print('Extract created!')
        created_flag = 1
        return created_flag
    else:
        print('Index out of bound!')
        created_flag = 0
        return created_flag
#%%
def main():
    """business logic for when running this module as the primary one!"""
    print('Main Code!')
    
    if host == 'vm': 
        path_main = '/home/Javier.Concha/Val_Prot/'
        path_source = '/DataArchive/OC/OLCI/sources_baseline_2.23/'
        listname = 'OLCI_list_uniq.txt' #'OLCI_list_uniq.txt '
        path_to_list = os.path.join(path_main,'codes',listname)        
    elif host == 'mac':
        path_main = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts'
        path_source = os.path.join(path_main,'data/source')
        listname = 'OLCI_list_test.txt' #'OLCI_list_uniq.txt '
        path_to_list = os.path.join(path_main,listname)
    else:
        print('Error: host flag is not either mac or vm')
    
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
            
            if not os.path.exists(path_to_unzip):
                if os.path.exists(path_to_unzip+'.SEN3'):
                    path_to_unzip = path_to_unzip+'.SEN3'
                else:
                    print('dir '+path_to_unzip+' does NOT exist!')
                    continue    
            
#            print(path_to_unzip)
            created_flag = extract_box(path_source=path_to_unzip,path_output=output_dir,in_situ_lat=lat_Venise,in_situ_lon=lon_Venise)
            if not created_flag:
                cmd = 'rm '+path_to_unzip+'/*' # remove .nc
                print(cmd)
                (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
                cmd = 'rm '+output_dir+'/*.zip' # remove .zip
                print(cmd)
                (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
                cmd = 'mv '+output_dir+' '+os.path.join(path_main,'data/output/not_processed') # move directory to be deleted after
                print(cmd)
                (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
                print(ls_status)
                print(ls_output)
            else:
                cmd = 'rm '+path_to_unzip+'/*' # remove .nc
                print(cmd)
                (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
                cmd = 'mv '+path_to_unzip+' '+os.path.join(path_main,'data/output/not_processed')# move unzipped directory to be deleted after
                print(cmd)
                (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
                cmd = 'rm '+output_dir+'/*.zip' # remove .zip
                print(cmd)
                (ls_status, ls_output) = subprocess.getstatusoutput(cmd)

        
#%%
if __name__ == '__main__':
    main()        