#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:54:42 2019

@author: javier
"""
import os
from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import subprocess
#%% Open in situ in netcdf format from excel_to_nc_AquaAlta_merge_newsite.py by Marco B.
"""
<class 'netCDF4._netCDF4.Dataset'>
root group (NETCDF4 data model, file format HDF5):
    dimensions(sizes): Time(1044), Central_wavelenghts(8)
    variables(dimensions): 
        <class 'str'> Time(Time), 
        float32 Level(Time), 
        <class 'str'> Julian_day(Time), 
        int32 Instrument_number(Time), 
        float32 Exact_wavelengths(Time,Central_wavelenghts), 
        float32 Solar_zenith(Time,Central_wavelenghts), 
        float32 Solar_azimuth(Time,Central_wavelenghts), 
        float32 Lt_mean(Time,Central_wavelenghts), 
        float32 Lt_standard_deviation(Time,Central_wavelenghts), 
        float32 Lt_min_rel(Time,Central_wavelenghts), 
        float32 Li_mean(Time,Central_wavelenghts), 
        float32 Li_standard_deviation(Time,Central_wavelenghts), 
        float32 AOT(Time,Central_wavelenghts), 
        float32 OOT(Time,Central_wavelenghts), 
        float32 ROT(Time,Central_wavelenghts), 
        float32 Lw(Time,Central_wavelenghts), 
        float32 Lw_Q(Time,Central_wavelenghts), 
        float32 Lwn(Time,Central_wavelenghts), 
        float32 Lwn_fonQ(Time,Central_wavelenghts), 
        float32 Pressure(Time), 
        float32 Wind_speed(Time), 
        float32 CHL-A(Time), 
        float32 SSR(Time), 
        float32 O3(Time)
"""
path_main = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'
path = os.path.join(path_main,'netcdf_file')
filename = 'Venise_20_201601001_201612031.nc'
filename_insitu = os.path.join(path,filename)
print(filename_insitu)

if not os.path.exists(filename_insitu):
    print('File does not exist')
    
nc_f0 = Dataset(filename_insitu,'r')

Time = nc_f0.variables['Time'][:]
Level = nc_f0.variables['Level'][:]
Julian_day = nc_f0.variables['Julian_day'][:]
Exact_wavelengths = nc_f0.variables['Exact_wavelengths'][:]
Lwn_fonQ = nc_f0.variables['Lwn_fonQ'][:]

day_vec =np.array([float(Time[i].replace(' ',':').split(':')[0]) for i in range(0,len(Time))])
month_vec =np.array([float(Time[i].replace(' ',':').split(':')[1]) for i in range(0,len(Time))])
year_vec =np.array([float(Time[i].replace(' ',':').split(':')[2]) for i in range(0,len(Time))])
hour_vec =np.array([float(Time[i].replace(' ',':').split(':')[3]) for i in range(0,len(Time))])
minute_vec =np.array([float(Time[i].replace(' ',':').split(':')[4]) for i in range(0,len(Time))])
second_vec =np.array([float(Time[i].replace(' ',':').split(':')[5]) for i in range(0,len(Time))])

doy_vec = np.array([int(float(Julian_day[i])) for i in range(0,len(Time))])
#%% build L2 filename based on date and see if it exist
"""
Example file: S3A_OL_2_WFR____20160528T091009_20160528T091209_20171101T001113_0119_004_321______MR1_R_NT_002
"""
count = 0
f = open(os.path.join(path_main,'OLCI_list.txt'),"w+")

for idx in range(0,len(Time)):
    year_str = str(int(year_vec[idx]))
    
    month_str = str(int(month_vec[idx]))
    if month_vec[idx] < 10:
        month_str = '0'+month_str
        
      
    doy_str = str(int(doy_vec[idx]))  
    if doy_vec[idx] < 100:
        if doy_vec[idx] < 10:
            doy_str = '00'+doy_str
        else:
            doy_str = '0'+doy_str
    
    day_str = str(int(day_vec[idx]))
    if day_vec[idx] < 10:
        day_str = '0'+day_str
    
    dir_path = os.path.join('/DataArchive/OC/OLCI/sources_baseline_2.23',year_str,doy_str)
    L2_filename = 'S3A_OL_2_WFR____'+year_str+month_str+day_str
    
    
#    print(L2_filename)
    path = os.path.join(dir_path,L2_filename)
#    print(path)
    (ls_status, ls_output) = subprocess.getstatusoutput("grep "+L2_filename+" 20160426_20190228.txt")
    if not ls_status:
#        print('--------------------------------------')
        f.write('/'+year_str+'/'+doy_str+'/'+ls_output+'\n')
        count = count+1
print('Matchups Total: '+str(int(count)))
f.close()

(ls_status, ls_output) = subprocess.getstatusoutput('cat '+os.path.join(path_main,'OLCI_list.txt')\
 +'|sort|uniq >'+os.path.join(path_main,'OLCI_list_uniq.txt'))

#%% Look for source, create output dir, unzip, extract box, and save it as nc file
    
    

