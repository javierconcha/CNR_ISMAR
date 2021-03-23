#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Aug  6 21:40:14 2019
Get filenames from EUMETSAT servers based on the in situ data in netcdf file
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
#%%
import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import subprocess
import matplotlib.pyplot as plt
import sys


import datetime
from scipy import stats

import olci_getscenes
import Matchups_hres
import argparse

path_main = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'

#%%
print('Main Code!')
path_out = os.path.join(path_main,'Figures')
path = os.path.join(path_main,'netcdf_file')

station_list = ['Venise','Galata_Platform','Gloria','Helsinki_Lighthouse','Gustav_Dalen_Tower']
parser = argparse.ArgumentParser(description="Get filenames from EUMETSAT servers based on the in situ data in netcdf file.")
parser.add_argument("-s", "--station" , help="The Aeronet OC station", type=str,choices=station_list)
args = parser.parse_args()

if args.station:
    station_name  = args.station
else:
    station_name = 'Venise'
print('The station name is: '+station_name)   

if station_name == 'Venise':
    filename = 'Venise_20_201604026_201908031.nc'
elif station_name == 'Galata_Platform':
    filename = 'Galata_Platform_20_201601001_201908031.nc'
elif station_name == 'Gloria':
    filename = 'Gloria_20_201601001_201908031.nc'
elif station_name == 'Helsinki_Lighthouse':
    filename = 'Helsinki_Lighthouse_20_201604026_201908031.nc'
elif station_name == 'Gustav_Dalen_Tower':
    filename = 'Gustav_Dalen_Tower_20_201604026_201908031.nc'
    
filename_insitu = os.path.join(path,filename)
if not os.path.exists(filename_insitu):
    print('File does not exist')
    
nc_f0 = Dataset(filename_insitu,'r')

Time = nc_f0.variables['Time'][:]
Level = nc_f0.variables['Level'][:]
Julian_day = nc_f0.variables['Julian_day'][:]
Exact_wavelengths = nc_f0.variables['Exact_wavelengths'][:]
Lwn_fonQ = nc_f0.variables['Lwn_fonQ'][:]

nc_f0.close()

day_vec =np.array([float(Time[i].replace(' ',':').split(':')[0]) for i in range(0,len(Time))])
month_vec =np.array([float(Time[i].replace(' ',':').split(':')[1]) for i in range(0,len(Time))])
year_vec =np.array([float(Time[i].replace(' ',':').split(':')[2]) for i in range(0,len(Time))])
hour_vec =np.array([float(Time[i].replace(' ',':').split(':')[3]) for i in range(0,len(Time))])
minute_vec =np.array([float(Time[i].replace(' ',':').split(':')[4]) for i in range(0,len(Time))])
second_vec =np.array([float(Time[i].replace(' ',':').split(':')[5]) for i in range(0,len(Time))])

Julian_day_vec =np.array([float(Julian_day[i]) for i in range(0,len(Time))])
date_format = "%d:%m:%Y %H:%M:%S"
ins_time = np.array([datetime.datetime.strptime(Time[i], date_format) for i in range(0,len(Time))])

doy_vec = np.array([int(float(Julian_day[i])) for i in range(0,len(Time))])

lat_ins, lon_ins = Matchups_hres.get_lat_lon_ins(station_name)

f = open(path_main+'OLCI_list_'+station_name+'.txt','a+')

last_day = datetime.datetime(1990,1,1)

for i in range(len(Time)):
    
    date1 = datetime.datetime(int(year_vec[i]),int(month_vec[i]),int(day_vec[i]))
    
    if date1 != last_day:
        last_day = date1
        print('--------------------------------------------------------')
        print(date1)
        # Each degree of latitude is approximately 111 kilometers apart.    
        lat1 = lat_ins-0.05 # 111*0.05 = 5.55 km
        lat2 = lat_ins+0.05
    
        lon1 = lon_ins-0.05
        lon2 = lon_ins+0.05
        
        file_list = olci_getscenes.olci_get(date1,lat1,lat2,lon1,lon2)
        
        if file_list:
            for r in file_list:
                f.write(r+'\n')
            
f.close()  