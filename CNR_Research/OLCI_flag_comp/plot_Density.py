#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:56:24 2019

@author: javier
"""
#%% Initializaion
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap 
import os.path

import sys  
sys.path.append('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Python_examples/') 
from plot_map_hres import plot_mask
#%% To plot density  
path_in = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/'

#region_flag = 'med'
region_flag = 'NAS'

if region_flag=='med':
    filename = 'Density_med_Final.nc' # open an original for to copy properties to output file
    parallel_steps = [30, 35, 40, 45]
    meridian_steps = [-5, 0, 5, 10, 15, 20, 25, 30, 35]
elif region_flag== 'NAS':
    filename = 'Density_NAS_Final.nc' # open an original for to copy properties to output file
    meridian_steps = [12.5, 13, 13.5]
    parallel_steps = [44, 44.5, 45, 45.5]   

nc_f0=Dataset(os.path.join(path_in,filename), 'r')
    
lat0 = nc_f0.variables['lat'][:]
lon0 = nc_f0.variables['lon'][:]

chl_diff_den = nc_f0.variables['chl_diff_den'][:,:]
cover_sum = nc_f0.variables['cover_sum'][:,:]

ylen = len(lat0)
xlen = len(lon0)

coords=[(min(lon0),min(lat0)), (min(lon0),max(lat0)), (max(lon0),max(lat0)), (max(lon0),min(lat0))]

#%%
fig  = plt.figure(figsize=(10,10))

plt.subplots_adjust(hspace=0.5)

#%
plt.subplot(3,1,1)
plt.title('Absolute Density')
m = plot_mask(lat0,lon0,coords,meridian_steps,parallel_steps,rivers_flag = False)
m.imshow(chl_diff_den,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
clb.ax.set_xlabel('counts')

#%
plt.subplot(3,1,2)
plt.title('Valid Occurences')
m = plot_mask(lat0,lon0,coords,meridian_steps,parallel_steps,rivers_flag = False)
m.imshow(cover_sum,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
clb.ax.set_xlabel('counts')

#%
plt.subplot(3,1,3)
plt.title('Occurence Percentage')
occurence_percent = chl_diff_den/cover_sum
m = plot_mask(lat0,lon0,coords,meridian_steps,parallel_steps,rivers_flag = False)
m.imshow(occurence_percent,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
clb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
clb.set_ticklabels(['0', '20', '40', '60', '80', '100'])
clb.ax.set_xlabel('%')

figname = os.path.join(path_in,'Density_'+region_flag+'_Final.pdf')
fig.savefig(figname, dpi=300)

plt.show()
plt.close()