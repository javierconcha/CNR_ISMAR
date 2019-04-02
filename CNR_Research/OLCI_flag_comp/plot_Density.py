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
#%% To plot density  
path_in = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/'

filename = 'Density_med_Final.nc' # open an original for to copy properties to output file

nc_f0=Dataset(os.path.join(path_in,filename), 'r')
    
lat0 = nc_f0.variables['lat'][:]
lon0 = nc_f0.variables['lon'][:]

chl_diff_den = nc_f0.variables['chl_diff_den'][:,:]
cover_sum = nc_f0.variables['cover_sum'][:,:]

ylen = len(lat0)
xlen = len(lon0)

#%%
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')

m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(chl_diff_den,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow')

clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
clb.ax.set_xlabel('Absolute Density (counts)')
#%%
plt.subplot(3,1,2)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')

m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(cover_sum,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow')
m.drawlsmask(land_color='white',ocean_color='none',resolution='f', grid=1.25)
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
clb.ax.set_xlabel('Valid Occurences (counts)')
#%%
plt.subplot(3,1,3)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')

occurence_percent = chl_diff_den/cover_sum

m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(occurence_percent,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',vmin=0)


clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
clb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
clb.set_ticklabels(['0', '20', '40', '60', '80', '100'])
clb.ax.set_xlabel('Occurence Percentage [%]')

figname = os.path.join(path_in,'Density_med_Final.pdf')
#    print(figname)
plt.savefig(figname, dpi=300)
plt.close()