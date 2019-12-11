#!/usr/bin/env python3
# coding: utf-8
"""
Created on Fri Dec  6 11:25:50 2019

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
#%% Initializaion
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import os.path
import os

#%%

path_main = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Case12/'
path_data = os.path.join(path_main,'data')
path_out = os.path.join(path_main,'output')

filename = 'Case12_med_v4_1997_2015.nc'

nc_f0=Dataset(os.path.join(path_out,filename), 'r')

lat0 = nc_f0.variables['lat'][:]
lon0 = nc_f0.variables['lon'][:]

case1_counts  = nc_f0.variables['case1_counts'][:]
case2_counts  = nc_f0.variables['case2_counts'][:]
case01_counts = nc_f0.variables['case01_counts'][:]
case02_counts = nc_f0.variables['case02_counts'][:]
case03_counts = nc_f0.variables['case03_counts'][:]
case04_counts = nc_f0.variables['case04_counts'][:]
case05_counts = nc_f0.variables['case05_counts'][:]
case06_counts = nc_f0.variables['case06_counts'][:]
case07_counts = nc_f0.variables['case07_counts'][:]
case08_counts = nc_f0.variables['case08_counts'][:]
case09_counts = nc_f0.variables['case09_counts'][:]
case10_counts = nc_f0.variables['case10_counts'][:]
valid_counts  = nc_f0.variables['valid_counts'][:]

ylen = len(lat0)
xlen = len(lon0)
#%%
fig = plt.figure(figsize=(12,18))
# Case 1		
plt.subplot(5,3,1)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case1_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('Case 1 [%]')		

# Case 2
plt.subplot(5,3,2)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case2_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('Case 2 [%]')

# valid counts
plt.subplot(5,3,14)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('valid counts [counts]')	

# 0.0 <= case <= 0.1
plt.subplot(5,3,4)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case01_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.0 <= case <= 0.1  [%]')

# 0.1 < case <= 0.2
plt.subplot(5,3,5)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case02_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.1 < case <= 0.2 [%]')

# 0.2 < case <= 0.3
plt.subplot(5,3,6)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case03_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.2 < case <= 0.3 [%]')

# 0.3 < case <= 0.4
plt.subplot(5,3,7)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case04_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.3 < case <= 0.4 [%]')

# 0.4 < case <= 0.5
plt.subplot(5,3,8)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case05_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.4 < case <= 0.5 [%]')

# 0.5 < case <= 0.6
plt.subplot(5,3,9)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case06_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.5 < case <= 0.6 [%]')

# 0.6 < case <= 0.7
plt.subplot(5,3,10)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case07_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.6 < case <= 0.7 [%]')

# 0.7 < case <= 0.8
plt.subplot(5,3,11)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case08_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.7 < case <= 0.8 [%]')

# 0.8 < case <= 0.9
plt.subplot(5,3,12)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case09_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.8 < case <= 0.9 [%]')

# 0.9 < case <= 1.0
plt.subplot(5,3,13)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(case10_counts*100/valid_counts, extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('0.9 < case <= 1.0 [%]')

# valid counts
plt.subplot(5,3,15)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
    	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
x,y=np.meshgrid(lon0, lat0)
m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
m.drawcoastlines(linewidth=0.1)
m.imshow(valid_counts*100/valid_counts.max().max(), extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                           cmap='rainbow',interpolation='nearest')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='vertical')
plt.gca().set_title('valid counts [%]')	

fig.tight_layout(pad=2)

figname = os.path.join(path_out,filename.split('.')[0]+'.pdf')
print(figname)
plt.savefig(figname, dpi=200)
#plt.show()