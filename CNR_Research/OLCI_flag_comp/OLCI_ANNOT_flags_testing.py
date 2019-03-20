#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:51:48 2019

@author: javier
"""
#%%
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import colors

#%%
path = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/2017/199/'


filename1 = 'O2017199--med-hr_brdf.nc'
filename2 = 'O2017199--med-hr_brdf_w_ANNOT_DROUT.nc'

nc_f1=Dataset(path+filename1, 'r')
nc_f2=Dataset(path+filename2, 'r')

var1=nc_f1.variables
var2=nc_f2.variables

#chl1 = var1['chl'][250:1300,1200:2200]
#chl2 = var2['chl'][250:1300,1200:2200]

chl1 = var1['chl'][:,:]
chl2 = var2['chl'][:,:]

lat1 = var1['lat'][:]
lat2 = var2['lat'][:]

lon1 = var1['lon'][:]
lon2 = var2['lon'][:]

#%%
# chl1 = chl1


plt.figure(figsize=(12,12))
plt.subplot(3, 2, 1)
plot_map(chl1,lat1,lon1)   
plt.title('chl w/o ANNOT flags')
# plt.ylabel('Damped oscillation')

plt.subplot(3, 2, 2)
plot_map(chl2,lat2,lon2)   
plt.title('chl w/ ANNOT flags')
# plt.xlabel('time (s)')
# plt.ylabel('Undamped')

mask1_valid = ~chl1.mask
mask2_valid = ~chl2.mask
mask_diff = np.ma.masked_where((mask1_valid ^ mask2_valid)==0,(mask1_valid ^ mask2_valid))

chl1_masked = np.ma.masked_where(~(mask1_valid ^ mask2_valid),chl1)


plt.subplot(3, 2, 3)
#plt.imshow(mask_diff, cmap='gray') 
m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
    	llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='l')
x,y=np.meshgrid(lon1, lat1)
m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1])
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1])
m.drawlsmask(land_color='white',ocean_color='white',resolution='l', grid=5)
m.drawcoastlines()
cmap1 = colors.ListedColormap(['white','blue','red'])
m.imshow(mask_diff,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],\
                                           cmap='gist_rainbow',interpolation='nearest')
#    plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Difference Pixels -- Mask')


plt.subplot(3, 2, 4)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')

m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
    	llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='l')
x,y=np.meshgrid(lon1, lat1)
m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1])
m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1])
m.drawlsmask(land_color='white',ocean_color='white',resolution='l', grid=5)
m.drawcoastlines()
m.imshow(chl1_masked,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],\
                                           vmin=0, vmax=0.1,cmap='rainbow')
#plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Difference Pixels -- chl')

plt.subplot(3, 2, 5)
kwargs = dict(bins=np.logspace(-4,2,200),histtype='step', alpha=0.3)

n, bins, patches = plt.hist(chl1[~chl1.mask], **kwargs) 
plt.xscale('log')
plt.hist(chl2[~chl2.mask], **kwargs) 
plt.xscale('log')
plt.xlabel('chl')
plt.ylabel('Frequency')
#plt.xlim(1E-4,1E1)

str1 = 'w/o ANNOT flags\n\
min: {:f}\n\
max: {:f}\n\
std: {:f}\n\
median: {:f}\n\
mean: {:f}\n\
N: {:,.0f}'\
.format(np.nanmin(chl1[~chl1.mask]),
        np.nanmax(chl1[~chl1.mask]),
        np.nanstd(chl1[~chl1.mask]),
        np.nanmedian(chl1[~chl1.mask]),
        np.nanmean(chl1[~chl1.mask]),
        sum(sum(~chl1.mask)))

str2 = 'w/ ANNOT flags\n\
min: {:f}\n\
max: {:f}\n\
std: {:f}\n\
median: {:f}\n\
mean: {:f}\n\
N: {:,.0f}\n\
Diff: {:,.0f}'\
.format(np.nanmin(chl2[~chl2.mask]),
        np.nanmax(chl2[~chl2.mask]),
        np.nanstd(chl2[~chl2.mask]),
        np.nanmedian(chl2[~chl2.mask]),
        np.nanmean(chl2[~chl2.mask]),
        sum(sum(~chl2.mask)),
        sum(sum(~chl1.mask))-sum(sum(~chl2.mask)))

bottom, top = plt.ylim()
left, right = plt.xlim()
xpos = 10**(np.log10(left)+0.02*((np.log10(right))-(np.log10(left))))
plt.text(xpos, 0.45*top, str1, fontsize=12,color='blue')
xpos = 10**(np.log10(left)+0.6*((np.log10(right))-(np.log10(left))))
plt.text(xpos, 0.37*top, str2, fontsize=12,color='brown')

plt.subplot(3, 2, 6)
plt.hist(chl1_masked[~chl1_masked.mask], **kwargs) 
plt.xscale('log')
plt.xlabel('chl')
plt.ylabel('Frequency')
#plt.xlim(1E-4,1E1)

str3 = 'Diff. Pixels\n\
min: {:f}\n\
max: {:f}\n\
std: {:f}\n\
median: {:f}\n\
mean: {:f}\n\
N: {:,.0f}'\
.format(np.nanmin(chl1_masked[~chl1_masked.mask]),
        np.nanmax(chl1_masked[~chl1_masked.mask]),
        np.nanstd(chl1_masked[~chl1_masked.mask]),
        np.nanmedian(chl1_masked[~chl1_masked.mask]),
        np.nanmean(chl1_masked[~chl1_masked.mask]),
        sum(sum(~chl1_masked.mask)))

bottom, top = plt.ylim()
left, right = plt.xlim()
xpos = 10**(np.log10(left)+0.6*((np.log10(right))-(np.log10(left))))
plt.text(xpos, 0.45*top, str3, fontsize=12,color='blue')

plt.show()
plt.close()

#%%
# w/o ANNOT flags
#print(np.nanmin(chl1[~chl1.mask]))
#print(np.nanmax(chl1[~chl1.mask]))
#print(np.nanstd(chl1[~chl1.mask]))
#print(np.nanmedian(chl1[~chl1.mask]))
#print(np.nanmean(chl1[~chl1.mask]))
#print(sum(sum(~chl1.mask)))
#print(type(chl1))
#
## w/ ANNOT flags
#print(np.nanmin(chl2[~chl2.mask]))
#print(np.nanmax(chl2[~chl2.mask]))
#print(np.nanstd(chl2[~chl2.mask]))
#print(np.nanmedian(chl2[~chl2.mask]))
#print(np.nanmean(chl2[~chl2.mask]))
#print(type(chl2))
#print(sum(sum(~chl2.mask)))
#print('Pixel Difference:\n')
#print(sum(sum(~chl1.mask))-sum(sum(~chl2.mask)))
#print(sum(sum(~chl1_masked.mask)))
#%%
#figpath = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/Figures/'
#figname = 'fig1'
#plt.savefig(figpath+figname, dpi=200)

#%%
def plot_map(var,lat,lon):
    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
    	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
    x,y=np.meshgrid(lon, lat)
    m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1])
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1])
    m.drawlsmask(land_color='white',ocean_color='white',resolution='l', grid=5)
    m.drawcoastlines()
    m.imshow(var,origin='upper', extent=[min(lon), max(lon), min(lat), max(lat)],\
                                          vmin=0, vmax=0.1, cmap='rainbow')
#    plt.colorbar(fraction=0.046, pad=0.04)