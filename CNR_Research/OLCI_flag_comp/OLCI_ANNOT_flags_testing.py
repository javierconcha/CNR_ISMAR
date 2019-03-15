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

#%%
path = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/'
filename1 = 'O2017200090106--med-hr_brdf.nc'
filename2 = 'O2017200090106--med-hr_brdf_w_ANNOT_DROUT.nc'

nc_f1=Dataset(path+filename1, 'r')
nc_f2=Dataset(path+filename2, 'r')

var1=nc_f1.variables
var2=nc_f2.variables

#chl1 = var1['chl'][250:1300,1200:2200]
#chl2 = var2['chl'][250:1300,1200:2200]

chl1 = var1['chl'][:,:]
chl2 = var2['chl'][:,:]

lat1 = var1['lat']
lat2 = var2['lat']
#%%
# chl1 = chl1

plt.figure(figsize=(12,12))
plt.subplot(3, 2, 1)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
plt.imshow(chl1,vmin=0, vmax=0.1, cmap='rainbow') 
plt.title('w/o ANNOT flags')
plt.colorbar(fraction=0.046, pad=0.04)
# plt.ylabel('Damped oscillation')

plt.subplot(3, 2, 2)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
plt.imshow(chl2,vmin=0, vmax=0.1, cmap='rainbow')
plt.title('w/ ANNOT flags')
plt.colorbar(fraction=0.046, pad=0.04)
# plt.xlabel('time (s)')
# plt.ylabel('Undamped')

mask1_valid = ~chl1.mask
mask2_valid = ~chl2.mask
mask_diff = (mask1_valid ^ mask2_valid)
chl1_masked = np.ma.masked_where(~mask_diff,chl1)


plt.subplot(3, 2, 3)
plt.imshow(mask_diff, cmap='gray') 

plt.subplot(3, 2, 4)
current_cmap = plt.cm.get_cmap()
current_cmap.set_bad(color='white')
plt.imshow(chl1_masked,vmin=0, vmax=0.1, cmap='rainbow') 
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(3, 2, 5)
kwargs = dict(bins='auto',histtype='stepfilled', alpha=0.3)

plt.hist(chl1[~chl1.mask], **kwargs) 
plt.xscale('log')

plt.hist(chl2[~chl2.mask], **kwargs) 
plt.xscale('log')
plt.xlabel('chl')
plt.ylabel('Frequency')

plt.show()

#%%
# w/o ANNOT flags
print(np.nanmin(chl1[~chl1.mask]))
print(np.nanmax(chl1[~chl1.mask]))
print(np.nanstd(chl1[~chl1.mask]))
print(np.nanmedian(chl1[~chl1.mask]))
print(np.nanmean(chl1[~chl1.mask]))
print(sum(sum(~chl1.mask)))
print(type(chl1))

# w/ ANNOT flags
print(np.nanmin(chl2[~chl2.mask]))
print(np.nanmax(chl2[~chl2.mask]))
print(np.nanstd(chl2[~chl2.mask]))
print(np.nanmedian(chl2[~chl2.mask]))
print(np.nanmean(chl2[~chl2.mask]))
print(type(chl2))
print(sum(sum(~chl2.mask)))
print('Pixel Difference:\n')
print(sum(sum(~chl1.mask))-sum(sum(~chl2.mask)))
print(sum(sum(~chl1_masked.mask)))
