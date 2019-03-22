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
import os.path

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
#%%


def plot_test_ANNOT_flags(path,filename1,filename2,year,doy):
    nc_f1=Dataset(path+filename1, 'r')
    nc_f2=Dataset(path+filename2, 'r')
    
    chl1 = nc_f1.variables['chl'][:,:]
    chl2 = nc_f2.variables['chl'][:,:]
    
    lat1 = nc_f1.variables['lat'][:]
    lat2 = nc_f2.variables['lat'][:]
    
    lon1 = nc_f1.variables['lon'][:]
    lon2 = nc_f2.variables['lon'][:]
    
    
    #%%
    plt.figure(figsize=(12,12))
    plt.suptitle('O'+year+doy)
    plt.subplot(3, 2, 1)
    plot_map(chl1,lat1,lon1)   
    plt.title('chl w/o ANNOT flags')
    clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
    clb.ax.set_xlabel('chl')
    
    #%%
    plt.subplot(3, 2, 2)
    plot_map(chl2,lat2,lon2)   
    plt.title('chl w/ ANNOT flags')
    clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
    clb.ax.set_xlabel('chl')
    
    #%% create masked difference and mask
    mask1_valid = ~chl1.mask
    mask2_valid = ~chl2.mask
    mask_diff = np.ma.masked_where((mask1_valid ^ mask2_valid)==0,(mask1_valid ^ mask2_valid))
    
    chl_diff = np.ma.masked_where(~(mask1_valid ^ mask2_valid),chl1)
    
    #%%
    plt.subplot(3, 2, 3)
    #plt.imshow(mask_diff, cmap='gray') 
    m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
        	llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='l')
    x,y=np.meshgrid(lon1, lat1)
    m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1])
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1])
    m.drawlsmask(land_color='white',ocean_color='white',resolution='l', grid=5)
    m.drawcoastlines()
    m.imshow(mask_diff,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],\
                                               cmap='gist_rainbow',interpolation='nearest')
    #    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Difference Pixels -- Mask')
    
    #%%
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
    m.imshow(chl_diff,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],\
                                               vmin=0, vmax=0.1,cmap='rainbow')
    clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
    clb.ax.set_xlabel('chl')
    plt.title('Difference Pixels -- chl')
    
    #%%
    plt.subplot(3, 2, 5)
    kwargs = dict(bins=np.logspace(-4,2,200),histtype='step')
    
    plt.hist(chl1[~chl1.mask],color='blue', **kwargs) 
    plt.xscale('log')
    plt.hist(chl2[~chl2.mask],color='red', **kwargs) 
    plt.xscale('log')
    plt.xlabel('chl')
    plt.ylabel('Frequency')
#    plt.xlim(1E-6,1E5)
    
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
    plt.text(xpos, 0.37*top, str2, fontsize=12,color='red')
    
    #%%
    plt.subplot(3, 2, 6)
    plt.hist(chl_diff[~chl_diff.mask],color='black', **kwargs) 
    plt.xscale('log')
    plt.xlabel('chl')
    plt.ylabel('Frequency')
#    plt.xlim(1E-6,1E5)
    
    str3 = 'Diff. Pixels\n\
min: {:f}\n\
max: {:f}\n\
std: {:f}\n\
median: {:f}\n\
mean: {:f}\n\
N: {:,.0f}'\
    .format(np.nanmin(chl_diff[~chl_diff.mask]),
            np.nanmax(chl_diff[~chl_diff.mask]),
            np.nanstd(chl_diff[~chl_diff.mask]),
            np.nanmedian(chl_diff[~chl_diff.mask]),
            np.nanmean(chl_diff[~chl_diff.mask]),
            sum(sum(~chl_diff.mask)))
    
    bottom, top = plt.ylim()
    left, right = plt.xlim()
    xpos = 10**(np.log10(left)+0.6*((np.log10(right))-(np.log10(left))))
    plt.text(xpos, 0.45*top, str3, fontsize=12,color='black')
 
    #%
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
    #print(sum(sum(~chl_diff.mask)))
    #%
    figname = path+'Figures'+'/'+'O'+year+doy+'ANNOT_flag_test.pdf'
#    print(figname)
    plt.savefig(figname, dpi=200)
#    plt.show()
    plt.close()
    
    return chl_diff
#%%
def main():
    """business logic for when running this module as the primary one!"""
    print('Main Code!')
    
    path = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/'
    
    n_im = 4 #number of images
    chl_diff_3d=np.ma.zeros((n_im,1580, 3308), dtype=np.float32)
    count = 0;
    
    for year_idx in range(2016, 2018):
        for doy_idx in range(199, 201):
            year = str(year_idx)
            doy = str(doy_idx)
            print(year)
            print(doy)
    
            
            
            filename1 = year+'/'+doy+'/''O'+year+doy+'--med-hr_brdf.nc'
            filename2 = year+'/'+doy+'/''O'+year+doy+'--med-hr_brdf_w_ANNOT_DROUT.nc'
            
            if os.path.exists(path+filename1) & os.path.exists(path+filename2):
                chl_diff = plot_test_ANNOT_flags(path,filename1,filename2,year,doy)
                
                chl_diff_3d[count,:] = chl_diff
                count = count + 1
    #%% To plot density
    
    filename = '2016/199/O2016199--med-hr_brdf.nc'
    
    nc_f0=Dataset(path+filename, 'r')
        
    lat0 = nc_f0.variables['lat'][:]
    lon0 = nc_f0.variables['lon'][:]
    
    plt.figure(figsize=(10,10))
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='white')
    
    chl_diff_den = np.sum(~chl_diff_3d.mask,axis=0)
    
    m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
        	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
    x,y=np.meshgrid(lon0, lat0)
    m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1])
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1])
    m.drawcoastlines()
    m.imshow(chl_diff_den,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                               cmap='rainbow',interpolation='nearest')
    m.drawlsmask(land_color='white',ocean_color='none',resolution='l', grid=5)
    
    clb = plt.colorbar(fraction=0.046, pad=0.04,orientation='horizontal')
    clb.ax.set_xlabel('Density')
    
    figname = path+'Figures/'+'Density.pdf'
    #    print(figname)
    plt.savefig(figname, dpi=200)
    #plt.show()

if __name__ == '__main__':
    main()