#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:08:37 2019

@author: javier
"""
#%% Initializaion
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
import os.path
import os
os.environ['QT_QPA_PLATFORM']='offscreen' # to avoid error "QXcbConnection: Could not connect to display"

from os import access, R_OK
#%% function to plot different products in a map
def plot_map(var,lat,lon,meridian_steps,parallel_steps):
    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
    	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
    x,y=np.meshgrid(lon, lat)
    m.drawparallels(parallel_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians(meridian_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(var,origin='upper', extent=[min(lon), max(lon), min(lat), max(lat)],\
                                          norm=LogNorm(), cmap='rainbow')
#    plt.colorbar(fraction=0.046, pad=0.04)
#%% function to plot the products and histograms

def plot_test_NAS_ANNOT_flags(path_in,path_out,filename1,fout,meridian_steps,parallel_steps):
    #%%
    nc_f1=Dataset(path_in+filename1, 'r')
    lat1 = nc_f1.variables['lat'][:]
    lon1 = nc_f1.variables['lon'][:]
    lat2 = lat1
    lon2 = lon1
    
    chl1 = nc_f1.groups['Geo_data']['chl'][:]
    chl1.fill_value = -999.0
    ANNOT_flag = nc_f1.groups['Geo_data']['ANNOT_DROUT'][:]   
    
    ylen = len(lat1)
    xlen = len(lon1)
    chl2 = np.ma.zeros((ylen, xlen), dtype=np.float32) #
    chl2.mask=True
    chl2.fill_value = chl1.fill_value
    
    chl2.mask = chl1.mask | (ANNOT_flag == 1)
    chl2[~chl2.mask] = chl1[~chl2.mask]
    #%%
    if ((len(lon1)*len(lat1) == np.sum(chl1.mask)) & (len(lon2)*len(lat2) == np.sum(chl2.mask))) \
    | (len(lat1)*len(lon1)-sum(sum(chl1.mask))<2) | (len(lat2)*len(lon2)-sum(sum(chl2.mask))<2): 
        print('File empty: '+filename1)
        fout.write('File empty: '+filename1+'\n')
        return False, False, False
    else:
        #%% Create figure and subplot for chl w/o ANNOTS flags
        pad_value = 0.07
        plt.figure(figsize=(12,12))
        plt.suptitle(filename1)
        plt.subplot(3, 2, 1)
        plot_map(chl1,lat1,lon1,meridian_steps,parallel_steps) 
        plt.title('chl w/o ANNOT flags')
        clb = plt.colorbar(fraction=0.046, pad=pad_value,orientation='horizontal')
        plt.clim(1E-3,1E2)
#        clb.ax.set_xlabel('chl')
        
        #%% subplot for chl w/ ANNOTS flags
        plt.subplot(3, 2, 2)
        plot_map(chl2,lat2,lon2,meridian_steps,parallel_steps)  
        plt.title('chl w/ ANNOT flags')
        clb = plt.colorbar(fraction=0.046, pad=pad_value,orientation='horizontal')
        plt.clim(1E-3,1E2)
#        clb.ax.set_xlabel('chl')
        
        #%% create masked difference and mask
        mask1_valid = ~chl1.mask
        mask2_valid = ~chl2.mask
        mask_diff = np.ma.masked_where((mask1_valid ^ mask2_valid)==0,(mask1_valid ^ mask2_valid))
        
        chl_diff = np.ma.masked_where(~(mask1_valid ^ mask2_valid),chl1)
        
        #%% Mask of the difference pixels
        plt.subplot(3, 2, 3)
        #plt.imshow(mask_diff, cmap='gray') 
        m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
            	llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='l')
        x,y=np.meshgrid(lon1, lat1)
        m.drawparallels(parallel_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
        m.drawmeridians(meridian_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
        m.drawcoastlines(linewidth=0.1)
        m.imshow(mask_diff,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],\
                                                   cmap='gist_rainbow',interpolation='nearest')
        #    plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Difference Pixels -- Mask')
        
        #%% chl difference pixels 
        plt.subplot(3, 2, 4)
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='white')
        
        m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
            	llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='l')
        x,y=np.meshgrid(lon1, lat1)
        m.drawparallels(parallel_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
        m.drawmeridians(meridian_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
        m.drawcoastlines(linewidth=0.1)
        m.imshow(chl_diff,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],\
                                                   norm=LogNorm(),cmap='rainbow')
        clb = plt.colorbar(fraction=0.046, pad=pad_value,orientation='horizontal')
        plt.clim(1E-3,1E2)
#        clb.ax.set_xlabel('chl')
        plt.title('Difference Pixels -- chl')
        
        #%% histogram of the two chl products
        plt.subplot(3, 2, 5)
        kwargs = dict(bins=np.logspace(-4,2,200),histtype='step')
        
        plt.hist(chl1[~chl1.mask],color='blue', **kwargs) 
        plt.xscale('log')
        plt.hist(chl2[~chl2.mask],color='red', **kwargs) 
        plt.xscale('log')
        plt.xlabel('chl')
        plt.ylabel('Frequency')
        plt.xlim(1E-3,1E3)
        
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
        
        #%% histogram of the difference pixels
        plt.subplot(3, 2, 6)
        plt.hist(chl_diff[~chl_diff.mask],color='black', **kwargs) 
        plt.xscale('log')
        plt.xlabel('chl')
        plt.ylabel('Frequency')
        plt.xlim(1E-3,1E3)
        
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
        
        ofname = filename1[:-3]+'_ANNOT_flag.pdf'
        ofname = os.path.join(path_out,ofname)
        
#        plt.tight_layout()
        
        plt.savefig(ofname, dpi=200)
    #    plt.show()
        plt.close()
        
        # Save netCDF4 file
    
        ofname = filename1[:-3]+'_pxdiff.nc'
        ofname = os.path.join(path_out,ofname)
        fmb = Dataset(ofname, 'w', format='NETCDF4')
        fmb.description = 'Chl difference netCDF4 file'
        
        fmb.createDimension("lat", len(lat1))
        fmb.createDimension("lon", len(lon1))
        
        lat = fmb.createVariable('lat',  'single', ('lat',)) 
        lon = fmb.createVariable('lon',  'single', ('lon',))
        
        lat[:] = lat1
        lon[:] = lon1
        
        gridd_var=fmb.createVariable('chl_diff', 'single', ('lat', 'lon',), fill_value=chl_diff.fill_value, zlib=True, complevel=6)
    
        gridd_var[:] = chl_diff
    
        fmb.close()
        
        return chl_diff.mask, mask1_valid, True
#%%
def main():
    """business logic for when running this module as the primary one!"""
    print('Main Code!')
    
    host = 'mac'
#    host = 'vm'
    
    if host == 'mac':
        path_in = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/DataArchive/OLCI_NAS/20160426_20190228/'   
        path_out = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/data/NAS/'        
        path_list = path_in
    elif host == 'vm':
        path_in = '/store3/OLCI_NAS/20160426_20190228/'
        path_out = '/home/Vittorio.Brando/Javier/data'
        path_list = '/home/Vittorio.Brando/Javier/codes'
    else:
        print('Not sure from where this script will be run!')
        
    #%% To plot density      
    filename = 'VGOCS_2016117094003_O.nc' # open an original for to copy properties to output file
    
    nc_f0=Dataset(os.path.join(path_in,filename), 'r')
        
    lat0 = nc_f0.variables['lat'][:]
    lon0 = nc_f0.variables['lon'][:]
    
    ylen = len(lat0)
    xlen = len(lon0)
    
    fout = open(os.path.join(path_out,'output_NAS.txt'),'w+')
    
    meridian_steps = [12.5, 13, 13.5]
    parallel_steps = [44, 44.5, 45, 45.5]
    #%%
    
    chl_diff_den = np.ma.zeros((ylen, xlen), dtype=np.float32) #   
    cover_sum = np.ma.zeros((ylen, xlen), dtype=np.float32) # total of observations
    
    count = 0;
    
    file = open(os.path.join(path_list,'file_list.txt'),'r')
    
    for line in file:       
    #            print(year+doy)    
        
        filename1 = line[2:-1]    
#        print(path_in+filename1)
        if os.path.exists(path_in+filename1):
            if access(path_in+filename1, R_OK) :
                chl_diff_mask, coverage, valid_flag = plot_test_NAS_ANNOT_flags\
                (path_in,path_out,filename1,fout,meridian_steps,parallel_steps)
                if valid_flag:   
                    print('File processing: '+filename1)
                    fout.write('File processing: '+filename1+'\n')
                    chl_diff_den = ~chl_diff_mask + chl_diff_den
                    
                    cover_sum = coverage + cover_sum
                    count = count + 1
            else:
                print('File access denied: '+filename1)                        
        else:
            print('File not found: '+filename1)
            fout.write('File not found: '+filename1+'\n')
    
    
    #%%
    plt.figure(figsize=(10,10))
    ax = plt.subplot(2,2,1)
    ax.set_title('Absolute Density (counts)')
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='white')
    
    chl_diff_den.mask = chl_diff_den==0
    
    m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
        	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
    x,y=np.meshgrid(lon0, lat0)
    m.drawparallels(parallel_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians(meridian_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(chl_diff_den,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                               cmap='rainbow',interpolation='nearest')
    
    clb = plt.colorbar(fraction=0.046, pad=0.05,orientation='horizontal')
#    clb.ax.set_xlabel('Absolute Density (counts)')
    #%%
    ax = plt.subplot(2,2,2)
    ax.set_title('Valid Occurences (counts)')
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='white')
    
    cover_sum.mask = cover_sum==0
    
    m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
        	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
    x,y=np.meshgrid(lon0, lat0)
    m.drawparallels(parallel_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians(meridian_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(cover_sum,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                               cmap='rainbow',interpolation='nearest')
    m.drawlsmask(land_color='white',ocean_color='none',resolution='f', grid=1.25)
    clb = plt.colorbar(fraction=0.046, pad=0.05,orientation='horizontal')
#    clb.ax.set_xlabel('Valid Occurences (counts)')
    #%%
    ax = plt.subplot(2,1,2)
    ax.set_title('Occurence Percentage [%]')
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='white')
    
    occurence_percent = chl_diff_den/cover_sum
    
    m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
        	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
    x,y=np.meshgrid(lon0, lat0)
    m.drawparallels(parallel_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians(meridian_steps,labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(occurence_percent,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                               cmap='rainbow',vmin=0, vmax=0.5,interpolation='nearest')
    
    clb = plt.colorbar(fraction=0.046, pad=0.05,orientation='horizontal')
#    clb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#    clb.set_ticklabels(['0', '20', '40', '60', '80', '100'])
    clb.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    clb.set_ticklabels(['0', '10', '20', '30', '40', '50'])    
#    clb.ax.set_xlabel('Occurence Percentage [%]')
    
    plt.tight_layout()
    
    figname = os.path.join(path_out,'Density_NAS.pdf')
    #    print(figname)
    plt.savefig(figname, dpi=200)
    #plt.show()
    plt.close()
    
    #%% Save netCDF4 file
    
    ofname = 'Density_NAS.nc'
    ofname = os.path.join(path_out,ofname)
    if os.path.exists(ofname):
        os.remove(ofname)
        print('File removed!')
        
    fmb = Dataset(ofname, 'w', format='NETCDF4')
    fmb.description = 'Chl difference density netCDF4 file'
    
    fmb.createDimension("lat", len(lat0))
    fmb.createDimension("lon", len(lon0))
    
    lat = fmb.createVariable('lat',  'single', ('lat',)) 
    lon = fmb.createVariable('lon',  'single', ('lon',))
    
    lat[:] = lat0
    lon[:] = lon0
    
    gridd_var1=fmb.createVariable('chl_diff_den', 'single', ('lat', 'lon',),fill_value=chl_diff_den.fill_value, zlib=True, complevel=6)
    gridd_var1[:,:] = chl_diff_den
    
    gridd_var2=fmb.createVariable('cover_sum', 'single', ('lat', 'lon',), fill_value=cover_sum.fill_value, zlib=True, complevel=6)
    gridd_var2[:,:] = cover_sum
    
    fmb.close()
    fout.close()                            
#%%
if __name__ == '__main__':
    main()