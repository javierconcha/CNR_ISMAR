#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:51:48 2019

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
from os.path import isfile
#%% function to plot different products in a map
def plot_map(var,lat,lon):
    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
    	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
    x,y=np.meshgrid(lon, lat)
    m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(var,origin='upper', extent=[min(lon), max(lon), min(lat), max(lat)],\
                                          norm=LogNorm(), cmap='rainbow')
#    plt.colorbar(fraction=0.046, pad=0.04)
#%% function to plot the products and histograms

def plot_test_ANNOT_flags(path_in,path_out,filename1,filename2,year,doy,fout,site_name):
    nc_f1=Dataset(path_in+filename1, 'r')
    nc_f2=Dataset(path_in+filename2, 'r')
    
    chl1 = nc_f1.variables['chl'][:,:]
    chl2 = nc_f2.variables['chl'][:,:]
    
    lat1 = nc_f1.variables['lat'][:]
    lat2 = nc_f2.variables['lat'][:]
    
    lon1 = nc_f1.variables['lon'][:]
    lon2 = nc_f2.variables['lon'][:]
    
    if ((len(lon1)*len(lat1) == np.sum(chl1.mask)) & (len(lon2)*len(lat2) == np.sum(chl2.mask))) \
    | (len(lat1)*len(lon1)-sum(sum(chl1.mask))<2) | (len(lat2)*len(lon2)-sum(sum(chl2.mask))<2): 
        print('File empty: '+year+doy)
        fout.write('File empty: '+year+doy+'\n')
        return False, False, False
    else:
        #%% Create figure and subplot for chl w/o ANNOTS flags
        plt.figure(figsize=(12,12))
        plt.suptitle('O'+year+doy)
        plt.subplot(3, 2, 1)
        plot_map(chl1,lat1,lon1)   
        plt.title('chl w/o ANNOT flags')
        clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
        clb.ax.set_xlabel('chl')
        
        #%% subplot for chl w/ ANNOTS flags
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
        
        #%% Mask of the difference pixels
        plt.subplot(3, 2, 3)
        #plt.imshow(mask_diff, cmap='gray') 
        m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
            	llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='l')
        x,y=np.meshgrid(lon1, lat1)
        m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
        m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
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
        m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
        m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
        m.drawcoastlines(linewidth=0.1)
        m.imshow(chl_diff,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],\
                                                   norm=LogNorm(),cmap='rainbow')
        clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
        clb.ax.set_xlabel('chl')
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
        
        #%% histogram of the difference pixels
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
        
        ofname = 'O'+year+doy+'ANNOT_flag_'+site_name+'.pdf'
        ofname = os.path.join(path_out,ofname)
    
        plt.savefig(ofname, dpi=200)
    #    plt.show()
        plt.close()
        
        # Save netCDF4 file
    
        ofname = 'O'+year+doy+'_pxdiff_'+site_name+'.nc'
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
        path_in = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/DataArchive/'   
        path_out = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/data'        
    elif host == 'vm':
        path_in = '/DataArchive/OC/OLCI/daily/'
        path_out = '/home/Vittorio.Brando/Javier/data'
    else:
        print('Not sure from where this script will be run!')
    
    year_start = 2016
    year_end = 2016
    
    doy_start = 199
    doy_end = 199
    
    #%% To plot density  
    
    site_name = 'med'
#    site_name= 'bs'
    
    filename = '2016/199/O2016199--'+site_name+'-hr_brdf.nc' # open an original for to copy properties to output file
    
    nc_f0=Dataset(os.path.join(path_in,filename), 'r')
        
    lat0 = nc_f0.variables['lat'][:]
    lon0 = nc_f0.variables['lon'][:]
    
    ylen = len(lat0)
    xlen = len(lon0)
    
    fout = open(os.path.join(path_out,'output_'+site_name+'.txt'),'w+')
    #%%

    chl_diff_den = np.ma.zeros((ylen, xlen), dtype=np.float32) #   
    cover_sum = np.ma.zeros((ylen, xlen), dtype=np.float32) # total of observations
    
    count = 0;
    
    for year_idx in range(year_start, year_end+1):
        for doy_idx in range(doy_start, doy_end+1):
            year = str(year_idx)
            doy = str(doy_idx)
            
            if float(doy) < 100:
                if float(doy) < 10:
                    doy = '00'+doy
                else:
                    doy = '0'+doy
                
#            print(year+doy)    
            
            filename1 = year+'/'+doy+'/''O'+year+doy+'--'+site_name+'-hr_brdf.nc'
            filename2 = year+'/'+doy+'/''O'+year+doy+'--'+site_name+'-hr_brdf_w_ANNOT_DROUT.nc'
            
            if os.path.exists(path_in+filename1) & os.path.exists(path_in+filename2):
                if access(path_in+filename1, R_OK) & access(path_in+filename2, R_OK):
                    chl_diff_mask, coverage, valid_flag = plot_test_ANNOT_flags\
                    (path_in,path_out,filename1,filename2,year,doy,fout,site_name)
                    if valid_flag:   
                        print('File processing: '+year+doy)
                        fout.write('File processing: '+year+doy+'\n')
                        chl_diff_den = ~chl_diff_mask + chl_diff_den
                        
                        cover_sum = coverage + cover_sum
                        count = count + 1
                else:
                    print('File access denied: '+year+doy)                        
            else:
                print('File not found: '+year+doy)
                fout.write('File not found: '+year+doy+'\n')
    
    
    #%%
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='white')
    
    chl_diff_den.mask = chl_diff_den==0
    
    m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
        	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
    x,y=np.meshgrid(lon0, lat0)
    m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(chl_diff_den,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                               cmap='rainbow',interpolation='nearest')
    
    clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
    clb.ax.set_xlabel('Absolute Density (counts)')
    #%%
    plt.subplot(3,1,2)
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='white')
    
    cover_sum.mask = cover_sum==0
    
    m = Basemap(llcrnrlat=min(lat0),urcrnrlat=max(lat0),\
        	llcrnrlon=min(lon0),urcrnrlon=max(lon0), resolution='l')
    x,y=np.meshgrid(lon0, lat0)
    m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(cover_sum,origin='upper', extent=[min(lon0), max(lon0), min(lat0), max(lat0)],\
                                               cmap='rainbow',interpolation='nearest')
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
                                               cmap='rainbow',vmin=0, vmax=1,interpolation='nearest')
    
    clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
    clb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    clb.set_ticklabels(['0', '20', '40', '60', '80', '100'])
    clb.ax.set_xlabel('Occurence Percentage [%]')
    
    figname = os.path.join(path_out,'Density_'+site_name+'.pdf')
    #    print(figname)
    plt.savefig(figname, dpi=200)
    #plt.show()
    
    #%% Save netCDF4 file

    ofname = 'Density_'+site_name+'.nc'
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