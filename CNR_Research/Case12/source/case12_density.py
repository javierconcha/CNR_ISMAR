#!/usr/bin/env python3
# coding: utf-8
"""
Created on Tue Dec  3 14:39:03 2019

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
os.environ['QT_QPA_PLATFORM']='offscreen' # to avoid error "QXcbConnection: Could not connect to display"
import sys
#%% function to plot different products in a map
def plot_map(var,lat,lon):
    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
    	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
    x,y=np.meshgrid(lon, lat)
    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.imshow(var, extent=[min(lon), max(lon), min(lat), max(lat)],cmap='rainbow')

    
def plot_WTM(file_path):
    # Example of file info:
	# dimensions(sizes): time(1), lat(1580), lon(3308)
 	# ariables(dimensions): int32 time(time), float32 lat(lat), float32 lon(lon), 
 	# float32 CHL(time,lat,lon), float32 WTM(time,lat,lon), 
 	# int32 SENSORMASK(time,lat,lon), float32 QI(time,lat,lon)

    nc_f0 = Dataset(file_path,'r')
    lat = nc_f0.variables['lat'][:]
    lon = nc_f0.variables['lon'][:]
    WTM = nc_f0.variables['WTM'][0,:,:]
    nc_f0.close()    

    if sys.platform == 'darwin':
	    plt.figure(figsize=(12,16))
	    plt.suptitle(file_path.split('/')[-1])
  
    # Case 1
    arr_case1 = (WTM.data == 1).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 1)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case1, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('Case 1')

    # Case 2
    arr_case2 = (WTM.data == 0).astype(int)
    
    if sys.platform == 'darwin':
        plt.subplot(5, 3, 2)
        m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
        x,y=np.meshgrid(lon, lat)
        m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
        m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
        m.drawcoastlines(linewidth=0.1)
        m.imshow(arr_case2, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
        plt.title('Case 2')  

    # Valid Mask
    arr_valid = (WTM.mask==False).astype(int)

    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 3)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_valid, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('Valid Pixels')         

    # 0.0 < case <= 0.1
    arr_case01 = ((WTM.data >= 0) & (WTM.data <= 0.1)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 4)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case01, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.0 <= case <= 0.1') 

    # 0.1 < case <= 0.2
    arr_case02 = ((WTM.data > 0.1) & (WTM.data <= 0.2)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 5)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case02, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.1 < case <= 0.2') 

    # 0.2 < case <= 0.3
    arr_case03 = ((WTM.data > 0.2) & (WTM.data <= 0.3)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 6)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case03, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.2 < case <= 0.3') 

    # 0.3 < case <= 0.4
    arr_case04 = ((WTM.data > 0.3) & (WTM.data <= 0.4)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 7)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case04, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.3 < case <= 0.4') 

    # 0.4 < case <= 0.5
    arr_case05 = ((WTM.data > 0.4) & (WTM.data <= 0.5)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 8)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case05, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.4 < case <= 0.5') 

    # 0.5 < case <= 0.6
    arr_case06 = ((WTM.data > 0.5) & (WTM.data <= 0.6)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 9)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case06, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.5 < case <= 0.6') 

    # 0.6 < case <= 0.7
    arr_case07 = ((WTM.data > 0.6) & (WTM.data <= 0.7)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 10)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case07, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.6 < case <= 0.7') 

    # 0.7 < case <= 0.8
    arr_case08 = ((WTM.data > 0.7) & (WTM.data <= 0.8)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 11)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case08, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.7 < case <= 0.8') 

	    # 0.8 < case <= 0.9
	    arr_case09 = ((WTM.data > 0.8) & (WTM.data <= 0.9)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 12)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case09, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.8 < case <= 0.9') 

    # 0.9 < case <= 1.0
    arr_case10 = ((WTM.data > 0.9) & (WTM.data <= 1.0)).astype(int)
    
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 13)
	    m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	        	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='l')
	    x,y=np.meshgrid(lon, lat)
	    m.drawparallels([30, 35, 40, 45],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[0,0,0,0],color='grey',linewidth=0.1)
	    m.drawcoastlines(linewidth=0.1)
	    m.imshow(arr_case10, extent=[min(lon), max(lon), min(lat), max(lat)],\
	                                               cmap='gray',interpolation='nearest')
	    plt.title('0.9 < case <= 1.0')     
    
    # WTM
    if sys.platform == 'darwin':
	    plt.subplot(5, 3, 14)

	    plot_map(WTM,lat,lon)  
	    plt.title('WTM')
	    clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
	    clb.ax.set_xlabel('WTM=1: Case 1, 0: Case 2, between the two mixture')

	    # clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
	    # clb.ax.set_xlabel('0: invalid | 1: valid')
  
    return arr_case1, arr_case2, arr_case01, arr_case02, arr_case03, \
    	arr_case04, arr_case05, arr_case06, arr_case07, arr_case08, \
    	arr_case09, arr_case10, arr_valid
#%%
def main():
    """business logic for when running this module as the primary one!"""
    print('Main Code!')
    
    site_name = 'med'
    
    if sys.platform == 'linux': 
        path_main = '/home/Javier.Concha/Case12/'
        # path_data = '/store2/OC/X/daily_v4/'
        path_data = '/store2/OC/X/daily_v201912/'
        path_out = os.path.join(path_main,'output')
        list_name = 'file_list.txt' #'OLCI_list_uniq.txt '
        path_to_list = os.path.join(path_main,'data',list_name)       
    elif sys.platform == 'darwin':
        path_main = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Case12/'
        path_data = os.path.join(path_main,'data')
        path_out = os.path.join(path_main,'output')
        list_name = 'file_list.txt' #'OLCI_list_uniq.txt '
        path_to_list = os.path.join(path_data,list_name)
    else:
        print('Error: host flag is not either mac or vm')

    with open(path_to_list,'r') as file:
        for cnt, line in enumerate(file):
            if cnt == 0:
                file_example_path = os.path.join(path_data,line[:-1])
                nc_f0 = Dataset(file_example_path, 'r')
                lat0 = nc_f0.variables['lat'][:]
                lon0 = nc_f0.variables['lon'][:]
                nc_f0.close()
                                
                ylen = len(lat0)
                xlen = len(lon0)
                
#                site_name = 'med'
#                fout = open(os.path.join(path_out,'output_'+site_name+'.txt'),'w+')   
                case1_counts 	= np.zeros((ylen, xlen), dtype=np.float32) # 
                case2_counts 	= np.zeros((ylen, xlen), dtype=np.float32) # 
                case01_counts 	= np.zeros((ylen, xlen), dtype=np.float32) # 
                case02_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case03_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case04_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case05_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case06_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case07_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case08_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case09_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                case10_counts 	= np.zeros((ylen, xlen), dtype=np.float32) #
                valid_counts 	= np.zeros((ylen, xlen), dtype=np.float32) # 
    
            print('------------------')
            file_path = os.path.join(path_data,line[:-1])
            print(file_path)
            arr_case1,  arr_case2, arr_case01, arr_case02, arr_case03, arr_case04,\
            arr_case05, arr_case06, arr_case07, arr_case08, arr_case09, arr_case10,\
            arr_valid = plot_WTM(file_path)
            case1_counts 	= arr_case1 + case1_counts
            case2_counts 	= arr_case2 + case2_counts
            case01_counts 	= arr_case01 + case01_counts
            case02_counts 	= arr_case02 + case02_counts
            case03_counts 	= arr_case03 + case03_counts
            case04_counts 	= arr_case04 + case04_counts
            case05_counts 	= arr_case05 + case05_counts
            case06_counts 	= arr_case06 + case06_counts
            case07_counts 	= arr_case07 + case07_counts
            case08_counts 	= arr_case08 + case08_counts
            case09_counts 	= arr_case09 + case09_counts
            case10_counts 	= arr_case10 + case10_counts
            valid_counts 	= arr_valid + valid_counts




	#%% Save netCDF4 file
    ofname = 'Case12_'+site_name+'.nc'
    ofname = os.path.join(path_out,ofname)
    if os.path.exists(ofname):
        os.remove(ofname)
        print('File removed!')
        
    fmb = Dataset(ofname, 'w', format='NETCDF4')
    fmb.description = 'Case 1 and Case 2 algorithm performance'
    
    fmb.createDimension("lat", len(lat0))
    fmb.createDimension("lon", len(lon0))
    
    lat = fmb.createVariable('lat',  'single', ('lat',)) 
    lon = fmb.createVariable('lon',  'single', ('lon',))
    
    lat[:] = lat0
    lon[:] = lon0
    
    gridd_var1=fmb.createVariable('case1_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var1[:,:] = case1_counts 

    gridd_var2=fmb.createVariable('case2_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var2[:,:] = case2_counts 

    gridd_var3=fmb.createVariable('case01_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var3[:,:] = case01_counts

    gridd_var4=fmb.createVariable('case02_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var4[:,:] = case02_counts

    gridd_var5=fmb.createVariable('case03_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var5[:,:] = case03_counts

    gridd_var6=fmb.createVariable('case04_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var6[:,:] = case04_counts

    gridd_var7=fmb.createVariable('case05_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var7[:,:] = case05_counts

    gridd_var8=fmb.createVariable('case06_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var8[:,:] = case06_counts

    gridd_var9=fmb.createVariable('case07_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var9[:,:] = case07_counts

    gridd_var10=fmb.createVariable('case08_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var10[:,:] = case08_counts

    gridd_var11=fmb.createVariable('case09_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var11[:,:] = case09_counts

    gridd_var12=fmb.createVariable('case10_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var12[:,:] = case10_counts

    gridd_var13=fmb.createVariable('valid_counts', 'single', ('lat', 'lon',), zlib=True, complevel=6)
    gridd_var13[:,:] = valid_counts 


    
    fmb.close()

#%%
if __name__ == '__main__':
    main()   