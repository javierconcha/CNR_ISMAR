#%% Initializaion
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import os.path
import sys  
sys.path.append('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/')   
from OLCI_NAS_ANNOT_flags_testing import plot_map

from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
import os
os.environ['QT_QPA_PLATFORM']='offscreen' # to avoid error "QXcbConnection: Could not connect to display"
#%%
path_in = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/DataArchive/OLCI_NAS/20160426_20190228'

filename = 'VGOCS_2016290095405_O.nc' # open an original for to copy properties to output file
    
nc_f1=Dataset(os.path.join(path_in,filename), 'r')

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

meridian_steps = [12.5, 13, 13.5]
parallel_steps = [44, 44.5, 45, 45.5]

#%% Create figure and subplot for chl w/o ANNOTS flags
plt.figure(figsize=(12,12))
plt.subplot(3, 2, 1)
plot_map(chl1,lat1,lon1,meridian_steps,parallel_steps) 
plt.title('chl w/o ANNOT flags')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
#        clb.ax.set_xlabel('chl')

#%% subplot for chl w/ ANNOTS flags
plt.subplot(3, 2, 2)
plot_map(chl2,lat2,lon2,meridian_steps,parallel_steps)  
plt.title('chl w/ ANNOT flags')
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
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
clb = plt.colorbar(fraction=0.046, pad=0.1,orientation='horizontal')
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



plt.tight_layout()
plt.show()
#plt.close()