#%% Initializaion
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import matplotlib
import os.path
import sys  
sys.path.append('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/')   
from OLCI_NAS_ANNOT_flags_testing import plot_map
sys.path.append('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Python_examples/') 
from plot_map_hres import plot_mask

from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
import os
os.environ['QT_QPA_PLATFORM']='offscreen' # to avoid error "QXcbConnection: Could not connect to display"

#% shape files and higher res for coastlines
from shapely.geometry import Polygon
import shapely.speedups
shapely.speedups.enable()



import geopandas as gpd

#from time import sleep
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
#%%
#sh0 = gpd.read_file('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Shapefiles/ISPRA/Bacini_idrografici_principali_0607/Bacini_idrografici_principali_0607.shp')
#sh0 = sh0.to_crs(epsg=4326)
#rivers0=list(sh0['geometry'])[:]
#
#sh2 = gpd.read_file('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Shapefiles/GSHHS_shp/h/GSHHS_h_L1.shp')
#borders=list(sh2['geometry']) #polygon
#Europa_shape = borders[0]

coords=[(12.,44.), (12.,46.), (14.,46.), (14.,44.)]
#poly_NAD=Polygon(coords) # POLYGON ((12 44, 12 46, 14 46, 14 44, 12 44))
#%%
plt.figure(figsize=(8,8))
#m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
#			llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='f')
#
#m.drawparallels(parallel_steps,labels=[1,0,0,1])
#m.drawmeridians(meridian_steps,labels=[1,0,0,1])
##m.drawrivers(linewidth=1.0, color='royalblue')
#
#xx,yy = Europa_shape.exterior.coords.xy 
#xy = np.array(list(zip(xx,yy))) # (N,2) numpy array
#    
#if Europa_shape.intersects(poly_NAD)==True:
#    m.plot(xx,yy,marker=None, color='black', linewidth=0.5)
#    poly = matplotlib.patches.Polygon(xy, facecolor='#EED5B7')
#    plt.gca().add_patch((poly))
#
#for (r,i) in zip(rivers0, range(len(rivers0))):
#    if i!=3 and i!=0 and i!=1:
#        elements=list(r.exterior.coords)
#        xx,yy=zip(*elements)
#        m.plot(xx,yy,marker=None, color='royalblue', linewidth=1)
#%
#%

m = plot_mask(lat1,lon1,coords,meridian_steps,parallel_steps)

cs=m.imshow(chl1,origin='upper', extent=[min(lon1), max(lon1), min(lat1), max(lat1)],cmap=plt.cm.Spectral_r)
#plt.axes().set_aspect('equal')

path_out = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp'
figname = os.path.join(path_out,'test.pdf')
#    print(figname)
plt.savefig(figname, dpi=200)
plt.show()
plt.close()