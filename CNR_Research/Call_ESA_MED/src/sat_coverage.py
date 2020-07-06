#!/usr/bin/env python3
# coding: utf-8
"""
Created on Mon Jul  6 12:33:11 2020

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
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import os.path
import os

from matplotlib.patches import Polygon

def create_map():
    m = Basemap(llcrnrlat=20,urcrnrlat=45,\
    	llcrnrlon=-15,urcrnrlon=35, resolution='l')
    m.drawparallels([30, 35, 40, 45],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians([-5, 0, 5, 10, 15, 20, 25, 30, 35],labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.fillcontinents(color='grey',lake_color='aqua')
    return m

def draw_polygon(lats,lons,m):
    x, y = m( lons, lats )
    xy = [(x[0],y[0]),(x[1],y[1]),(x[2],y[2]),(x[3],y[3])]
    poly = Polygon( xy, facecolor='red', alpha=0.4 )
    plt.gca().add_patch(poly)
        
#%%
# def main():
# plot_footprint(var,lat,lon)
path_source = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Images/OLCI/trimmed_sources/'
path_im = '2020/153/S3B_OL_2_WFR____20200601T103303_20200601T103603_20200602T171214__trim_MED_039_279_MAR_O_NT_002.SEN3'
coordinates_filename = 'geo_coordinates.nc'

filepah = os.path.join(path_source,path_im,coordinates_filename)
nc_f0 = Dataset(filepah,'r')

lat = nc_f0.variables['latitude'][:,:]
lon = nc_f0.variables['longitude'][:,:]

UL_lat = lat[0,0]
UL_lon = lon[0,0]
UR_lat = lat[0,-1]
UR_lon = lon[0,-1]
LL_lat = lat[-1,0]
LL_lon = lon[-1,0]
LR_lat = lat[-1,-1]
LR_lon = lon[-1,-1]

lats_poly =[UL_lat,UR_lat,LR_lat,LL_lat]
lons_poly =[UL_lon,UR_lon,LR_lon,LL_lon]

m =create_map()

draw_polygon(lats_poly,lons_poly,m)

print(UL_lat)
print(UL_lon)
print(UR_lat)
print(UR_lon)
print(LL_lat)
print(LL_lon)
print(LR_lat)
print(LR_lon)
           