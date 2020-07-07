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
import subprocess

import matplotlib.patches
import shapely.geometry
import pandas as pd
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader

from descartes import PolygonPatch

def create_map():
    lat_min = 10
    lat_max = 60
    step_lat = 10
    lon_min = -30
    lon_max = 60
    step_lon = 10
    
    m = Basemap(llcrnrlat=lat_min,urcrnrlat=lat_max,\
    	llcrnrlon=lon_min,urcrnrlon=lon_max, resolution='l')
    m.drawparallels(range(lat_min, lat_max, step_lat),labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawmeridians(range(lon_min, lon_max, step_lon),labels=[1,0,0,1],color='grey',linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)
    m.fillcontinents(color='grey',lake_color='aqua')
    return m

def draw_polygon(lats,lons,m,sensor):
    x, y = m( lons, lats )
    xy = [(x[0],y[0]),(x[1],y[1]),(x[2],y[2]),(x[3],y[3])]
    if sensor == 'S3A':
        fc = 'red'
    elif sensor == 'S3B':
        fc = 'blue'
    poly = matplotlib.patches.Polygon( xy, facecolor=fc, alpha=0.4 ,closed=True, ec='k', lw=1,)
    plt.gca().add_patch(poly)
    return m


def create_list_products(path_source):
    cmd = f'find {path_source} -name "*OL_2_WFR*trim_MED*"> {path_source}/file_list.txt'
    prog = subprocess.Popen(cmd, shell=True,stderr=subprocess.PIPE)
    out, err = prog.communicate()
    if err:
        print(err)   
        
def create_csv(path_to_list,df,m):
    with open(path_to_list,'r') as file:
        for cnt, line in enumerate(file):   
            path_im = line[:-1]
            coordinates_filename = 'geo_coordinates.nc'
            filepah = os.path.join(path_im,coordinates_filename)
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
            
            create_list_products(path_source)
            
            #%% create csv
            
            
            sensor = path_im.split('/')[-1].split('_')[0]
            datetimestr = path_im.split('/')[-1].split('_')[7]
            date =  datetimestr.split('T')[0]
            time = datetimestr.split('T')[1]
            doy = path_im.split('/')[-2]
            filename = path_im.split('/')[-1]
            
            granule = {
                    'sensor': sensor,
                    'datetimestr': datetimestr,
                    'date': date,
                    'time': time,
                    'doy': doy,
                    'UL_lat': UL_lat,
                    'UL_lon': UL_lon,
                    'UR_lat': UR_lat,
                    'UR_lon': UR_lon,
                    'LL_lat': LL_lat,
                    'LL_lon': LL_lon,
                    'LR_lat': LR_lat,
                    'LR_lon': LR_lon,
                    'filename': filename,
                    'filepah':   path_im
                    }
            
            
            df = df.append(granule,ignore_index=True) 
            # draw map
            m = draw_polygon(lats_poly,lons_poly,m,sensor)
            plt.gcf()
            plt.title(date)
            custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='blue', lw=4)]

            plt.legend(custom_lines, ['S3A', 'S3B'],loc='upper left')
            
    return df, date       
#%%
# def main():
# plot_footprint(var,lat,lon)
cols = ['sensor','datetimestr','date','time','doy','UL_lat','UL_lon','UR_lat','UR_lon','LL_lat','LL_lon','LR_lat','LR_lon','filename','filepah']
df = pd.DataFrame(columns = cols)        
path_source = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Images/OLCI/trimmed_sources'
path_to_list = os.path.join(path_source,'file_list.txt')  
path_out = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Call_ESA_MED/Figures'

m =create_map()
df,date = create_csv(path_to_list,df,m)  

# save figure
plt.gcf()
ofname = os.path.join(path_out,date+'.pdf')
plt.savefig(ofname, dpi=200)
print(df)

#%%

