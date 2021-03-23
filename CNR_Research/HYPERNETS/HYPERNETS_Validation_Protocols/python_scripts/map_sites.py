#!/usr/bin/env python3
# coding: utf-8
"""
Created on Mon Oct 12 19:25:31 2020

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
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.lines as mlines

import common_functions as cfs
#%%
station_list_S3 = ['Venise','Galata_Platform','Gloria','Helsinki_Lighthouse','Gustav_Dalen_Tower']
station_list_S2 = ['Galata_Platform','Gustav_Dalen_Tower','Helsinki_Lighthouse','Lake_Erie',\
                'LISCO','Palgrunden','Thornton_C-power','USC_SEAPRISM_2',\
                'Venise','WaveCIS_Site_CSI_6','Gloria']
station_n = {'Venise':'Venise','Galata_Platform':'Galata Platform','Gloria':'Gloria','Helsinki_Lighthouse':'Helsinki_Lighthouse','Gustav_Dalen_Tower':'Gustav Dalen Tower',\
             'Palgrunden':6,'Thornton_C-power':7,'LISCO':8,'Lake_Erie':9,'WaveCIS_Site_CSI_6':10,\
                 'USC_SEAPRISM_2':11}
station_label_loc = {'Venise':[1.01,1.06],'Galata_Platform':[1.01,0.90],'Gloria':[1.01,1.01],'Helsinki_Lighthouse':[1.01,1.01],'Gustav_Dalen_Tower':[1.01,1.01],\
             'Palgrunden':[0.98,1.01],'Thornton_C-power':[1.01,1.01],'LISCO':[1.02,1.01],'Lake_Erie':[1.04,1.01],'WaveCIS_Site_CSI_6':[1.04,1.1],\
                 'USC_SEAPRISM_2':[1.7,1.01]}
    
#%% legend
ms = 16
lenged_Venise               = mlines.Line2D([], [], ls='None', ms=ms, marker='o', c='b',mec='r', label=f'{station_n["Venise"]} Venise')
lenged_Galata_Platform      = mlines.Line2D([], [], ls='None', ms=ms, marker='o', c='b',mec='r', label=f'{station_n["Galata_Platform"]} Galata Platform')
lenged_Gloria               = mlines.Line2D([], [], ls='None', ms=ms, marker='o', c='b',mec='r', label=f'{station_n["Gloria"]} Gloria')
lenged_Helsinki_Lighthouse  = mlines.Line2D([], [], ls='None', ms=ms, marker='o', c='b',mec='r', label=f'{station_n["Helsinki_Lighthouse"]} Helsinki Lighthouse')
lenged_Gustav_Dalen_Tower   = mlines.Line2D([], [], ls='None', ms=ms, marker='o', c='b',mec='r', label=f'{station_n["Gustav_Dalen_Tower"]} Gustav Dalen Tower')
# lenged_Palgrunden           = mlines.Line2D([], [], ls='None', ms=8, marker='o', c='b',mec='b', label=f'{station_n["Palgrunden"]} Palgrunden')
# lenged_Thornton_C_power     = mlines.Line2D([], [], ls='None', ms=8, marker='o', c='b',mec='b', label=f'{station_n["Thornton_C-power"]} Thornton C-power')
# lenged_LISCO                = mlines.Line2D([], [], ls='None', ms=8, marker='o', c='b',mec='b', label=f'{station_n["LISCO"]} LISCO')
# lenged_Lake_Erie            = mlines.Line2D([], [], ls='None', ms=8, marker='o', c='b',mec='b', label=f'{station_n["Lake_Erie"]} Lake Erie')
# lenged_WaveCIS_Site_CSI_6   = mlines.Line2D([], [], ls='None', ms=8, marker='o', c='b',mec='b', label=f'{station_n["WaveCIS_Site_CSI_6"]} WaveCIS Site CSI 6')
# # lenged_USC_SEAPRISM         = mlines.Line2D([], [], ls='None', ms=8, marker='o', c='b',mec='b', label=f'{station_n["USC_SEAPRISM"]} USC SEAPRISM')
# lenged_USC_SEAPRISM_2       = mlines.Line2D([], [], ls='None', ms=8, marker='o', c='b',mec='b', label=f'{station_n["USC_SEAPRISM_2"]} USC SEAPRISM 2')


      
#%%
lat_plot_limits = [38,62] 
lon_plot_limits = [0,40]
map_kwargs = dict(projection='merc', resolution=None,
                  rsphere=(6378137.00,6356752.3142),\
                          lat_0=50.,lon_0=20.,lat_ts=30.,
                      llcrnrlat=lat_plot_limits[0], urcrnrlat=lat_plot_limits[1],
                      llcrnrlon=lon_plot_limits[0], urcrnrlon=lon_plot_limits[1])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

# leg1 = ax.legend(handles=[lenged_Venise,lenged_Galata_Platform,lenged_Gloria,lenged_Helsinki_Lighthouse,\
#                           lenged_Gustav_Dalen_Tower,lenged_Palgrunden,lenged_Thornton_C_power,lenged_LISCO,\
#                           lenged_Lake_Erie,lenged_WaveCIS_Site_CSI_6,lenged_USC_SEAPRISM_2],\
#                 loc='upper center',bbox_to_anchor=(0.5, 1.35),ncol=3,frameon=False,fontsize=10)    
# leg1 = ax.legend(handles=[lenged_Venise,lenged_Galata_Platform,lenged_Gloria,lenged_Helsinki_Lighthouse,\
#                           lenged_Gustav_Dalen_Tower],\
#                 loc='upper center',bbox_to_anchor=(0.5, 1.35),ncol=3,frameon=False,fontsize=10) 

m = Basemap(**map_kwargs)


meridianinterval = np.linspace(lon_plot_limits[0],lon_plot_limits[1],5) # 5 = number of "ticks"
m.drawmeridians(meridianinterval,labels=[0,0,0,1])
parallelinterval = np.linspace(lat_plot_limits[0]+5,lat_plot_limits[1]-5,5) # 5 = number of "ticks"
m.drawparallels(parallelinterval,labels=[1,0,0,0])    
# m.drawcoastlines()
m.shadedrelief()

# for station in station_list_S2:
#     lat0,lon0 = cfs.get_lat_lon_ins(station)
#     x,y = m(lon0,lat0)
#     m.plot(x,y,'bo')
#     xmul,ymul = station_label_loc[station]
#     plt.text(x*xmul,y*ymul,station_n[station])    
    
for station in station_list_S3:
    lat0,lon0 = cfs.get_lat_lon_ins(station)
    x,y = m(lon0,lat0)
    m.plot(x,y,'rx',markersize=14,markeredgewidth=4)
    xmul,ymul = station_label_loc[station] 
    # plt.text(x*xmul,y*ymul,station_n[station],fontsize=12,backgroundcolor='0.75')

 
