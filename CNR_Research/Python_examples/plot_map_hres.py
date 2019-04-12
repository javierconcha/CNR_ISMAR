#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:58:52 2019

@author: javier
"""

#%% Initializaion
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon
import shapely.speedups
shapely.speedups.enable()

import geopandas as gpd

def plot_mask(lat1,lon1,coords,meridian_steps,parallel_steps,rivers_flag):

    #%%
    sh0 = gpd.read_file('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Shapefiles/ISPRA/Bacini_idrografici_principali_0607/Bacini_idrografici_principali_0607.shp')
    sh0 = sh0.to_crs(epsg=4326)
    rivers0=list(sh0['geometry'])[:]
    
    sh2 = gpd.read_file('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Shapefiles/GSHHS_shp/h/GSHHS_h_L1.shp')
    borders=list(sh2['geometry']) #polygon
    Europa_shape = borders[0]
    Africa_shape = borders[2]
    
    poly_NAD=Polygon(coords) # POLYGON ((12 44, 12 46, 14 46, 14 44, 12 44))
    #%%

    m = Basemap(llcrnrlat=min(lat1),urcrnrlat=max(lat1),\
    			llcrnrlon=min(lon1),urcrnrlon=max(lon1), resolution='f')
    
    m.drawparallels(parallel_steps,labels=[1,0,0,1])
    m.drawmeridians(meridian_steps,labels=[1,0,0,1])
    
    xx,yy = Europa_shape.exterior.coords.xy 
    xy = np.array(list(zip(xx,yy))) # (N,2) numpy array
        
    if Europa_shape.intersects(poly_NAD)==True:
        m.plot(xx,yy,marker=None, color='black', linewidth=0.2)
        poly = matplotlib.patches.Polygon(xy, facecolor='#EED5B7')
        plt.gca().add_patch((poly))
        
        
    xx,yy = Africa_shape.exterior.coords.xy 
    xy = np.array(list(zip(xx,yy))) # (N,2) numpy array    
    if Africa_shape.intersects(poly_NAD)==True:
        m.plot(xx,yy,marker=None, color='black', linewidth=0.2)
        poly = matplotlib.patches.Polygon(xy, facecolor='#EED5B7')
        plt.gca().add_patch((poly))
    
    if rivers_flag:
        for (r,i) in zip(rivers0, range(len(rivers0))):
            if i!=3 and i!=0 and i!=1:
                elements=list(r.exterior.coords)
                xx,yy=zip(*elements)
                m.plot(xx,yy,marker=None, color='royalblue', linewidth=0.1)
            
    return m