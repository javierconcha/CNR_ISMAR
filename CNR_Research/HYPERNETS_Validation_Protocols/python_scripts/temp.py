#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:18:55 2019

@author: javier.concha
"""
#%%
import datetime
import olci_getscenes

path_main = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'
#%%

date1 = datetime.datetime(2018, 12, 30)
date2 = datetime.datetime(2018, 12, 31)
date3 = datetime.datetime(2017, 12, 30)

Latitude=43.044624
Longitude=28.193190

lat1 = Latitude-0.5
lat2 = Latitude+0.5

lon1 = Longitude-0.5
lon2 = Longitude+0.5

file_list = olci_getscenes.olci_get(date1,lat1,lat2,lon1,lon2)
print('file_list:')
print(file_list)
#
#file_list = olci_getscenes.olci_get(date2,lat1,lat2,lon1,lon2)
#print('file_list:')
#print(file_list)

#file_list = olci_getscenes.olci_get(date3,lat1,lat2,lon1,lon2)
#print('file_list:')
#print(file_list)

f = open(path_main+'OLCI_list_from_HUB.txt','a+')

if file_list:
    for r in file_list:
        f.write(r+'\n')

f.close()        
    
    
    