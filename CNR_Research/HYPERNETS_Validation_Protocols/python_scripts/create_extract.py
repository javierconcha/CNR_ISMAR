#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:41:12 2019

@author: javier
"""
import os
#%%
host = 'vm'

if host == 'vm':
    path_source = '/DataArchive/OC/OLCI/sources_baseline_2.23/'
    
elif host == 'mac':
    
    path_main = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts'
    path_source = os.path.join(path_main,'source')
    listname = 'OLCI_list_uniq.txt '
    path_to_list = os.path.join(path_main,listname)


with open(path_to_list,'r') as file:
    line = file.readline()
    print(line)