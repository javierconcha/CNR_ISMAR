#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:13:41 2019

@author: javier
"""
import os
import subprocess
#%%
path_main = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts'
path_source = os.path.join(path_main,'data/source')
listname = 'OLCI_list_test.txt' #'OLCI_list_uniq.txt '
path_to_list = os.path.join(path_main,listname)


with open(path_to_list,'r') as file:
    for cnt, line in enumerate(file):
        print('------------------')
        server_path = 'Javier.Concha@artov.ismar.cnr.it@192.168.10.79:/DataArchive/OC/OLCI/sources_baseline_2.23/'     
        line  = '2016/126/S3A_OL_2_WFR____20160505T090625_20160505T090825_20171031T074650_0119_003_378______MR1_R_NT_002'
        ext = '.SEN3.zip .'
        cmd = 'scp '+server_path+line+ext
        print(cmd)
        (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
        print(ls_status)
        print(ls_output)