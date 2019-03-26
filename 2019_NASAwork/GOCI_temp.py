#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:18:03 2019

@author: javier
"""

import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import os.path
import subprocess
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import time

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])
#%%
start = time.time()    
path_main = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/GOCI_ROI_STATS_R2018_vcal_aqua/'
#path_folder = 'G2016006061640.L2_COMS_BRDF7/G2016006061640.L2_COMS_BRDF7_valregion'
path_list = 'file_list_sort.txt'


filepath0 = os.path.join(path_main,path_list)

nlines = file_len(filepath0)
chlor_a_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
datetime_vec = np.zeros((nlines,), dtype=np.dtype(datetime)) # array with datetime

file0 = open(filepath0,'r')
count = 0

for line0 in file0:
    
    # time
    filepath1 = os.path.join(path_main,line0[2:-1])
    file = open(filepath1 , 'r') 
    for line in file:
        if line.split('=')[0] == 'time':
            datetime_vec[count] = datetime.strptime(line.split('=')[1][:-1],'%Y-%m-%d %H:%M:%S.%f')
        
    # chlor_a
    prod_name = 'chlor_a'
    filepath2 = os.path.join(path_main,line0[2:-1]+'.'+prod_name)
    file = open(filepath2 , 'r') 
    for line in file: 
        if line.split('=')[0] == 'filtered_mean':
            chlor_a_vec[count] = float(line.split('=')[1])
    count = count + 1

        
file0.close()
file.close()
end = time.time()
print(end-start)

#%%
start = time.time()  
path_main = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/GOCI_ROI_STATS_R2018_vcal_aqua/'
#path_folder = 'G2016006061640.L2_COMS_BRDF7/G2016006061640.L2_COMS_BRDF7_valregion'
path_list = 'file_list_sort.txt'


filepath0 = os.path.join(path_main,path_list)

nlines = file_len(filepath0)
chlor_a_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
datetime_vec = np.zeros((nlines,), dtype=np.dtype(datetime)) # array with datetime

file0 = open(filepath0,'r')

def extract_time(line0) :  
    # time
    filepath1 = os.path.join(path_main,line0[2:-1])
    file = open(filepath1 , 'r') 
    for line in file:
        if line.split('=')[0] == 'time':
            datetime_temp = datetime.strptime(line.split('=')[1][:-1],\
                        '%Y-%m-%d %H:%M:%S.%f')
    file.close()        
        
    # chlor_a
    prod_name = 'chlor_a'
    filepath2 = os.path.join(path_main,line0[2:-1]+'.'+prod_name)
    file = open(filepath2 , 'r') 
    for line in file: 
        if line.split('=')[0] == 'filtered_mean':
            chlor_a_temp = float(line.split('=')[1])
    file.close()
    
    return datetime_temp, chlor_a_temp

num_cores = multiprocessing.cpu_count()
     
results = Parallel(n_jobs=num_cores)(delayed(extract_time)(line0) for line0 in file0)

datetime_vec = [item[0] for item in results]
chlor_a_vec = [item[1] for item in results]
     
        
file0.close()

end = time.time()
print(end-start)

#%%
start = time.time()  
path_main = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/GOCI_ROI_STATS_R2018_vcal_aqua/'
#path_folder = 'G2016006061640.L2_COMS_BRDF7/G2016006061640.L2_COMS_BRDF7_valregion'
path_list = 'file_list_sort.txt'


filepath0 = os.path.join(path_main,path_list)

nlines = file_len(filepath0)
chlor_a_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
datetime_vec = np.zeros((nlines,), dtype=np.dtype(datetime)) # array with datetime

file0 = open(filepath0,'r')

def extract_time(line0) :  
    # time
    filepath1 = os.path.join(path_main,line0[2:-1])
    file = open(filepath1 , 'r') 
    for line in file:
        if line.split('=')[0] == 'time':
            datetime_temp = datetime.strptime(line.split('=')[1][:-1],\
                        '%Y-%m-%d %H:%M:%S.%f')
    file.close()        

    return datetime_temp

def extract_chlor_a(line0):      
        
    # chlor_a
    prod_name = 'chlor_a'
    filepath2 = os.path.join(path_main,line0[2:-1]+'.'+prod_name)
    file = open(filepath2 , 'r') 
    for line in file: 
        if line.split('=')[0] == 'filtered_mean':
            chlor_a_temp = float(line.split('=')[1])
    file.close()
    
    return chlor_a_temp

num_cores = multiprocessing.cpu_count()
     
datetime_vec = Parallel(n_jobs=num_cores)(delayed(extract_time)(line0) for line0 in file0)   
chlor_a_vec = Parallel(n_jobs=num_cores)(delayed(extract_chlor_a)(line0) for line0 in file0)     
        
file0.close()

end = time.time()
print(end-start)
#%% Plot
plt.figure(figsize=(12,8))
plt.plot(datetime_vec,chlor_a_vec,'*')
plt.ylim(0,0.5)
plt.show()

   