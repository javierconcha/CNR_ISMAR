#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:18:03 2019

@author: javier
"""

import numpy as np
#from netCDF4 import Dataset
from matplotlib import pyplot as plt
import os.path
import subprocess
from datetime import datetime
#from joblib import Parallel, delayed
#import multiprocessing
import time

import math

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def extract_data(line0,path_main,prod_name,par_name): 
    
#0 name=chlor_a
#1 group_name=geophysical_data
#2 center_value=0.116951
#3 valid_pixel_count=151718
#4 max=377.876709
#5 min=0.001000
#6 mean=0.174360
#7 median=0.143130
#8 stddev=1.964564
#9 rms=1.972280
#10 filtered_valid_pixel_count=151619
#11 filtered_max=3.037774
#12 filtered_min=0.001000
#13 filtered_mean=0.150334
#14 filtered_median=0.143090
#15 filtered_stddev=0.070266
#16 filtered_rms=0.165945
#17 iqr_valid_pixel_count=75860
#18 iqr_max=0.172553
#19 iqr_min=0.116392
#20 iqr_mean=0.143222
#21 iqr_median=0.143130
#22 iqr_stddev=0.015468
#23 iqr_rms=0.144055
# cond_area = [GOCI_Data.Rrs_555_filtered_valid_pixel_count]>= total_px_GOCI/ratio_from_the_total;    
    
    if par_name == 'filtered_max':
        line_num = 11
    elif par_name == 'filtered_min':
        line_num = 12
    elif par_name == 'filtered_mean':
        line_num = 13
    elif par_name == 'filtered_median':
        line_num = 14
    elif par_name == 'filtered_stddev':
        line_num = 15 
    elif par_name == 'center_value':
        line_num = 2 
    elif par_name == 'filtered_valid_pixel_count':
        line_num = 10        
        
    # time
    if prod_name == 'main_prod':
        with open(path_main+line0, 'r') as file:
            line = file.readline()
            datetime_temp = datetime.strptime(line.split('=')[1][:-1],\
                            '%Y-%m-%d %H:%M:%S.%f')
#            print(datetime_temp)
        return datetime_temp     
    else:
        prod_temp = np.zeros((1,), dtype=np.float32)    
        with open(path_main+line0+'.'+prod_name, 'r') as file:
            line = file.readlines()[line_num]          
            prod_temp = float(line.split('=')[1])
        return prod_temp

#%%
    
def main():
#    start = time.time()  
    path_main = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/GOCI_ROI_STATS_R2018_vcal_aqua/'
    #path_folder = 'G2016006061640.L2_COMS_BRDF7/G2016006061640.L2_COMS_BRDF7_valregion'
    path_list = 'file_list_short.txt'
#    print(path_list)
    
    
    filepath0 = os.path.join(path_main,path_list)
#    print(filepath0)
    nlines = None 
    nlines = file_len(filepath0)
#    print(nlines)
    chlor_a_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
    chlor_a_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
    chlor_a_filtered_mean_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
    # chlor_a_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
    # chlor_a_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
#    chlor_a_CV_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values

    Rrs_412_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_412 values
    Rrs_412_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_412 values
    Rrs_412_filtered_mean_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_412 values
    Rrs_412_CV_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_412 values

    Rrs_443_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_443 values
    Rrs_443_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_443 values
    Rrs_443_filtered_mean_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_443 values
    Rrs_443_CV_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_443 values    

    Rrs_490_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_490 values
    Rrs_490_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_490 values
    Rrs_490_filtered_mean_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_490 values
    Rrs_490_CV_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_490 values

    Rrs_555_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_555 values
    Rrs_555_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_555 values
    Rrs_555_filtered_mean_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_555 values 
    Rrs_555_CV_vec = np.zeros((nlines,), dtype=np.float32) # array with Rrs_555 values
    
    aot_865_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with aot_865 values
    aot_865_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with aot_865 values
    aot_865_filtered_mean_vec = np.zeros((nlines,), dtype=np.float32) # array with aot_865 values   
    aot_865_CV_vec = np.zeros((nlines,), dtype=np.float32) # array with aot_865 values  

    median_CV_vec = np.zeros((nlines,), dtype=np.float32) # array with median_CV values   
    
    datetime_vec = np.zeros((nlines,), dtype=np.dtype(datetime)) # array with datetime
#    print(len(chlor_a_filtered_mean_vec))
    
    file0 = open(filepath0,'r')
    
    #%% Sequencial
    for idx, line in enumerate(file0):
        datetime_vec[idx] = extract_data(line[2:-1],path_main,'main_prod',None)
      
        chlor_a_filtered_mean_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_mean')
        chlor_a_filtered_median_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_median')
        chlor_a_filtered_stddev_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_stddev')
#        chlor_a_filtered_max_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_max')
#        chlor_a_filtered_min_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_min')
        # chlor_a_CV_vec = chlor_a_filtered_stddev_vec[idx]/chlor_a_filtered_mean_vec[idx]

        Rrs_412_filtered_mean_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_412','filtered_mean')
        Rrs_412_filtered_median_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_412','filtered_median')
        Rrs_412_filtered_stddev_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_412','filtered_stddev')
        Rrs_412_CV_vec[idx] = Rrs_412_filtered_stddev_vec[idx]/Rrs_412_filtered_mean_vec[idx]
        
        Rrs_443_filtered_mean_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_443','filtered_mean')
        Rrs_443_filtered_median_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_443','filtered_median')
        Rrs_443_filtered_stddev_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_443','filtered_stddev') 
        Rrs_443_CV_vec[idx] = Rrs_443_filtered_stddev_vec[idx]/Rrs_443_filtered_mean_vec[idx]

        Rrs_490_filtered_mean_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_490','filtered_mean')
        Rrs_490_filtered_median_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_490','filtered_median')
        Rrs_490_filtered_stddev_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_490','filtered_stddev') 
        Rrs_490_CV_vec[idx] = Rrs_490_filtered_stddev_vec[idx]/Rrs_490_filtered_mean_vec[idx]

        Rrs_555_filtered_mean_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_555','filtered_mean')
        Rrs_555_filtered_median_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_555','filtered_median')
        Rrs_555_filtered_stddev_vec[idx] = extract_data(line[2:-1],path_main,'Rrs_555','filtered_stddev') 
        Rrs_555_CV_vec[idx] = Rrs_555_filtered_stddev_vec[idx]/Rrs_555_filtered_mean_vec[idx]
        
        aot_865_filtered_mean_vec[idx] = extract_data(line[2:-1],path_main,'aot_865','filtered_mean')
        aot_865_filtered_median_vec[idx] = extract_data(line[2:-1],path_main,'aot_865','filtered_median')
        aot_865_filtered_stddev_vec[idx] = extract_data(line[2:-1],path_main,'aot_865','filtered_stddev') 
        aot_865_CV_vec[idx] = aot_865_filtered_stddev_vec[idx]/aot_865_filtered_mean_vec[idx]

        median_CV_vec[idx] = np.median([Rrs_412_CV_vec[idx],Rrs_443_CV_vec[idx],Rrs_490_CV_vec[idx],Rrs_555_CV_vec[idx],aot_865_CV_vec[idx]])
        
    #%% parallel
#    num_cores = multiprocessing.cpu_count()       
#    results = Parallel(n_jobs=num_cores, verbose=10)(delayed(extract_data)(line0[2:-1]) for line0 in file0)  
#    datetime_vec = [item[0] for item in results]
#    chlor_a_vec = [item[1] for item in results]
                 
    file0.close()
    
#    end = time.time()
#    print('Time processing: {} min'.format((end-start)/60))
    #%% Plot
    print(datetime_vec)
    print(chlor_a_filtered_stddev_vec)
    print(chlor_a_filtered_mean_vec)
    plt.figure(figsize=(12,8))
    # plt.subplot(2,1,1)
    # plt.plot(datetime_vec,chlor_a_filtered_mean_vec,'*b')
    # plt.plot(datetime_vec,chlor_a_filtered_mean_vec+chlor_a_filtered_stddev_vec,'--b')
    # plt.plot(datetime_vec,chlor_a_filtered_mean_vec-chlor_a_filtered_stddev_vec,'--b')
    # plt.plot(datetime_vec,chlor_a_filtered_median_vec,'*r')
    # plt.plot(datetime_vec,chlor_a_filtered_max_vec,'-b')
    # plt.plot(datetime_vec,chlor_a_filtered_min_vec,'-b')
    
#    plt.ylim(0,0.f5)
    
    # plt.subplot(2,1,2)
    
#    print(chlor_a_CV_vec)
#    plt.plot(datetime_vec,chlor_a_CV_vec,'k')
    plt.plot(datetime_vec,Rrs_412_CV_vec,'r')
    plt.plot(datetime_vec,Rrs_443_CV_vec,'g')
    plt.plot(datetime_vec,Rrs_490_CV_vec,'b')
    plt.plot(datetime_vec,Rrs_555_CV_vec,'k')
    plt.plot(datetime_vec,aot_865_CV_vec,'m')
    plt.plot(datetime_vec,median_CV_vec,'--c')


    plt.show()
    
#      % CV from Matlab script
#      if strcmp(sensor_id,'GOCI')
#            satcell.median_CV = nanmedian([...
#                  satcell.Rrs_412_CV,...
#                  satcell.Rrs_443_CV,...
#                  satcell.Rrs_490_CV,...
#                  satcell.Rrs_555_CV,...
#                  satcell.aot_865_CV]);
    
#%%
if __name__ == '__main__':
    main()    

   