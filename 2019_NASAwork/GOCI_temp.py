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
    
#def main():
#    start = time.time()  
path_main = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/GOCI_ROI_STATS_R2018_vcal_aqua/'
#path_folder = 'G2016006061640.L2_COMS_BRDF7/G2016006061640.L2_COMS_BRDF7_valregion'
path_list = 'file_list_sort.txt'
#    print(path_list)


filepath0 = os.path.join(path_main,path_list)
#    print(filepath0)
nlines = None 
nlines = file_len(filepath0)
#    print(nlines)
chlor_a_filtered_median_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
chlor_a_filtered_stddev_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
chlor_a_filtered_mean_vec = np.zeros((nlines,), dtype=np.float32) # array with chlor_a values
chlor_a_filtered_valid_pixel_count_vec = np.zeros((nlines,), dtype=np.float32)
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

senz_center_value_vec = np.zeros((nlines,), dtype=np.float32)
solz_center_value_vec = np.zeros((nlines,), dtype=np.float32)

datetime_vec = np.zeros((nlines,), dtype=np.dtype(datetime)) # array with datetime
#    print(len(chlor_a_filtered_mean_vec))

file0 = open(filepath0,'r')

#%% Sequencial
for idx, line in enumerate(file0):
    datetime_vec[idx] = extract_data(line[2:-1],path_main,'main_prod',None)
  
    chlor_a_filtered_mean_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_mean')
    chlor_a_filtered_median_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_median')
    chlor_a_filtered_stddev_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_stddev')
    chlor_a_filtered_valid_pixel_count_vec[idx] = extract_data(line[2:-1],path_main,'chlor_a','filtered_valid_pixel_count')
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

    senz_center_value_vec[idx] = extract_data(line[2:-1],path_main,'senz','center_value')
    solz_center_value_vec[idx] = extract_data(line[2:-1],path_main,'solz','center_value')   

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
#    print(datetime_vec)
#    print(chlor_a_filtered_stddev_vec)
#    print(chlor_a_filtered_mean_vec)

#plot
total_px_GOCI = 968*433 # new GCWS
ratio_from_the_total = 3 # 2 3 4 % half or third or fourth of the total of pixels
cond_area = chlor_a_filtered_valid_pixel_count_vec >= total_px_GOCI/ratio_from_the_total

plt.figure(figsize=(12,8))
plt.subplot(5,1,1)
plt.plot(datetime_vec,chlor_a_filtered_valid_pixel_count_vec,'o-b')
plt.plot(datetime_vec[cond_area],chlor_a_filtered_valid_pixel_count_vec[cond_area],'o-g')
# plt.plot(datetime_vec,chlor_a_filtered_mean_vec,'o-b')
# plt.plot(datetime_vec,chlor_a_filtered_mean_vec+chlor_a_filtered_stddev_vec,'--b')
# plt.plot(datetime_vec,chlor_a_filtered_mean_vec-chlor_a_filtered_stddev_vec,'--b')
# plt.plot(datetime_vec,chlor_a_filtered_median_vec,'*r')
# plt.plot(datetime_vec,chlor_a_filtered_max_vec,'-b')
# plt.plot(datetime_vec,chlor_a_filtered_min_vec,'-b')

#    plt.ylim(0,0.f5)

senz_lim = 60
cond_senz = senz_center_value_vec <=senz_lim

plt.subplot(5,1,2)
plt.plot(datetime_vec,senz_center_value_vec,'o-b')
plt.plot(datetime_vec[cond_senz],senz_center_value_vec[cond_senz],'o-g')
#    print(chlor_a_CV_vec)
#    plt.plot(datetime_vec,chlor_a_CV_vec,'k')
# plt.plot(datetime_vec,Rrs_412_CV_vec,'r')
# plt.plot(datetime_vec,Rrs_443_CV_vec,'g')
# plt.plot(datetime_vec,Rrs_490_CV_vec,'b')
# plt.plot(datetime_vec,Rrs_555_CV_vec,'k')
# plt.plot(datetime_vec,aot_865_CV_vec,'m')

solz_lim = 90
cond_solz = solz_center_value_vec <=solz_lim

plt.subplot(5,1,3)
plt.plot(datetime_vec,solz_center_value_vec,'o-b')
plt.plot(datetime_vec[cond_solz],solz_center_value_vec[cond_solz],'o-g')

median_CV_lim = 0.25
cond_median_CV = median_CV_vec <= median_CV_lim

plt.subplot(5,1,4)
plt.plot(datetime_vec,median_CV_vec,'o-b')
plt.plot(datetime_vec[cond_median_CV],median_CV_vec[cond_median_CV],'o-g')


cond_used = cond_area & cond_senz & cond_solz & cond_median_CV


chlor_a_filtered = chlor_a_filtered_mean_vec[cond_used]
datetime_filtered = datetime_vec[cond_used]
senz_filtered = senz_center_value_vec[cond_used]
solz_filtered = solz_center_value_vec[cond_used]

plt.subplot(5,1,5)
plt.plot(datetime_vec,chlor_a_filtered_mean_vec,'o-b')
plt.plot(datetime_filtered,chlor_a_filtered,'o-g')

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
year_vec = np.array([datetime_filtered[i].year for i in range(0,len(datetime_filtered))])
month_vec = np.array([datetime_filtered[i].month for i in range(0,len(datetime_filtered))])
day_vec = np.array([datetime_filtered[i].day for i in range(0,len(datetime_filtered))])
hour_vec = np.array([datetime_filtered[i].hour for i in range(0,len(datetime_filtered))])
minute_vec = np.array([datetime_filtered[i].minute for i in range(0,len(datetime_filtered))])
second_vec = np.array([datetime_filtered[i].second for i in range(0,len(datetime_filtered))])
doy_vec = np.array([datetime_filtered[i].timetuple().tm_yday for i in range(0,len(datetime_filtered))])
#%%

plt.figure(figsize=(12,8))
#% Spring - from March 1 to May 31; 3, 4, 5
cond_spring = ((month_vec == 3) | (month_vec == 4) | (month_vec == 5))
ax = plt.subplot(2,2,1)
ax.set_title('Spring')
cond_used = cond_spring & (hour_vec == 0)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'or',mfc='none')
cond_used = cond_spring & (hour_vec == 1)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color='lime',mfc='none')
cond_used = cond_spring & (hour_vec == 2)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ob',mfc='none')
cond_used = cond_spring & (hour_vec == 3)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ok',mfc='none')
cond_used = cond_spring & (hour_vec == 4)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'oc',mfc='none')
cond_used = cond_spring & (hour_vec == 5)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'om',mfc='none')
cond_used = cond_spring & (hour_vec == 6)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(1, 0.5, 0),mfc='none')
cond_used = cond_spring & (hour_vec == 7)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(0.5, 0, 0.5),mfc='none')
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

#% Summer - from June 1 to August 31; 6, 7, 8
cond_summer = ((month_vec == 6) | (month_vec == 7) | (month_vec == 8))
ax = plt.subplot(2,2,2)
ax.set_title('Summer')
cond_used = cond_summer & (hour_vec == 0)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'or',mfc='none')
cond_used = cond_summer & (hour_vec == 1)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color='lime',mfc='none')
cond_used = cond_summer & (hour_vec == 2)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ob',mfc='none')
cond_used = cond_summer & (hour_vec == 3)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ok',mfc='none')
cond_used = cond_summer & (hour_vec == 4)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'oc',mfc='none')
cond_used = cond_summer & (hour_vec == 5)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'om',mfc='none')
cond_used = cond_summer & (hour_vec == 6)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(1, 0.5, 0),mfc='none')
cond_used = cond_summer & (hour_vec == 7)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(0.5, 0, 0.5),mfc='none')
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

#% Fall (autumn) - from September 1 to November 30; and, 9, 10, 11
cond_fall = ((month_vec == 9) | (month_vec == 10) | (month_vec == 11))
ax = plt.subplot(2,2,3)
ax.set_title('Fall')
cond_used = cond_fall & (hour_vec == 0)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'or',mfc='none')
cond_used = cond_fall & (hour_vec == 1)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color='lime',mfc='none')
cond_used = cond_fall & (hour_vec == 2)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ob',mfc='none')
cond_used = cond_fall & (hour_vec == 3)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ok',mfc='none')
cond_used = cond_fall & (hour_vec == 4)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'oc',mfc='none')
cond_used = cond_fall & (hour_vec == 5)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'om',mfc='none')
cond_used = cond_fall & (hour_vec == 6)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(1, 0.5, 0),mfc='none')
cond_used = cond_fall & (hour_vec == 7)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(0.5, 0, 0.5),mfc='none')
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

#% Winter - from December 1 to February 28 (February 29 in a leap year). 12, 1, 2
cond_winter = ((month_vec == 12) | (month_vec == 1) | (month_vec == 2))
ax = plt.subplot(2,2,4)
ax.set_title('Winter')
cond_used = cond_winter & (hour_vec == 0)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'or',mfc='none')
cond_used = cond_winter & (hour_vec == 1)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color='lime',mfc='none')
cond_used = cond_winter & (hour_vec == 2)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ob',mfc='none')
cond_used = cond_winter & (hour_vec == 3)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'ok',mfc='none')
cond_used = cond_winter & (hour_vec == 4)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'oc',mfc='none')
cond_used = cond_winter & (hour_vec == 5)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'om',mfc='none')
cond_used = cond_winter & (hour_vec == 6)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(1, 0.5, 0),mfc='none')
cond_used = cond_winter & (hour_vec == 7)
plt.plot(chlor_a_filtered[cond_used],solz_filtered[cond_used],'o',color=(0.5, 0, 0.5),mfc='none')
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

plt.tight_layout()

path_out = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/Figures'
figname = os.path.join(path_out,'SZAvschlor_a_AllSeasons.pdf')
    #    print(figname)
plt.savefig(figname, dpi=300)

plt.show()
plt.close()
#%% 
cond_spring = ((month_vec == 3) | (month_vec == 4) | (month_vec == 5))
cond_summer = ((month_vec == 6) | (month_vec == 7) | (month_vec == 8))
cond_fall = ((month_vec == 9) | (month_vec == 10) | (month_vec == 11))
cond_winter = ((month_vec == 12) | (month_vec == 1) | (month_vec == 2))

#season_str = 'spring' 
#season_str = 'summer'
#season_str = 'fall'
season_str = ['spring','summer','fall','winter']

for season in season_str:
    
    doy_vec_aux = doy_vec

    if season == 'spring':
        cond_season = cond_spring
        vmin = 61
        vmax = 153
    elif season == 'summer':
        cond_season = cond_summer
        vmin = 153
        vmax = 245        
    elif season == 'fall':
        cond_season = cond_fall
        vmin = 245
        vmax = 336        
    elif season == 'winter':
        cond_season = cond_winter
        doy_vec_aux[doy_vec < 61] = doy_vec[doy_vec < 61] + 366
        vmin = 336
        vmax = 366+61        
    
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 8))
    plt.suptitle(season,y=1.05)
    
    kwargs = dict(vmin=vmin,vmax=vmax)
    
    ax = plt.subplot(4,2,1)
    ax.set_title('09:00 AM')
    cond_used = cond_season & (hour_vec == 0)
    plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    ax = plt.subplot(4,2,2)
    ax.set_title('10:00 AM')
    cond_used = cond_season & (hour_vec == 1)
    plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    ax = plt.subplot(4,2,3)
    ax.set_title('11:00 AM')
    cond_used = cond_season & (hour_vec == 2)
    plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    ax = plt.subplot(4,2,4)
    ax.set_title('12:00 PM')
    cond_used = cond_season & (hour_vec == 3)
    plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    ax = plt.subplot(4,2,5)
    ax.set_title('01:00 PM')
    cond_used = cond_season & (hour_vec == 4)
    plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    ax = plt.subplot(4,2,6)
    ax.set_title('02:00 PM')
    cond_used = cond_season & (hour_vec == 5)
    plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    ax = plt.subplot(4,2,7)
    ax.set_title('03:00 PM')
    cond_used = cond_season & (hour_vec == 6)
    plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    ax = plt.subplot(4,2,8)
    ax.set_title('04:00 PM')
    cond_used = cond_season & (hour_vec == 7)
    im = plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
    plt.xlabel('chlor_a')
    plt.ylabel('SZA ($^o$)')
    plt.xlim(0,0.3)
    plt.ylim(0,90)
    plt.grid(True)
    
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)
    
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8]) # control the colobar dimension
    cbar = fig.colorbar(im, cax=cb_ax)
    
    doy_month = [1,32,61,92,122,153,183,214,245,275,306,336,366,366+31,366+61]
    cbar.set_ticks(doy_month)
    cbar.set_ticklabels(['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan', 'Feb', 'Mar'])
    
    plt.tight_layout()
    
    path_out = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/Figures'
    figname = os.path.join(path_out,'SZAvschlor_a_'+season+'.pdf')
        #    print(figname)
    plt.savefig(figname, dpi=300,bbox_inches = 'tight')
        
    plt.show()
    plt.close()
    
#%%

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 8))
plt.suptitle('All Seasons',y=1.05)

doy_vec_aux = doy_vec
doy_vec_aux[doy_vec < 245] = doy_vec[doy_vec < 245] + (153-1)
doy_vec_aux[(doy_vec >= 245) & (doy_vec < 336)] = doy_vec[(doy_vec >= 245) & (doy_vec < 336)] - (245-1)
doy_vec_aux[doy_vec >=336] = doy_vec[doy_vec >=336] - (245-1)

kwargs = dict(vmin=1)

cond_season = cond_spring | cond_fall

ax = plt.subplot(4,2,1)
ax.set_title('09:00 AM')
cond_used = cond_season & (hour_vec == 0)
plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

ax = plt.subplot(4,2,2)
ax.set_title('10:00 AM')
cond_used = cond_season & (hour_vec == 1)
plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

ax = plt.subplot(4,2,3)
ax.set_title('11:00 AM')
cond_used = cond_season & (hour_vec == 2)
plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

ax = plt.subplot(4,2,4)
ax.set_title('12:00 PM')
cond_used = cond_season & (hour_vec == 3)
plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

ax = plt.subplot(4,2,5)
ax.set_title('01:00 PM')
cond_used = cond_season & (hour_vec == 4)
plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

ax = plt.subplot(4,2,6)
ax.set_title('02:00 PM')
cond_used = cond_season & (hour_vec == 5)
plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

ax = plt.subplot(4,2,7)
ax.set_title('03:00 PM')
cond_used = cond_season & (hour_vec == 6)
plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

ax = plt.subplot(4,2,8)
ax.set_title('04:00 PM')
cond_used = cond_season & (hour_vec == 7)
im = plt.scatter(chlor_a_filtered[cond_used],solz_filtered[cond_used],c=doy_vec_aux[cond_used],**kwargs)
plt.xlabel('chlor_a')
plt.ylabel('SZA ($^o$)')
plt.xlim(0,0.3)
plt.ylim(0,90)
plt.grid(True)

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                wspace=0.02, hspace=0.02)

cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8]) # control the colobar dimension
cbar = fig.colorbar(im, cax=cb_ax)

doy_month = [1,32,61,92,122,153,183,214,245,275,306,336,366]
cbar.set_ticks(doy_month)
cbar.set_ticklabels(['Sep','Oct','Nov','Dec','Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep'])

plt.tight_layout()

path_out = '/Users/javier/Desktop/Javier/2019_ROMA/2019_NASAwork/Figures'
figname = os.path.join(path_out,'SZAvschlor_a_All.pdf')
    #    print(figname)
plt.savefig(figname, dpi=300,bbox_inches = 'tight')
    
plt.show()
plt.close()    
#%%
#if __name__ == '__main__':
#    main()    