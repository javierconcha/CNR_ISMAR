#!/usr/bin/env python3
# coding: utf-8
"""
Created on Wed Jul  3 11:54:42 2019

@author: javier
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
import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import subprocess
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
import sys

from datetime import datetime
from scipy import stats
# from statsmodels.graphics.gofplots import qqplot

import pandas as pd   
from tabulate import tabulate

# to import apply_flags_OLCI.py
path_main = '/Users/javier.concha/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'
sys.path.insert(0,path_main)
import apply_flags_OLCI as OLCI_flags

# for plotting
color_dict = dict({\
 '400.00':'LightBlue',\
 '412.50':'DeepSkyBlue',\
 '442.50':'DodgerBlue',\
 '490.00':'Blue',\
 '510.00':'ForestGreen',\
 '560.00':'Green',\
 '620.00':'LightCoral',\
 '665.00':'Red',\
 '673.75':'Crimson',\
 '681.25':'FireBrick',\
 '708.75':'Silver',\
 '753.75':'Gray',\
 '778.75':'DimGray',\
 '865.00':'SlateGray',\
 '865.50':'SlateGray',\
 '885.00':'DarkSlateGray',\
'1020.50':'Black'})
    
station_n = {'Venise':1,'Galata_Platform':2,'Gloria':3,'Helsinki_Lighthouse':4,'Gustav_Dalen_Tower':5,\
             'Palgrunden':6,'Thornton_C-power':7,'LISCO':8,'Lake_Erie':9,'WaveCIS_Site_CSI_6':10,\
                 'USC_SEAPRISM':11,'USC_SEAPRISM_2':12}

create_list_flag = 0

plot_flag = False
#%% Open in situ in netcdf format from excel_to_nc_AquaAlta_merge_newsite.py by Marco B.
"""
<class 'netCDF4._netCDF4.Dataset'>
root group (NETCDF4 data model, file format HDF5):
    dimensions(sizes): Time(1044), Central_wavelenghts(8)
    variables(dimensions): 
        <class 'str'> Time(Time), 
        float32 Level(Time), 
        <class 'str'> Julian_day(Time), 
        int32 Instrument_number(Time), 
        float32 Exact_wavelengths(Time,Central_wavelenghts), 
        float32 Solar_zenith(Time,Central_wavelenghts), 
        float32 Solar_azimuth(Time,Central_wavelenghts), 
        float32 Lt_mean(Time,Central_wavelenghts), 
        float32 Lt_standard_deviation(Time,Central_wavelenghts), 
        float32 Lt_min_rel(Time,Central_wavelenghts), 
        float32 Li_mean(Time,Central_wavelenghts), 
        float32 Li_standard_deviation(Time,Central_wavelenghts), 
        float32 AOT(Time,Central_wavelenghts), 
        float32 OOT(Time,Central_wavelenghts), 
        float32 ROT(Time,Central_wavelenghts), 
        float32 Lw(Time,Central_wavelenghts), 
        float32 Lw_Q(Time,Central_wavelenghts), 
        float32 Lwn(Time,Central_wavelenghts), 
        float32 Lwn_fonQ(Time,Central_wavelenghts), 
        float32 Pressure(Time), 
        float32 Wind_speed(Time), 
        float32 CHL-A(Time), 
        float32 SSR(Time), 
        float32 O3(Time)
"""
#%%
def create_OLCI_list(path_main,Time,year_vec,month_vec,doy_vec,day_vec):
    #% build L2 filename based on date and see if it exist
    """
    Example file: S3A_OL_2_WFR____20160528T091009_20160528T091209_20171101T001113_0119_004_321______MR1_R_NT_002
    """
    count = 0
    f = open(os.path.join(path_main,'OLCI_list.txt'),"w+")
    
    for idx in range(0,len(Time)):
        year_str = str(int(year_vec[idx]))
        
        month_str = str(int(month_vec[idx]))
        if month_vec[idx] < 10:
            month_str = '0'+month_str
            
          
        doy_str = str(int(doy_vec[idx]))  
        if doy_vec[idx] < 100:
            if doy_vec[idx] < 10:
                doy_str = '00'+doy_str
            else:
                doy_str = '0'+doy_str
        
        day_str = str(int(day_vec[idx]))
        if day_vec[idx] < 10:
            day_str = '0'+day_str
        
#        dir_path = os.path.join('/DataArchive/OC/OLCI/sources_baseline_2.23',year_str,doy_str)
        L2_filename = 'S3A_OL_2_WFR____'+year_str+month_str+day_str
        
        
        #    print(L2_filename)
#        path = os.path.join(dir_path,L2_filename)
        #    print(path)
        (ls_status, ls_output) = subprocess.getstatusoutput("grep "+L2_filename+" 20160426_20190228.txt")
        if not ls_status:
#        print('--------------------------------------')
            f.write('/'+year_str+'/'+doy_str+'/'+ls_output+'\n')
            count = count+1
    print('Matchups Total: '+str(int(count)))
    f.close()
    
    (ls_status, ls_output) = subprocess.getstatusoutput('cat '+os.path.join(path_main,'OLCI_list.txt')\
     +'|sort|uniq >'+os.path.join(path_main,'OLCI_list_uniq.txt'))
#%% Get FO from Thuiller
def get_F0(wl,path_main):
    path_to_file = os.path.join(path_main,'Thuillier_F0.nc')
    nc_f0 = Dataset(path_to_file,'r')
    Wavelength = nc_f0.variables['Wavelength'][:]
    F0 = nc_f0.variables['F0'][:]
    F0_val = np.interp(wl,Wavelength,F0)
    return F0_val
#%%    root mean squared error
def rmse(predictions, targets):
    return np.sqrt(((np.asarray(predictions) - np.asarray(targets)) ** 2).mean())       
#%%
def plot_scatter(x,y,str1,path_out,prot_name,sensor_name,station_vec,min_val,max_val): 

    # replace nan in y (sat data)
    x = np.array(x)
    y = np.array(y)
    station_vec = np.array(station_vec)

    x = x[~np.isnan(y)] # it is assumed that only sat data could be nan
    station_vec = station_vec[~np.isnan(y)]
    y = y[~np.isnan(y)]

    rmse_val = np.nan
    mean_abs_rel_diff = np.nan
    mean_rel_diff = np.nan
    r_value = np.nan
    mean_abs_rel_diff = np.nan
    mean_bias = np.nan
    mean_abs_error = np.nan

    rmse_val_Venise = np.nan
    mean_abs_rel_diff_Venise = np.nan
    mean_rel_diff_Venise = np.nan
    r_value_Venise = np.nan
    mean_abs_rel_diff_Venise = np.nan
    mean_bias_Venise = np.nan
    mean_abs_error_Venise = np.nan

    rmse_val_Gloria = np.nan
    mean_abs_rel_diff_Gloria = np.nan
    mean_rel_diff_Gloria = np.nan
    r_value_Gloria = np.nan
    mean_abs_rel_diff_Gloria = np.nan
    mean_bias_Gloria = np.nan
    mean_abs_error_Gloria = np.nan

    rmse_val_Galata_Platform = np.nan
    mean_abs_rel_diff_Galata_Platform = np.nan
    mean_rel_diff_Galata_Platform = np.nan
    r_value_Galata_Platform = np.nan
    mean_abs_rel_diff_Galata_Platform = np.nan
    mean_bias_Galata_Platform = np.nan
    mean_abs_error_Galata_Platform = np.nan

    rmse_val_Helsinki_Lighthouse = np.nan
    mean_abs_rel_diff_Helsinki_Lighthouse = np.nan
    mean_rel_diff_Helsinki_Lighthouse = np.nan
    r_value_Helsinki_Lighthouse = np.nan
    mean_abs_rel_diff_Helsinki_Lighthouse = np.nan
    mean_bias_Helsinki_Lighthouse = np.nan
    mean_abs_error_Helsinki_Lighthouse = np.nan

    rmse_val_Gustav_Dalen_Tower = np.nan
    mean_abs_rel_diff_Gustav_Dalen_Tower = np.nan
    mean_rel_diff_Gustav_Dalen_Tower = np.nan
    r_value_Gustav_Dalen_Tower  = np.nan
    mean_abs_rel_diff_Gustav_Dalen_Tower = np.nan
    mean_bias_Gustav_Dalen_Tower = np.nan
    mean_abs_error_Gustav_Dalen_Tower = np.nan

    count_Venise = 0
    count_Gloria = 0
    count_Galata_Platform = 0
    count_Helsinki_Lighthouse = 0
    count_Gustav_Dalen_Tower = 0

    plt.figure()
    #plt.errorbar(x, y, xerr=e_x, yerr=e_y, fmt='or')
    for cnt, line in enumerate(y):
        if station_vec[cnt] == 'Venise':
            mrk_color = 'r'
            count_Venise = count_Venise+1
        elif station_vec[cnt] == 'Gloria':
            mrk_color = 'g'
            count_Gloria = count_Gloria+1
        elif station_vec[cnt] == 'Galata_Platform':
            mrk_color = 'b'
            count_Galata_Platform = count_Galata_Platform+1
        elif station_vec[cnt] == 'Helsinki_Lighthouse':
            mrk_color = 'm'
            count_Helsinki_Lighthouse = count_Helsinki_Lighthouse+1
        elif station_vec[cnt] == 'Gustav_Dalen_Tower':
            mrk_color = 'c'
            count_Gustav_Dalen_Tower = count_Gustav_Dalen_Tower+1

        if prot_name[:2] == 'ba':
            mrk_style = 'x'
        elif  prot_name[:2] == 'zi':
            mrk_style = '+' 

        plt.plot(x[cnt], y[cnt],color=mrk_color,marker=mrk_style)
    plt.axis([min_val, max_val, min_val, max_val])
    plt.gca().set_aspect('equal', adjustable='box')
    # plot 1:1 line
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    plt.plot([xmin,xmax],[ymin, ymax],'--k')
    
    # Generated linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    line = slope*np.array([xmin,xmax],dtype=np.float32)+intercept
    plt.plot([xmin,xmax], line)
    # plt.legend(['1:1','Regression Line'])
    plt.xlabel('$L^{PRS}_{WN}$',fontsize=12)
    plt.ylabel('$L^{'+sensor_name+'}_{WN}$',fontsize=12)
    if (xmin<0 or ymin<0):
        plt.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)  
    
    # stats
    N = len(x)
    
    ref_obs = np.asarray(x)
    sat_obs = np.asarray(y)
    rmse_val = rmse(sat_obs,ref_obs)

    diff = sat_obs-ref_obs

    # the mean of relative (signed) percent differences
    rel_diff = 100*diff/ref_obs
    mean_rel_diff = np.mean(rel_diff)
        
    #  the mean of absolute (unsigned) percent differences
    mean_abs_rel_diff = np.mean(np.abs(rel_diff))
    
    # mean bias
    mean_bias = np.mean(diff)

    # mean absolute error (MAE)
    mean_abs_error = np.mean(np.abs(diff))
    

    cond_station = np.asarray(station_vec)=='Venise'
    if sum(cond_station):
        ref_obs_Venise = ref_obs[cond_station]
        sat_obs_Venise = sat_obs[cond_station]
        slope_Venise, intercept_Venise, r_value_Venise, p_value_Venise, std_err_Venise = stats.linregress(ref_obs_Venise,sat_obs_Venise)
        rmse_val_Venise = rmse(sat_obs_Venise,ref_obs_Venise)
        diff_Venise = (sat_obs_Venise-ref_obs_Venise)
        rel_diff_Venise = 100*diff_Venise/ref_obs_Venise
        mean_rel_diff_Venise = np.mean(rel_diff_Venise)
        mean_abs_rel_diff_Venise = np.mean(np.abs(rel_diff_Venise))
        mean_bias_Venise = np.mean(diff_Venise)
        mean_abs_error_Venise = np.mean(np.abs(diff_Venise))

    cond_station = np.asarray(station_vec)=='Gloria'
    if sum(cond_station):    
        ref_obs_Gloria = ref_obs[cond_station]
        sat_obs_Gloria = sat_obs[cond_station]
        slope_Gloria, intercept_Gloria, r_value_Gloria, p_value_Gloria, std_err_Gloria = stats.linregress(ref_obs_Gloria,sat_obs_Gloria)
        rmse_val_Gloria = rmse(sat_obs_Gloria,ref_obs_Gloria)
        diff_Gloria = (sat_obs_Gloria-ref_obs_Gloria)
        rel_diff_Gloria = 100*diff_Gloria/ref_obs_Gloria
        mean_rel_diff_Gloria = np.mean(rel_diff_Gloria)
        mean_abs_rel_diff_Gloria = np.mean(np.abs(rel_diff_Gloria))
        mean_bias_Gloria = np.mean(diff_Gloria)
        mean_abs_error_Gloria = np.mean(np.abs(diff_Gloria))
        
    cond_station = np.asarray(station_vec)=='Galata_Platform'
    if sum(cond_station):    
        ref_obs_Galata_Platform = ref_obs[cond_station]
        sat_obs_Galata_Platform = sat_obs[cond_station]
        slope_Galata_Platform, intercept_Galata_Platform, r_value_Galata_Platform, p_value_Galata_Platform, std_err_Galata_Platform = stats.linregress(ref_obs_Galata_Platform,sat_obs_Galata_Platform)
        rmse_val_Galata_Platform = rmse(sat_obs_Galata_Platform,ref_obs_Galata_Platform)
        diff_Galata_Platform = (sat_obs_Galata_Platform-ref_obs_Galata_Platform)
        rel_diff_Galata_Platform = 100*diff_Galata_Platform/ref_obs_Galata_Platform
        mean_rel_diff_Galata_Platform = np.mean(rel_diff_Galata_Platform)
        mean_abs_rel_diff_Galata_Platform = np.mean(np.abs(rel_diff_Galata_Platform))
        mean_bias_Galata_Platform = np.mean(diff_Galata_Platform)
        mean_abs_error_Galata_Platform = np.mean(np.abs(diff_Galata_Platform))
        
    cond_station = np.asarray(station_vec)=='Helsinki_Lighthouse'
    if sum(cond_station):    
        ref_obs_Helsinki_Lighthouse = ref_obs[cond_station]
        sat_obs_Helsinki_Lighthouse = sat_obs[cond_station]
        slope_Helsinki_Lighthouse, intercept_Helsinki_Lighthouse, r_value_Helsinki_Lighthouse, p_value_Helsinki_Lighthouse, std_err_Helsinki_Lighthouse = stats.linregress(ref_obs_Helsinki_Lighthouse,sat_obs_Helsinki_Lighthouse)
        rmse_val_Helsinki_Lighthouse = rmse(sat_obs_Helsinki_Lighthouse,ref_obs_Helsinki_Lighthouse)
        diff_Helsinki_Lighthouse = (sat_obs_Helsinki_Lighthouse-ref_obs_Helsinki_Lighthouse)
        rel_diff_Helsinki_Lighthouse = 100*diff_Helsinki_Lighthouse/ref_obs_Helsinki_Lighthouse
        mean_rel_diff_Helsinki_Lighthouse = np.mean(rel_diff_Helsinki_Lighthouse)
        mean_abs_rel_diff_Helsinki_Lighthouse = np.mean(np.abs(rel_diff_Helsinki_Lighthouse))
        mean_bias_Helsinki_Lighthouse = np.mean(diff_Helsinki_Lighthouse)
        mean_abs_error_Helsinki_Lighthouse = np.mean(np.abs(diff_Helsinki_Lighthouse))
        
    cond_station = np.asarray(station_vec)=='Gustav_Dalen_Tower'
    if sum(cond_station):    
        ref_obs_Gustav_Dalen_Tower = ref_obs[cond_station]
        sat_obs_Gustav_Dalen_Tower = sat_obs[cond_station]
        slope_Gustav_Dalen_Tower, intercept_Gustav_Dalen_Tower, r_value_Gustav_Dalen_Tower, p_value_Gustav_Dalen_Tower, std_err_Gustav_Dalen_Tower = stats.linregress(ref_obs_Gustav_Dalen_Tower,sat_obs_Gustav_Dalen_Tower)
        rmse_val_Gustav_Dalen_Tower = rmse(sat_obs_Gustav_Dalen_Tower,ref_obs_Gustav_Dalen_Tower)
        diff_Gustav_Dalen_Tower = (sat_obs_Gustav_Dalen_Tower-ref_obs_Gustav_Dalen_Tower)
        rel_diff_Gustav_Dalen_Tower = 100*diff_Gustav_Dalen_Tower/ref_obs_Gustav_Dalen_Tower
        mean_rel_diff_Gustav_Dalen_Tower = np.mean(rel_diff_Gustav_Dalen_Tower)
        mean_abs_rel_diff_Gustav_Dalen_Tower = np.mean(np.abs(rel_diff_Gustav_Dalen_Tower))
        mean_bias_Gustav_Dalen_Tower = np.mean(diff_Gustav_Dalen_Tower)
        mean_abs_error_Gustav_Dalen_Tower = np.mean(np.abs(diff_Gustav_Dalen_Tower))
    

    str2 = str1
    # to print without .0
    if str1[-2:]=='.0':
        str2 = str2[:-2]
        
    
    str0 = '{}nm\nN={:d}\nrmse={:,.2f}\nMAPD={:,.0f}%\nMPD={:,.0f}%\nMean Bias={:,.2f}\nMAE={:,.2f}\n$r^2$={:,.2f}'\
    .format(str2,\
            N,\
            rmse_val,\
            mean_abs_rel_diff,\
            mean_rel_diff,\
            mean_bias,\
            mean_abs_error,\
            r_value**2)
        
    plt.text(0.05, 0.58, str0,horizontalalignment='left', fontsize=12,transform=plt.gca().transAxes)
    
    ofname = sensor_name+'_scatter_mu_'+str1.replace(".","p")+'_'+prot_name+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    
    plt.savefig(ofname, dpi=300)

    # latex table
    if prot_name[:2] == 'ba':
        prot_name_str = 'BW06'
    elif  prot_name[:2] == 'zi':
        prot_name_str = 'Z09'

    if str1 == '412.5':
        print('proto & nm & N & rmse & MAPD & MPD & mean bias & MAE & $r^2$\n')
    str_table = '{} & {} & {:d} & {:,.2f} & {:,.1f} & {:,.1f} & {:,.2f} & {:,.2f} & {:,.2f}\\\\'\
    .format(prot_name_str,\
            str2,\
            N,\
            rmse_val,\
            mean_abs_rel_diff,\
            mean_rel_diff,\
            mean_bias,\
            mean_abs_error,\
            r_value**2)

    print(str_table)
 
    print('count_Venise: '+str(count_Venise))
    print('count_Gloria: '+str(count_Gloria))
    print('count_Galata_Platform: '+str(count_Galata_Platform))
    print('count_Helsinki_Lighthouse: '+str(count_Helsinki_Lighthouse))
    print('count_Gustav_Dalen_Tower: '+str(count_Gustav_Dalen_Tower))

    # plt.show()   
    return rmse_val, mean_abs_rel_diff, mean_rel_diff, mean_bias, mean_abs_error, r_value**2,\
        rmse_val_Venise, mean_abs_rel_diff_Venise, mean_rel_diff_Venise, mean_bias_Venise, mean_abs_error_Venise, r_value_Venise**2,\
        rmse_val_Gloria, mean_abs_rel_diff_Gloria, mean_rel_diff_Gloria, mean_bias_Gloria, mean_abs_error_Gloria, r_value_Gloria**2,\
        rmse_val_Galata_Platform, mean_abs_rel_diff_Galata_Platform, mean_rel_diff_Galata_Platform, mean_bias_Galata_Platform, mean_abs_error_Galata_Platform, r_value_Galata_Platform**2,\
        rmse_val_Helsinki_Lighthouse, mean_abs_rel_diff_Helsinki_Lighthouse, mean_rel_diff_Helsinki_Lighthouse, mean_bias_Helsinki_Lighthouse, mean_abs_error_Helsinki_Lighthouse, r_value_Helsinki_Lighthouse**2,\
        rmse_val_Gustav_Dalen_Tower, mean_abs_rel_diff_Gustav_Dalen_Tower, mean_rel_diff_Gustav_Dalen_Tower, mean_bias_Gustav_Dalen_Tower, mean_abs_error_Gustav_Dalen_Tower, r_value_Gustav_Dalen_Tower**2
#%%
def plot_both_methods(wl_str,notation_flag,path_out,min_val,max_val):

    np.warnings.filterwarnings('ignore')

    print('=====================================')
    print(wl_str)
    if wl_str == '412.5':
        str0 = '0412p50'
        str3 = '412.5'
    elif wl_str == '442.5':
        str0 = '0442p50'
        str3 = '442.5'
    elif wl_str == '490.0':
        str0 = '0490p00'
        str3 = '490'
    elif wl_str == '560.0':
        str0 = '0560p00'
        str3 = '560'
    elif wl_str == '665.0':
        str0 = '0665p00'
        str3 = '665'
    
    ins_zi_station = globals()['mu_Lwn_'+str0+'_fq_ins_zi_station']
    ins_ba_station = globals()['mu_Lwn_'+str0+'_fq_ins_ba_station']

    sat_zi_stop_time = globals()['mu_Lwn_'+str0+'_fq_sat_zi_stop_time']
    sat_ba_stop_time = globals()['mu_Lwn_'+str0+'_fq_sat_ba_stop_time']

    ins_zi_time = globals()['mu_Lwn_'+str0+'_fq_ins_zi_time']
    ins_ba_time = globals()['mu_Lwn_'+str0+'_fq_ins_ba_time']

    sat_zi = globals()['mu_Lwn_'+str0+'_fq_sat_zi']
    sat_ba = globals()['mu_Lwn_'+str0+'_fq_sat_ba']

    ins_zi = globals()['mu_Lwn_'+str0+'_fq_ins_zi']
    ins_ba = globals()['mu_Lwn_'+str0+'_fq_ins_ba']

    count_both = 0
    count_zi = len(ins_zi_station)
    count_ba = len(ins_ba_station)

    diff = []
    sat_same_zi = []
    sat_same_ba = []
    ins_same_zi = []
    ins_same_ba = []
    ins_same_station = []

    #%% time series with two methods
    plt.figure(figsize=(16,4))
    for cnt, line in enumerate(ins_zi_station):
        if ins_zi_station[cnt] == 'Venise':
            mrk_style = '+r'
            mrk_style_ins = '>r'
        elif ins_zi_station[cnt] == 'Gloria':
            mrk_style = '+g'
            mrk_style_ins = '>g'
        elif ins_zi_station[cnt] == 'Galata_Platform':
            mrk_style = '+b'
            mrk_style_ins = '>b'
        elif ins_zi_station[cnt] == 'Helsinki_Lighthouse':
            mrk_style = '+m'
            mrk_style_ins = '>m'
        elif ins_zi_station[cnt] == 'Gustav_Dalen_Tower':
            mrk_style = '+c'
            mrk_style_ins = '>c'
            
        cond1 = sat_zi_stop_time[cnt]==np.array(sat_ba_stop_time) # same S3A/OLCI file
        cond2 = ins_zi_station[cnt]==np.array(ins_ba_station) # same stations for both methods
        idx=np.where(cond1&cond2)
        # print('=====================================')
        # print(cnt)
        if len(idx[0]) == 1:
            count_both = count_both+1
            # print('There is coincident matchups in BW.')
            # print(sat_zi_stop_time[cnt])
            # print(sat_ba_stop_time[idx[0][0]])
            # print(ins_zi_station[cnt])
            # print(ins_ba_station[idx[0][0]])
            plt.plot(sat_zi_stop_time[cnt], sat_zi[cnt],mrk_style)
            plt.plot([sat_zi_stop_time[cnt],sat_ba_stop_time[idx[0][0]]],\
                [sat_zi[cnt],sat_ba[idx[0][0]]],mrk_style[1])
            
            plt.plot(ins_zi_time[cnt], ins_zi[cnt],mrk_style_ins,mfc='none')
            plt.plot([ins_zi_time[cnt],ins_ba_time[idx[0][0]]],\
                [ins_zi[cnt],ins_ba[idx[0][0]]],mrk_style_ins[1])

            # to connect in situ with sat
            plt.plot([ins_zi_time[cnt],sat_zi_stop_time[cnt]],\
                [ins_zi[cnt],sat_zi[cnt]],mrk_style_ins[1],linestyle='dashed')

            plt.plot([ins_ba_time[idx[0][0]],sat_ba_stop_time[idx[0][0]]],\
                [ins_ba[idx[0][0]],sat_ba[idx[0][0]]],mrk_style_ins[1],linestyle='dotted')

            diff.append(sat_zi[cnt]-sat_ba[idx[0][0]])

            # same dataset
            sat_same_zi.append(sat_zi[cnt])
            sat_same_ba.append(sat_ba[idx[0][0]])
            ins_same_zi.append(ins_zi[cnt])
            ins_same_ba.append(ins_ba[idx[0][0]])
            ins_same_station.append(ins_zi_station[cnt])

            percent_change = 100*np.abs(sat_zi[cnt]-sat_ba[idx[0][0]])/max([sat_zi[cnt],sat_ba[idx[0][0]]])
            str2 = '{:,.2f}%'.format(percent_change)
            # print(str2)
            if notation_flag:
                plt.text(sat_zi_stop_time[cnt], sat_zi[cnt],str2)
            
        else:
            # print(idx[0])
            # print(sat_zi_stop_time[cnt])
            plt.plot(sat_zi_stop_time[cnt], sat_zi[cnt],mrk_style)
            plt.plot(ins_zi_time[cnt], ins_zi[cnt],mrk_style_ins,mfc='none')
            # to connect in situ with sat
            plt.plot([ins_zi_time[cnt],sat_zi_stop_time[cnt]],\
            [ins_zi[cnt],sat_zi[cnt]],mrk_style_ins[1],linestyle='dashed')
                
    for cnt, line in enumerate(ins_ba_station):
        if ins_ba_station[cnt] == 'Venise':
            mrk_style = 'xr'
            mrk_style_ins = 'or'
        elif ins_ba_station[cnt] == 'Gloria':
            mrk_style = 'xg'
            mrk_style_ins = 'og'
        elif ins_ba_station[cnt] == 'Galata_Platform':
            mrk_style = 'xb'
            mrk_style_ins = 'ob'
        elif ins_ba_station[cnt] == 'Helsinki_Lighthouse':
            mrk_style = 'xm'
            mrk_style_ins = 'om'
        elif ins_ba_station[cnt] == 'Gustav_Dalen_Tower':
            mrk_style = 'xc'
            mrk_style_ins = 'oc'
        plt.plot(sat_ba_stop_time[cnt], sat_ba[cnt],mrk_style)
        plt.plot(ins_ba_time[cnt], ins_ba[cnt],mrk_style_ins,mfc='none')

        # to connect in situ with sat
        plt.plot([ins_ba_time[cnt],sat_ba_stop_time[cnt]],\
                [ins_ba[cnt],sat_ba[cnt]],mrk_style_ins[1],linestyle='dotted')

    plt.xlabel('Time',fontsize=12)
    sensor_name = 'OLCI'
    plt.ylabel('$L^{'+sensor_name+'}_{WN}$',fontsize=12)
    # zero line
    xmin, xmax = plt.gca().get_xlim()
    plt.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)  
    plt.text(0.05, 0.95, str3+'nm',horizontalalignment='left',fontsize=12,transform=plt.gca().transAxes)

    # save fig
    ofname = sensor_name+'_timeseries_diff_ba_zi_'+wl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    #%% histograms of both dataset: zi and ba
    kwargs2 = dict(bins='auto', histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w')
    ax1.hist(sat_zi,color='red', **kwargs2)
    ax1.hist(sat_ba,color='black', **kwargs2)
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim([x0,x0+1*(x1-x0)])

    ax1.set_ylabel('Frequency (counts)',fontsize=12)

    str1 = f'Z09\nmedian: {np.nanmedian(sat_zi):,.2f}\
            \nmean: {np.nanmean(sat_zi):,.2f}\nN: {len(sat_zi):,.0f}'

    str2 = f'BW06\nmedian: {np.nanmedian(sat_ba):,.2f}\
            \nmean: {np.nanmean(sat_ba):,.2f}\nN: {len(sat_ba):,.0f}'

    bottom, top = ax1.get_ylim()
    left, right = ax1.get_xlim()
    ypos = bottom+0.78*(top-bottom)
    ax1.text(left+0.01*(right-left),bottom+0.95*(top-bottom), f'{str3}nm', fontsize=12,color='black')
    ax1.text(left+0.75*(right-left),ypos, str1, fontsize=12,color='red')
    ax1.text(left+0.50*(right-left),ypos, str2, fontsize=12,color='black')

    fig.text(0.5,0.01,'$L^{'+sensor_name+'}_{WN}$',ha='center',fontsize=12)

    # save fig
    ofname = sensor_name+'_hist_ba_zi_'+wl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    #latex table
    if wl_str == '412.5':
        print('proto & nm & min & max & std & median & mean & N\\\\')
    str_table = 'Z09 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(wl_str,\
            np.nanmin(sat_zi),
            np.nanmax(sat_zi),
            np.nanstd(sat_zi),
            np.nanmedian(sat_zi),
            np.nanmean(sat_zi),
            len(sat_zi))
    print(str_table)
    str_table = 'BW06 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(wl_str,\
            np.nanmin(sat_ba),
            np.nanmax(sat_ba),
            np.nanstd(sat_ba),
            np.nanmedian(sat_ba),
            np.nanmean(sat_ba),
            len(sat_ba))
    print(str_table)
    
    #%% histograms of same dataset: zi and ba
    kwargs2 = dict(bins='auto', histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w')
    ax1.hist(sat_same_zi,color='red', **kwargs2)
    ax1.hist(sat_same_ba,color='black', **kwargs2)
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim([x0,x0+1*(x1-x0)])

    ax1.set_ylabel('Frequency (counts)',fontsize=12)

    str1 = f'Z09\nmedian: {np.nanmedian(sat_same_zi):,.2f}\
            \nmean: {np.nanmean(sat_same_zi):,.2f}\nN: {len(sat_same_zi):,.0f}'

    str2 = f'BW06\nmedian: {np.nanmedian(sat_same_ba):,.2f}\
            \nmean: {np.nanmean(sat_same_ba):,.2f}\nN: {len(sat_same_ba):,.0f}'

    bottom, top = ax1.get_ylim()
    left, right = ax1.get_xlim()
    ypos = bottom+0.78*(top-bottom)
    ax1.text(left+0.01*(right-left),bottom+0.95*(top-bottom), f'{str3}nm', fontsize=12,color='black')
    ax1.text(left+0.75*(right-left),ypos, str1, fontsize=12,color='red')
    ax1.text(left+0.50*(right-left),ypos, str2, fontsize=12,color='black')

    fig.text(0.5,0.01,'$L^{'+sensor_name+'}_{WN}$',ha='center',fontsize=12)

    # save fig
    ofname = sensor_name+'_hist_same_ba_zi_'+wl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    #latex table
    if wl_str == '412.5':
        print('proto & nm & min & max & std & median & mean & N\\\\')
    str_table = 'Z09 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(wl_str,\
            np.nanmin(sat_same_zi),
            np.nanmax(sat_same_zi),
            np.nanstd(sat_same_zi),
            np.nanmedian(sat_same_zi),
            np.nanmean(sat_same_zi),
            len(sat_same_zi))
    print(str_table)
    str_table = 'BW06 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(wl_str,\
            np.nanmin(sat_same_ba),
            np.nanmax(sat_same_ba),
            np.nanstd(sat_same_ba),
            np.nanmedian(sat_same_ba),
            np.nanmean(sat_same_ba),
            len(sat_same_ba))
    print(str_table)
    #%% # normality test
    # plot_normality(np.array(diff),wl_str)
    
    #%% histogram of the difference
    kwargs2 = dict(bins='auto', histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w')
    ax1.hist(diff, **kwargs2)
    # x0, x1 = ax1.get_xlim()
    # ax1.set_xlim([x0,x0+0.15*(x1-x0)])

    ax1.set_ylabel('Frequency (counts)',fontsize=12)

    str1 = f'{str3}nm\nmedian: {np.nanmedian(diff):,.4f}\
            \nmean: {np.nanmean(diff):,.4f}\nN: {len(diff):,.0f}'

    if wl_str == '412.5':
        print('diff & nm & min & max & std & median & mean & N\\\\')
    str_table = 'diff & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.4f} & {:,.4f} & {:,.0f}'\
    .format(str3,
            np.nanmin(diff),
            np.nanmax(diff),
            np.nanstd(diff),
            np.nanmedian(diff),
            np.nanmean(diff),
            len(diff))
    print(str_table)

    bottom, top = ax1.get_ylim()
    left, right = ax1.get_xlim()
    xpos = left+0.02*(right-left)
    ax1.text(xpos,bottom+0.78*(top-bottom), str1, fontsize=12)

    fig.text(0.5,0.01,'Diff. $L^{'+sensor_name+'}_{WN}$',ha='center',fontsize=12)

    # ax2.hist(diff, **kwargs2)
    # x0, x1 = ax2.get_xlim()
    # ax2.set_xlim([x0+0.70*(x1-x0),x1])

    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.yaxis.tick_right()
    # ax2.tick_params(labelright='on')

    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((1-d, 1+d), (-d, +d), **kwargs)        # top-left diagonal
    # ax1.plot((1-d,1+d), (1-d, 1+d), **kwargs)  # top-right diagonal
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, d), (- d, + d), **kwargs)  # bottom-left diagonal
    # ax2.plot((- d,  d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # save fig
    ofname = sensor_name+'_hist_diff_ba_zi_'+wl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)
    plt.show()

    print('Number of Matchups for both: '+str(count_both))
    print('Number of Matchups for Zibordi: '+str(count_zi))
    print('Number of Matchups for Bailey: '+str(count_ba))

    #%   scatter plot with both methods
    plt.figure()
    for cnt, line in enumerate(ins_zi_station):
            if ins_zi_station[cnt] == 'Venise':
                mrk_style = '+r'
            elif ins_zi_station[cnt] == 'Gloria':
                mrk_style = '+g'
            elif ins_zi_station[cnt] == 'Galata_Platform':
                mrk_style = '+b'
            elif ins_zi_station[cnt] == 'Helsinki_Lighthouse':
                mrk_style = '+m'
            elif ins_zi_station[cnt] == 'Gustav_Dalen_Tower':
                mrk_style = '+c'
                
            cond1 = sat_zi_stop_time[cnt]==np.array(sat_ba_stop_time) # same S3A/OLCI file
            cond2 = ins_zi_station[cnt]==np.array(ins_ba_station) # same stations for both methods
            idx=np.where(cond1&cond2)
            # print('=====================================')
            # print(cnt)
            if len(idx[0]) == 1:
                plt.plot(ins_zi[cnt], sat_zi[cnt],mrk_style)
                plt.plot([ins_zi[cnt],ins_ba[idx[0][0]]],\
                    [sat_zi[cnt],sat_ba[idx[0][0]]],mrk_style[1])
                
                percent_change = 100*np.abs(sat_zi[cnt]-sat_ba[idx[0][0]])/max([sat_zi[cnt],sat_ba[idx[0][0]]])
                str1 = '{:,.2f}%'.format(percent_change)
                # print(str1)
                if notation_flag:
                    plt.text(ins_zi[cnt], sat_zi[cnt],str1)
            else:
                plt.plot(ins_zi[cnt], sat_zi[cnt],mrk_style)
                
    for cnt, line in enumerate(ins_ba_station):
            if ins_ba_station[cnt] == 'Venise':
                mrk_style = 'xr'
            elif ins_ba_station[cnt] == 'Gloria':
                mrk_style = 'xg'
            elif ins_ba_station[cnt] == 'Galata_Platform':
                mrk_style = 'xb'
            elif ins_ba_station[cnt] == 'Helsinki_Lighthouse':
                mrk_style = 'xm'
            elif ins_ba_station[cnt] == 'Gustav_Dalen_Tower':
                mrk_style = 'xc'
            plt.plot(ins_ba[cnt], sat_ba[cnt],mrk_style)

    plt.axis([min_val, max_val, min_val, max_val])
    plt.gca().set_aspect('equal', adjustable='box')
    # plot 1:1 line
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    plt.plot([xmin,xmax],[ymin, ymax],'--k')        
    plt.xlabel('$L^{PRS}_{WN}$',fontsize=12)
    sensor_name = 'OLCI'
    plt.ylabel('$L^{'+sensor_name+'}_{WN}$',fontsize=12)
    if (xmin<0 or ymin<0):
        plt.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)  
    plt.text(0.05, 0.95, str3+'nm',horizontalalignment='left', fontsize=12,transform=plt.gca().transAxes)

    # save fig
    ofname = sensor_name+'_scatter_diff_ba_zi_'+wl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    #%   scatter plot common matchups
    plt.figure()
    for cnt, line in enumerate(ins_same_station):
        if ins_same_station[cnt] == 'Venise':
            mrk_style = '+r'
        elif ins_same_station[cnt] == 'Gloria':
            mrk_style = '+g'
        elif ins_same_station[cnt] == 'Galata_Platform':
            mrk_style = '+b'
        elif ins_same_station[cnt] == 'Helsinki_Lighthouse':
            mrk_style = '+m'
        elif ins_same_station[cnt] == 'Gustav_Dalen_Tower':
                mrk_style = '+c'
                
        plt.plot(ins_same_zi[cnt], sat_same_zi[cnt],mrk_style)
                
    for cnt, line in enumerate(ins_same_station):
        if ins_same_station[cnt] == 'Venise':
            mrk_style = 'xr'
        elif ins_same_station[cnt] == 'Gloria':
            mrk_style = 'xg'
        elif ins_same_station[cnt] == 'Galata_Platform':
            mrk_style = 'xb'
        elif ins_same_station[cnt] == 'Helsinki_Lighthouse':
            mrk_style = 'xm'
        elif ins_same_station[cnt] == 'Gustav_Dalen_Tower':
            mrk_style = 'xc'
        
        plt.plot(ins_same_ba[cnt], sat_same_ba[cnt],mrk_style)

    plt.axis([min_val, max_val, min_val, max_val])
    plt.gca().set_aspect('equal', adjustable='box')
    # plot 1:1 line
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    plt.plot([xmin,xmax],[ymin, ymax],'--k')        
    plt.xlabel('$L^{PRS}_{WN}$',fontsize=12)
    sensor_name = 'OLCI'
    plt.ylabel('$L^{'+sensor_name+'}_{WN}$',fontsize=12)
    if (xmin<0 or ymin<0):
        plt.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)  
    plt.text(0.05, 0.95, str3+'nm',horizontalalignment='left', fontsize=12,transform=plt.gca().transAxes)

    # save fig
    ofname = sensor_name+'_scatter_diff_same_ba_zi_'+wl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    return  sat_same_zi, sat_same_ba, ins_same_zi, ins_same_ba, ins_same_station
#%% normality test
def plot_normality(data,wl_str):
    # print(data)
    # plot distribution
    kwargs = dict(bins='auto', histtype='step',density=True)
    plt.figure(figsize=(16, 7))
    plt.subplot(1,3,1)
    plt.hist(data, **kwargs)
    x = np.arange(np.min(data),np.max(data),(np.max(data)-np.min(data))/100)
    norm_dist = stats.norm.pdf(x,np.mean(data),np.std(data))
    plt.plot(x,norm_dist)
    plt.legend(['Theoretical Values','Data'],loc='upper left')
    plt.title('Histogram')
    
    # ks test using scipy.stats.kstest        
    D_value0, p_value0 = stats.kstest(data,'norm',args=(np.mean(data), np.std(data)))
    print(f'KS: D: {D_value0:.4f}; p-value: {p_value0:.5f}')

    # ks test using statsmodels.stats.diagnostic.ktest  
    import statsmodels.stats.diagnostic as smd
    D_value1, p_value1 = smd.kstest_normal(data, dist='norm')
    print(f'KS2: D: {D_value1:.4f}; p-value: {p_value1:.5f}')
    
    # jb test using scipy.stats.jarque_bera 
    D_value2, p_value2 = stats.jarque_bera(data)
    print(f'JB: JB: {D_value2:.4f}; p-value: {p_value2:.5f}')
    
    # jb test using statsmodels.stats.stattools.jarque_bera 
    import statsmodels.stats.stattools as sms
    D_value3, p_value3, skew, kurtosis = sms.jarque_bera(data, axis=0)
    print(f'JB2: D: {D_value3:.4f}; p-value: {p_value3:.5f}')
    
    plt.subplot(1,3,2)
    plt.plot(np.sort(data), np.arange(len(data)) / len(data))
    X,CY = cdf(data)
    plt.plot(X, CY)
    plt.title('Comparing CDFs for KS-Test')
    # plt.plot(np.sort(norm_dist), np.arange(len(norm_dist)) / len(norm_dist))
    # plt.plot(np.sort(norm_dist), np.linspace(0, 1, len(norm_dist), endpoint=False))
    # plt.hist(norm_dist,normed=True)
    
    plt.subplot(1,3,3)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Probability Plot)')
    
    plt.suptitle(f'{wl_str}nm; p-value: KS={p_value0:.5f}, KS2={p_value1:.5f}, JB={p_value2:.5f}, JB2={p_value3:.5f}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plot_histogram_and_qq(data, data.mean(), data.std())
    # save fig
    ofname = 'OLCI_norm_test_diff_'+wl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)
    

def cdf(data):
    # Create some test data
    dx = (np.max(data)-np.min(data))/100
    X = np.arange(np.min(data),np.max(data),(np.max(data)-np.min(data))/100)
    Y = stats.norm.pdf(X,np.mean(data),np.std(data))
    
    # Normalize the data to a proper PDF
    Y /= (dx * Y).sum()
    
    # Compute the CDF
    CY = np.cumsum(Y * dx)
    
    # # Plot both
    # plt.plot(X, Y)
    # plt.plot(X, CY, 'r--')
    return X,CY    
#%%
#def main():
#    """business logic for when running this module as the primary one!"""
print('Main Code!')
path_out = os.path.join(path_main,'Figures')
path = os.path.join(path_main,'netcdf_file')
   
#%% Open extract.nc
# extract.nc is created by create_extract.py
# Solar spectral irradiance F0 in uW/cm^2/nm
F0_0412p50 = get_F0(412.5,path_main)  
F0_0442p50 = get_F0(442.5,path_main)
F0_0490p00 = get_F0(490.0,path_main)
F0_0560p00 = get_F0(560.0,path_main)
F0_0665p00 = get_F0(665.0,path_main)  

# from AERONET-OC: Lwn in [mW/(cm^2 sr um)]

# Zibordi: initialization
mu_Lwn_0412p50_fq_ins_zi = []
mu_Lwn_0442p50_fq_ins_zi = []
mu_Lwn_0490p00_fq_ins_zi = []
mu_Lwn_0560p00_fq_ins_zi = []
mu_Lwn_0665p00_fq_ins_zi = []

mu_Lwn_0412p50_fq_sat_zi = []
mu_Lwn_0442p50_fq_sat_zi = []
mu_Lwn_0490p00_fq_sat_zi = []
mu_Lwn_0560p00_fq_sat_zi = []
mu_Lwn_0665p00_fq_sat_zi = []

mu_Lwn_0412p50_fq_ins_zi_station = []
mu_Lwn_0442p50_fq_ins_zi_station = []
mu_Lwn_0490p00_fq_ins_zi_station = []
mu_Lwn_0560p00_fq_ins_zi_station = []
mu_Lwn_0665p00_fq_ins_zi_station = []

mu_Lwn_0412p50_fq_sat_zi_stop_time = []
mu_Lwn_0442p50_fq_sat_zi_stop_time = []
mu_Lwn_0490p00_fq_sat_zi_stop_time = []
mu_Lwn_0560p00_fq_sat_zi_stop_time = []
mu_Lwn_0665p00_fq_sat_zi_stop_time = []    

mu_Lwn_0412p50_fq_ins_zi_time = []
mu_Lwn_0442p50_fq_ins_zi_time = []
mu_Lwn_0490p00_fq_ins_zi_time = []
mu_Lwn_0560p00_fq_ins_zi_time = []
mu_Lwn_0665p00_fq_ins_zi_time = [] 

# Bailey and Werdell: initialization
mu_Lwn_0412p50_fq_ins_ba = []
mu_Lwn_0442p50_fq_ins_ba = []
mu_Lwn_0490p00_fq_ins_ba = []
mu_Lwn_0560p00_fq_ins_ba = []
mu_Lwn_0665p00_fq_ins_ba = []

mu_Lwn_0412p50_fq_sat_ba = []
mu_Lwn_0442p50_fq_sat_ba = []
mu_Lwn_0490p00_fq_sat_ba = []
mu_Lwn_0560p00_fq_sat_ba = []
mu_Lwn_0665p00_fq_sat_ba = []

mu_Lwn_0412p50_fq_ins_ba_station = []
mu_Lwn_0442p50_fq_ins_ba_station = []
mu_Lwn_0490p00_fq_ins_ba_station = []
mu_Lwn_0560p00_fq_ins_ba_station = []
mu_Lwn_0665p00_fq_ins_ba_station = [] 

mu_Lwn_0412p50_fq_sat_ba_stop_time = []
mu_Lwn_0442p50_fq_sat_ba_stop_time = []
mu_Lwn_0490p00_fq_sat_ba_stop_time = []
mu_Lwn_0560p00_fq_sat_ba_stop_time = []
mu_Lwn_0665p00_fq_sat_ba_stop_time = []  

mu_Lwn_0412p50_fq_ins_ba_time = []
mu_Lwn_0442p50_fq_ins_ba_time = []
mu_Lwn_0490p00_fq_ins_ba_time = []
mu_Lwn_0560p00_fq_ins_ba_time = []
mu_Lwn_0665p00_fq_ins_ba_time = []        


station_list_main = ['Venise','Galata_Platform','Gloria','Helsinki_Lighthouse','Gustav_Dalen_Tower']
# station_list_main = ['Venise','Galata_Platform','Gloria']
# station_list_main = ['Helsinki_Lighthouse','Gustav_Dalen_Tower']
# station_list_main = ['Venise']

# for counting potential matchups
pot_mu_cnt_zi = 0
pot_mu_cnt_ba = 0

pot_mu_cnt_ba_Venise = 0
pot_mu_cnt_ba_Gloria = 0
pot_mu_cnt_ba_Galata_Platform = 0
pot_mu_cnt_ba_Helsinki_Lighthouse = 0
pot_mu_cnt_ba_Gustav_Dalen_Tower = 0

pot_mu_cnt_zi_Venise = 0
pot_mu_cnt_zi_Gloria = 0
pot_mu_cnt_zi_Galata_Platform = 0
pot_mu_cnt_zi_Helsinki_Lighthouse = 0
pot_mu_cnt_zi_Gustav_Dalen_Tower = 0

# for counting rejection reasons
rej_cvs_mu_cnt_zi = 0
rej_cvs_mu_cnt_zi_Venise = 0
rej_cvs_mu_cnt_zi_Gloria = 0
rej_cvs_mu_cnt_zi_Galata_Platform = 0
rej_cvs_mu_cnt_zi_Helsinki_Lighthouse = 0
rej_cvs_mu_cnt_zi_Gustav_Dalen_Tower = 0
rej_cvL_mu_cnt_zi = 0
rej_cvL_mu_cnt_zi_Venise = 0
rej_cvL_mu_cnt_zi_Gloria = 0
rej_cvL_mu_cnt_zi_Galata_Platform = 0
rej_cvL_mu_cnt_zi_Helsinki_Lighthouse = 0
rej_cvL_mu_cnt_zi_Gustav_Dalen_Tower = 0
rej_cvA_mu_cnt_zi = 0
rej_cvA_mu_cnt_zi_Venise = 0
rej_cvA_mu_cnt_zi_Gloria = 0
rej_cvA_mu_cnt_zi_Galata_Platform = 0
rej_cvA_mu_cnt_zi_Helsinki_Lighthouse = 0
rej_cvA_mu_cnt_zi_Gustav_Dalen_Tower = 0
rej_ang_mu_cnt_zi = 0
rej_ang_mu_cnt_zi_Venise = 0
rej_ang_mu_cnt_zi_Gloria = 0
rej_ang_mu_cnt_zi_Galata_Platform = 0
rej_ang_mu_cnt_zi_Helsinki_Lighthouse = 0
rej_ang_mu_cnt_zi_Gustav_Dalen_Tower = 0
rej_sza_mu_cnt_zi = 0
rej_sza_mu_cnt_zi_Venise = 0
rej_sza_mu_cnt_zi_Gloria = 0
rej_sza_mu_cnt_zi_Galata_Platform = 0
rej_sza_mu_cnt_zi_Helsinki_Lighthouse = 0
rej_sza_mu_cnt_zi_Gustav_Dalen_Tower = 0
rej_vza_mu_cnt_zi = 0
rej_vza_mu_cnt_zi_Venise = 0
rej_vza_mu_cnt_zi_Gloria = 0
rej_vza_mu_cnt_zi_Galata_Platform = 0
rej_vza_mu_cnt_zi_Helsinki_Lighthouse = 0
rej_vza_mu_cnt_zi_Gustav_Dalen_Tower = 0
rej_inv_mu_cnt_zi = 0
rej_inv_mu_cnt_zi_Venise = 0
rej_inv_mu_cnt_zi_Gloria = 0
rej_inv_mu_cnt_zi_Galata_Platform = 0
rej_inv_mu_cnt_zi_Helsinki_Lighthouse = 0
rej_inv_mu_cnt_zi_Gustav_Dalen_Tower = 0
rej_cvs_mu_cnt_ba = 0
rej_cvs_mu_cnt_ba_Venise = 0
rej_cvs_mu_cnt_ba_Gloria = 0
rej_cvs_mu_cnt_ba_Galata_Platform = 0
rej_cvs_mu_cnt_ba_Helsinki_Lighthouse = 0
rej_cvs_mu_cnt_ba_Gustav_Dalen_Tower = 0
rej_ang_mu_cnt_ba = 0
rej_ang_mu_cnt_ba_Venise = 0
rej_ang_mu_cnt_ba_Gloria = 0
rej_ang_mu_cnt_ba_Galata_Platform = 0
rej_ang_mu_cnt_ba_Helsinki_Lighthouse = 0
rej_ang_mu_cnt_ba_Gustav_Dalen_Tower = 0
rej_sza_mu_cnt_ba = 0
rej_sza_mu_cnt_ba_Venise = 0
rej_sza_mu_cnt_ba_Gloria = 0
rej_sza_mu_cnt_ba_Galata_Platform = 0
rej_sza_mu_cnt_ba_Helsinki_Lighthouse = 0
rej_sza_mu_cnt_ba_Gustav_Dalen_Tower = 0
rej_vza_mu_cnt_ba = 0
rej_vza_mu_cnt_ba_Venise = 0
rej_vza_mu_cnt_ba_Gloria = 0
rej_vza_mu_cnt_ba_Galata_Platform = 0
rej_vza_mu_cnt_ba_Helsinki_Lighthouse = 0
rej_vza_mu_cnt_ba_Gustav_Dalen_Tower = 0
rej_inv_mu_cnt_ba = 0
rej_inv_mu_cnt_ba_Venise = 0
rej_inv_mu_cnt_ba_Gloria = 0
rej_inv_mu_cnt_ba_Galata_Platform = 0
rej_inv_mu_cnt_ba_Helsinki_Lighthouse = 0
rej_inv_mu_cnt_ba_Gustav_Dalen_Tower = 0

mu_cnt_zi = 0
mu_cnt2_zi = 0
mu_cnt_ba = 0
mu_cnt2_ba = 0

idx_medianCV = []

dt_mu_zi = [] # delta time for the matchups zi
dt_mu_ba = [] # delta time for the matchups ba

olci_wl_list = [412.5,442.5,490.0,560.0,665.0]

map_valid_pxs_ba = np.zeros([len(station_list_main),len(olci_wl_list),5,5],dtype=int)

number_used_pixels = np.empty([0,len(olci_wl_list)],dtype=int)

columns = ['station','sat_datetime','insitu_datetime','vza','sza',\
         'BW06_MU','BW06_l2_mask',\
         'BW06: rhow_412_box','BW06: rho_412_filt_mean',\
         'BW06: rhow_442_box','BW06: rho_442_filt_mean',\
         'BW06: rhow_490_box','BW06: rho_490_filt_mean',\
         'BW06: rhow_560_box','BW06: rho_560_filt_mean',\
         'BW06: rhow_665_box','BW06: rho_665_filt_mean',\
         'BW06: MedianCV','BW06: Nfilt_560','BW06: NGP','BW06: NTP',\
         'BW06: CV_rhow_412.5','BW06: CV_rhow_442.5',\
         'BW06: CV_rhow_490.0','BW06: CV_rhow_560.0',\
         'BW06: CV_rhow_665.0','BW06: CV_aot_865.5',\
         'BW06: MedianCV_band_idx',\
         'Z09_MU','Z09_l2_mask',\
         'Z09: rhow_412_box','Z09: rho_412_mean',\
         'Z09: rhow_442_box','Z09: rho_442_mean',\
         'Z09: rhow_490_box','Z09: rho_490_mean',\
         'Z09: rhow_560_box','Z09: rho_560_mean',\
         'Z09: rhow_665_box','Z09: rho_665_mean',\
         'Z09: CV_560','Z09: CV_865p5']
df_matchups = pd.DataFrame(columns=columns)

# CVs = [CV_filtered_rhow_0412p50, CV_filtered_rhow_0442p50,\
#                                      CV_filtered_rhow_0490p00, CV_filtered_rhow_0560p00,\
#                                      CV_filtered_AOT_0865p50]
columns = ['station_idx','rhow_412.5','rhow_442.5','rhow_490.0','rhow_560.0','rhow_665.0',\
           'aot_865.5','MedianCV','MedianCV_band_idx','sat_datetime']
df_CVs_ba = pd.DataFrame(columns=columns)

columns = ['station_idx','Lwn_560.0','aot_865.5','sat_datetime']
df_CVs_zi = pd.DataFrame(columns=columns)

data_masked_Gloria = []
data_masked_Gloria_value = []

def open_insitu(station):
    
    #    filename = station_name+'_20V3_20190927_20200110.nc'
    filename = station+'_20V3_20160426_20200206.nc'
    # filename = station_name+'_20V3_20180622_20180822.nc'
    filename_insitu = os.path.join(path,filename)
    if not os.path.exists(filename_insitu):
        print('File does not exist')
        
    nc_f0 = Dataset(filename_insitu,'r')

    Time = nc_f0.variables['Time'][:]
    Level = nc_f0.variables['Level'][:]
    Julian_day = nc_f0.variables['Julian_day'][:]
    Exact_wavelengths = nc_f0.variables['Exact_wavelengths'][:]
    Lwn_fonQ = nc_f0.variables['Lwn_fonQ'][:]

    nc_f0.close()
    
    return Time, Level, Julian_day, Exact_wavelengths, Lwn_fonQ

for station_idx in range(len(station_list_main)):  

    station_name = station_list_main[station_idx]

    Time, Level, Julian_day, Exact_wavelengths, Lwn_fonQ = \
        open_insitu(station_name)
    
    day_vec    = np.array([float(Time[i].replace(' ',':').split(':')[0]) for i in range(0,len(Time))])
    month_vec  = np.array([float(Time[i].replace(' ',':').split(':')[1]) for i in range(0,len(Time))])
    year_vec   = np.array([float(Time[i].replace(' ',':').split(':')[2]) for i in range(0,len(Time))])
    hour_vec   = np.array([float(Time[i].replace(' ',':').split(':')[3]) for i in range(0,len(Time))])
    minute_vec = np.array([float(Time[i].replace(' ',':').split(':')[4]) for i in range(0,len(Time))])
    second_vec = np.array([float(Time[i].replace(' ',':').split(':')[5]) for i in range(0,len(Time))])

    Julian_day_vec =np.array([float(Julian_day[i]) for i in range(0,len(Time))])
    date_format = "%d:%m:%Y %H:%M:%S"
    ins_time = np.array([datetime.strptime(Time[i], date_format) for i in range(0,len(Time))])

    doy_vec = np.array([int(float(Julian_day[i])) for i in range(0,len(Time))])

    if create_list_flag:
        create_OLCI_list(path_main,Time,year_vec,month_vec,doy_vec,day_vec)

    path_to_list = os.path.join(path_main,'data','output','extract_list_'+station_name+'.txt')
    # create list of extract by station
    cmd = 'find '+os.path.join(path_main,'data','output')+' -name "extract_'+station_name+'.nc"|sort|uniq >'+path_to_list
    (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
    with open(path_to_list,'r') as file:
        for cnt, line in enumerate(file):  
            # print('----------------------------')
            # print('line '+str(cnt))
            
            # Initialization
            sat_stop_time = np.nan
            vza = np.nan
            sza = np.nan
            BW06_MU = np.nan
            flags_mask_ba = np.nan
            MedianCV = np.nan
            rhow_0560p00_fq_box_ba = np.nan
            NGP = np.nan
            NTP = np.nan
            CV_filtered_rhow_0412p50 = np.nan
            CV_filtered_rhow_0442p50 = np.nan
            CV_filtered_rhow_0490p00 = np.nan
            CV_filtered_rhow_0560p00 = np.nan
            CV_filtered_rhow_0665p00 = np.nan
            CV_filtered_AOT_0865p50 = np.nan
            rhow_0412p50_fq_box_ba = np.nan
            rhow_0442p50_fq_box_ba = np.nan
            rhow_0490p00_fq_box_ba = np.nan
            rhow_0560p00_fq_box_ba = np.nan
            rhow_0665p00_fq_box_ba = np.nan
            mean_filtered_rhow_0412p50 = np.nan
            mean_filtered_rhow_0442p50 = np.nan
            mean_filtered_rhow_0490p00 = np.nan
            mean_filtered_rhow_0560p00 = np.nan
            mean_filtered_rhow_0665p00 = np.nan
        
            idx_m = np.nan
            Z09_MU = np.nan
            flags_mask_zi = np.nan
            rhow_0412p50_fq_box_zi = np.nan
            rhow_0442p50_fq_box_zi = np.nan
            rhow_0490p00_fq_box_zi = np.nan
            rhow_0560p00_fq_box_zi = np.nan
            rhow_0665p00_fq_box_zi = np.nan
            Lwn_560_CV = np.nan
            AOT_0865p50_CV = np.nan
            
            insitu_Lwn_fonQ_zi = []
            insitu_Lwn_fonQ_ba = []
            
            Z09_MU = False
            BW06_MU = False
            
            year_str = line.split('/')[-3]
            doy_str = line.split('/')[-2]       
            nc_f1 = Dataset(line[:-1],'r')
            
            date_format = "%Y-%m-%dT%H:%M:%S.%fZ" 
            sat_start_time = datetime.strptime(nc_f1.start_time, date_format)
            sat_stop_time = datetime.strptime(nc_f1.stop_time, date_format)
            
            # from sat
            rhow_0412p50_fq = nc_f1.variables['rhow_0412p50_fq'][:]
            rhow_0442p50_fq = nc_f1.variables['rhow_0442p50_fq'][:]
            rhow_0490p00_fq = nc_f1.variables['rhow_0490p00_fq'][:]
            rhow_0560p00_fq = nc_f1.variables['rhow_0560p00_fq'][:]
            rhow_0665p00_fq = nc_f1.variables['rhow_0665p00_fq'][:]
                    
            WQSF = nc_f1.variables['WQSF'][:]
            AOT_0865p50 = nc_f1.variables['AOT_0865p50'][:]
            sza = nc_f1.variables['sza_value'][:]
            vza = nc_f1.variables['vza_value'][:]

            #############################################################################                
            # Zibordi et al. 2009 #######################################################
            delta_time = 2# float in hours       
            time_diff = ins_time - sat_stop_time
            dt_hour = [i.total_seconds()/(60*60) for i in time_diff] # time diffence between in situ measurements and sat in hours
            idx_min = np.argmin(np.abs(dt_hour))
            matchup_idx_vec = np.abs(dt_hour) <= delta_time 

            # in situ data
            # olci_wl_list = [412.5,442.5,490.0,560.0,665.0]
            if Exact_wavelengths[idx_min,13] != -999:
                idx_560 = 13
            elif Exact_wavelengths[idx_min,12] != -999:
                idx_560 = 12
            else: 
                idx_560 = 11

            insitu_Lwn_fonQ_zi = [Lwn_fonQ[idx_min,3],Lwn_fonQ[idx_min,5],Lwn_fonQ[idx_min,6],Lwn_fonQ[idx_min,idx_560],Lwn_fonQ[idx_min,15]]
            Exact_wavelengths_zi = Exact_wavelengths[idx_min,:]
            insitu_Lwn_fonQ_zi_all = Lwn_fonQ[idx_min,:]
    
            nday = sum(matchup_idx_vec)
            if nday >=1:
                print('----------------------------')
                print('line '+str(cnt))
                print('--Zibordi et al. 2009')
                print(str(nday)+' matchups per '+year_str+' '+doy_str)
    #            print(Lwn_fonQ[idx_min,:])
    #            print(Exact_wavelengths[idx_min,:])
                
                center_px = int(len(rhow_0412p50_fq)/2 + 0.5)
                size_box = 3
                start_idx_x = int(center_px-int(size_box/2))
                stop_idx_x = int(center_px+int(size_box/2)+1)
                start_idx_y = int(center_px-int(size_box/2))
                stop_idx_y = int(center_px+int(size_box/2)+1)

                flags_mask_zi = OLCI_flags.create_mask(WQSF[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
                print('flags_mask_zi:')
                print(flags_mask_zi)

                rhow_0412p50_fq_box_zi = rhow_0412p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0442p50_fq_box_zi = rhow_0442p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0490p00_fq_box_zi = rhow_0490p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0560p00_fq_box_zi = rhow_0560p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0665p00_fq_box_zi = rhow_0665p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                AOT_0865p50_box_zi     = AOT_0865p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]

                # apply L2 mask
                rhow_0412p50_fq_box_zi = ma.masked_array(rhow_0412p50_fq_box_zi,flags_mask_zi) 
                rhow_0442p50_fq_box_zi = ma.masked_array(rhow_0442p50_fq_box_zi,flags_mask_zi) 
                rhow_0490p00_fq_box_zi = ma.masked_array(rhow_0490p00_fq_box_zi,flags_mask_zi) 
                rhow_0560p00_fq_box_zi = ma.masked_array(rhow_0560p00_fq_box_zi,flags_mask_zi) 
                rhow_0665p00_fq_box_zi = ma.masked_array(rhow_0665p00_fq_box_zi,flags_mask_zi) 
                AOT_0865p50_box_zi     = ma.masked_array(AOT_0865p50_box_zi    ,flags_mask_zi) 


                # if nan, change mask
                rhow_0412p50_fq_box_zi = ma.masked_invalid(rhow_0412p50_fq_box_zi)
                rhow_0442p50_fq_box_zi = ma.masked_invalid(rhow_0442p50_fq_box_zi)
                rhow_0490p00_fq_box_zi = ma.masked_invalid(rhow_0490p00_fq_box_zi)
                rhow_0560p00_fq_box_zi = ma.masked_invalid(rhow_0560p00_fq_box_zi)
                rhow_0665p00_fq_box_zi = ma.masked_invalid(rhow_0665p00_fq_box_zi)
                AOT_0865p50_box_zi = ma.masked_invalid(AOT_0865p50_box_zi)

                # from AERONET-OC V3 file
                # 0         1         2         3         4         5         6         7         8         9         10        11        12        13        14        15        16        17        18        19        20        21        22 
                # Lw[340nm],Lw[380nm],Lw[400nm],Lw[412nm],Lw[440nm],Lw[443nm],Lw[490nm],Lw[500nm],Lw[510nm],Lw[531nm],Lw[532nm],Lw[551nm],Lw[555nm],Lw[560nm],Lw[620nm],Lw[667nm],Lw[675nm],Lw[681nm],Lw[709nm],Lw[779nm],Lw[865nm],Lw[870nm],Lw[1020nm]    
                # -999,     -999,     -999,     412,      -999,     441.8,    488.5,    -999,     -999,     -999,     530.3,    551,      -999,     -999,     -999,     667.9,    -999,     -999,     -999,     -999,     -999,     870.8,    1020.5,
 
                # cv
                Lwn_560 = rhow_0560p00_fq_box_zi*F0_0560p00/np.pi
                Lwn_560_CV = np.abs(Lwn_560.std()/Lwn_560.mean())                    
                AOT_0865p50_CV = np.abs(AOT_0865p50_box_zi.std()/AOT_0865p50_box_zi.mean())

                # to count potential matchups total and per stations
                pot_mu_cnt_zi += 1
                if station_name == 'Venise':
                    pot_mu_cnt_zi_Venise += 1
                elif station_name == 'Gloria':
                    pot_mu_cnt_zi_Gloria += 1
                elif station_name == 'Galata_Platform':
                    pot_mu_cnt_zi_Galata_Platform += 1
                elif station_name == 'Helsinki_Lighthouse':
                    pot_mu_cnt_zi_Helsinki_Lighthouse += 1
                elif station_name == 'Gustav_Dalen_Tower':
                    pot_mu_cnt_zi_Gustav_Dalen_Tower += 1    

                # count rejections
                if Lwn_560_CV > 0.2 or AOT_0865p50_CV > 0.2:
                    rej_cvs_mu_cnt_zi += 1
                    if station_name == 'Venise':
                        rej_cvs_mu_cnt_zi_Venise += 1
                    elif station_name == 'Gloria':
                        rej_cvs_mu_cnt_zi_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_cvs_mu_cnt_zi_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_cvs_mu_cnt_zi_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_cvs_mu_cnt_zi_Gustav_Dalen_Tower += 1 

                if Lwn_560_CV > 0.2:
                    rej_cvL_mu_cnt_zi += 1
                    if station_name == 'Venise':
                        rej_cvL_mu_cnt_zi_Venise += 1
                    elif station_name == 'Gloria':
                        rej_cvL_mu_cnt_zi_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_cvL_mu_cnt_zi_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_cvL_mu_cnt_zi_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_cvL_mu_cnt_zi_Gustav_Dalen_Tower += 1 

                if AOT_0865p50_CV > 0.2:
                    rej_cvA_mu_cnt_zi += 1
                    if station_name == 'Venise':
                        rej_cvA_mu_cnt_zi_Venise += 1
                    elif station_name == 'Gloria':
                        rej_cvA_mu_cnt_zi_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_cvA_mu_cnt_zi_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_cvA_mu_cnt_zi_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_cvA_mu_cnt_zi_Gustav_Dalen_Tower += 1 

                if sza>70 or vza>56:
                    rej_ang_mu_cnt_zi += 1
                    if station_name == 'Venise':
                        rej_ang_mu_cnt_zi_Venise += 1
                    elif station_name == 'Gloria':
                        rej_ang_mu_cnt_zi_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_ang_mu_cnt_zi_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_ang_mu_cnt_zi_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_ang_mu_cnt_zi_Gustav_Dalen_Tower += 1 

                if sza>70:
                    rej_sza_mu_cnt_zi += 1
                    if station_name == 'Venise':
                        rej_sza_mu_cnt_zi_Venise += 1
                    elif station_name == 'Gloria':
                        rej_sza_mu_cnt_zi_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_sza_mu_cnt_zi_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_sza_mu_cnt_zi_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_sza_mu_cnt_zi_Gustav_Dalen_Tower += 1   

                if vza>56:
                    rej_vza_mu_cnt_zi += 1
                    if station_name == 'Venise':
                        rej_vza_mu_cnt_zi_Venise += 1
                    elif station_name == 'Gloria':
                        rej_vza_mu_cnt_zi_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_vza_mu_cnt_zi_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_vza_mu_cnt_zi_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_vza_mu_cnt_zi_Gustav_Dalen_Tower += 1                               

                if flags_mask_zi.any() or ((rhow_0412p50_fq_box_zi.mask.any() or np.isnan(rhow_0412p50_fq_box_zi).any())\
                        or (rhow_0442p50_fq_box_zi.mask.any() or np.isnan(rhow_0442p50_fq_box_zi).any())\
                        or (rhow_0490p00_fq_box_zi.mask.any() or np.isnan(rhow_0490p00_fq_box_zi).any())\
                        or (rhow_0560p00_fq_box_zi.mask.any() or np.isnan(rhow_0560p00_fq_box_zi).any())\
                        or (rhow_0665p00_fq_box_zi.mask.any() or np.isnan(rhow_0665p00_fq_box_zi).any())):
                    rej_inv_mu_cnt_zi += 1
                    if station_name == 'Venise':
                        rej_inv_mu_cnt_zi_Venise += 1
                    elif station_name == 'Gloria':
                        rej_inv_mu_cnt_zi_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_inv_mu_cnt_zi_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_inv_mu_cnt_zi_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_inv_mu_cnt_zi_Gustav_Dalen_Tower += 1 

                # create matchup
                if (sza<=70 and vza<=56) and (not flags_mask_zi.any()) and (Lwn_560_CV <= 0.2 and AOT_0865p50_CV <= 0.2): # if any of the pixels if flagged, Fails validation criteria because all have to be valid in Zibordi 2018
                    dt_mu_zi.append(dt_hour[idx_min])
                    mu_cnt_zi += 1    
                    # if any is invalid, do not calculated matchup
                    if not ((rhow_0412p50_fq_box_zi.mask.any() or np.isnan(rhow_0412p50_fq_box_zi).any())\
                        or (rhow_0442p50_fq_box_zi.mask.any() or np.isnan(rhow_0442p50_fq_box_zi).any())\
                        or (rhow_0490p00_fq_box_zi.mask.any() or np.isnan(rhow_0490p00_fq_box_zi).any())\
                        or (rhow_0560p00_fq_box_zi.mask.any() or np.isnan(rhow_0560p00_fq_box_zi).any())\
                        or (rhow_0665p00_fq_box_zi.mask.any() or np.isnan(rhow_0665p00_fq_box_zi).any())):

                        mu_cnt2_zi += 1
                        Z09_MU = True
                    # Rrs 0412p50
                    # print('412.5')
                    # if not (rhow_0412p50_fq_box_zi.mask.any() == True or np.isnan(rhow_0412p50_fq_box_zi).any() == True):
                    #     print('At least one element in sat product is invalid!')
                    # else:
                        mu_Lwn_0412p50_fq_sat_zi.append(rhow_0412p50_fq_box_zi.mean()*F0_0412p50/np.pi)
                        mu_Lwn_0412p50_fq_ins_zi.append(insitu_Lwn_fonQ_zi[olci_wl_list.index(412.5)]) # 412,
                        mu_Lwn_0412p50_fq_ins_zi_station.append(station_name)
                        mu_Lwn_0412p50_fq_sat_zi_stop_time.append(sat_stop_time)
                        mu_Lwn_0412p50_fq_ins_zi_time.append(ins_time[idx_min])
                        
                    # Rrs 0442p50
                    # print('442.5')
                    # if not (rhow_0442p50_fq_box_zi.mask.any() == True or np.isnan(rhow_0442p50_fq_box_zi).any() == True):
                        # print('At least one element in sat product is invalid!')
                    # else:
                        mu_Lwn_0442p50_fq_sat_zi.append(rhow_0442p50_fq_box_zi.mean()*F0_0442p50/np.pi)
                        mu_Lwn_0442p50_fq_ins_zi.append(insitu_Lwn_fonQ_zi[olci_wl_list.index(442.5)]) # 441.8
                        mu_Lwn_0442p50_fq_ins_zi_station.append(station_name)
                        mu_Lwn_0442p50_fq_sat_zi_stop_time.append(sat_stop_time)
                        mu_Lwn_0442p50_fq_ins_zi_time.append(ins_time[idx_min])
                        
                    # Rrs 0490p00
                    # print('490.0')
                    # if not (rhow_0490p00_fq_box_zi.mask.any() == True or np.isnan(rhow_0490p00_fq_box_zi).any() == True):
                        # print('At least one element in sat product is invalid!')
                    # else:
                        mu_Lwn_0490p00_fq_sat_zi.append(rhow_0490p00_fq_box_zi.mean()*F0_0490p00/np.pi)
                        mu_Lwn_0490p00_fq_ins_zi.append(insitu_Lwn_fonQ_zi[olci_wl_list.index(490.0)]) # 488.5
                        mu_Lwn_0490p00_fq_ins_zi_station.append(station_name)
                        mu_Lwn_0490p00_fq_sat_zi_stop_time.append(sat_stop_time)
                        mu_Lwn_0490p00_fq_ins_zi_time.append(ins_time[idx_min])
                        
                    # Rrs 0560p00
                    # print('560.0')
                    # if not (rhow_0560p00_fq_box_zi.mask.any() == True or np.isnan(rhow_0560p00_fq_box_zi).any() == True):
                        # print('At least one element in sat product is invalid!')
                    # else:
                        mu_Lwn_0560p00_fq_sat_zi.append(rhow_0560p00_fq_box_zi.mean()*F0_0560p00/np.pi)
                        mu_Lwn_0560p00_fq_ins_zi.append(insitu_Lwn_fonQ_zi[olci_wl_list.index(560.0)]) # 551,
                        mu_Lwn_0560p00_fq_ins_zi_station.append(station_name)
                        mu_Lwn_0560p00_fq_sat_zi_stop_time.append(sat_stop_time)
                        mu_Lwn_0560p00_fq_ins_zi_time.append(ins_time[idx_min])
                        
                    # Rrs 0665p00
                    # print('665.0')
                    # if not (rhow_0665p00_fq_box_zi.mask.any() == True or np.isnan(rhow_0665p00_fq_box_zi).any() == True):
                        # print('At least one element in sat product is invalid!')
                    # else:
                        mu_Lwn_0665p00_fq_sat_zi.append(rhow_0665p00_fq_box_zi.mean()*F0_0665p00/np.pi)
                        mu_Lwn_0665p00_fq_ins_zi.append(insitu_Lwn_fonQ_zi[olci_wl_list.index(665.0)]) # 667.9    
                        mu_Lwn_0665p00_fq_ins_zi_station.append(station_name)
                        mu_Lwn_0665p00_fq_sat_zi_stop_time.append(sat_stop_time)
                        mu_Lwn_0665p00_fq_ins_zi_time.append(ins_time[idx_min])
                    
                    # to analyse CV
                    
                    df_CVs_zi = df_CVs_zi.append({'station_idx':station_idx,'Lwn_560.0':Lwn_560_CV,\
                                    'aot_865.5':AOT_0865p50_CV,'sat_datetime':sat_stop_time},ignore_index=True)
                    # CVs_zi[station_idx,:] = np.append(CVs_zi[station_idx,:],[[Lwn_560_CV,AOT_0865p50_CV]],axis=0)
                    # CVs_zi_time.append(sat_stop_time)                        
        #         else:
        #             print('CV exceeds criteria: CV[Lwn(560)]='+str(Lwn_560_CV)+'; CV[AOT(865.5)]='+str(AOT_0865p50_CV))
        #     else:
        #         print('Angles exceeds criteria: sza='+str(sza)+'; vza='+str(vza)+'; OR some pixels are flagged!')              
        # else:
        #     print('Not matchups per '+year_str+' '+doy_str)
            #############################################################################    
            # Bailey and Werdell 2006 ###################################################
            delta_time = 3# float in hours       
            time_diff = ins_time - sat_stop_time
            dt_hour = [i.total_seconds()/(60*60) for i in time_diff] # time diffence between in situ measurements and sat in hours
            idx_min = np.argmin(np.abs(dt_hour))
            matchup_idx_vec = np.abs(dt_hour) <= delta_time 

            # in situ data
            # olci_wl_list = [412.5,442.5,490.0,560.0,665.0]
            if Exact_wavelengths[idx_min,13] != -999:
                idx_560 = 13
            elif Exact_wavelengths[idx_min,12] != -999:
                idx_560 = 12
            else: 
                idx_560 = 11

            insitu_Lwn_fonQ_ba     = [Lwn_fonQ[idx_min,3],Lwn_fonQ[idx_min,5],Lwn_fonQ[idx_min,6],Lwn_fonQ[idx_min,idx_560],Lwn_fonQ[idx_min,15]]
            Exact_wavelengths_ba   = Exact_wavelengths[idx_min,:]
            insitu_Lwn_fonQ_ba_all = Lwn_fonQ[idx_min,:]

            nday = sum(matchup_idx_vec)
            if nday >=1:                
                print('--Bailey and Werdell 2006')
                print(str(nday)+' matchups per '+year_str+' '+doy_str)
     #           print(Lwn_fonQ[idx_min,:])
     #           print(Exact_wavelengths[idx_min,:])
                
                center_px = int(len(rhow_0412p50_fq)/2 + 0.5)
                size_box = 5
                NTP = size_box*size_box # Number Total Pixels, excluding land pixels, Bailey and Werdell 2006
                start_idx_x = int(center_px-int(size_box/2))
                stop_idx_x = int(center_px+int(size_box/2)+1)
                start_idx_y = int(center_px-int(size_box/2))
                stop_idx_y = int(center_px+int(size_box/2)+1)
                rhow_0412p50_fq_box_ba = rhow_0412p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0442p50_fq_box_ba = rhow_0442p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0490p00_fq_box_ba = rhow_0490p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0560p00_fq_box_ba = rhow_0560p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0665p00_fq_box_ba = rhow_0665p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                AOT_0865p50_box_ba = AOT_0865p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                
                print('rhow_0412p50_fq_box_ba:')
                print(rhow_0412p50_fq_box_ba)
                print('rhow_0412p50_fq_box_ba.mask:')
                print(rhow_0412p50_fq_box_ba.mask)
                
                flags_mask_ba = OLCI_flags.create_mask(WQSF[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
                print('flags_mask_ba:')
                print(flags_mask_ba)
                
                NGP = np.count_nonzero(flags_mask_ba == 0) # Number Good Pixels, Bailey and Werdell 2006

                # apply L2 mask
                rhow_0412p50_fq_box_ba = ma.masked_array(rhow_0412p50_fq_box_ba,flags_mask_ba) 
                rhow_0442p50_fq_box_ba = ma.masked_array(rhow_0442p50_fq_box_ba,flags_mask_ba) 
                rhow_0490p00_fq_box_ba = ma.masked_array(rhow_0490p00_fq_box_ba,flags_mask_ba) 
                rhow_0560p00_fq_box_ba = ma.masked_array(rhow_0560p00_fq_box_ba,flags_mask_ba) 
                rhow_0665p00_fq_box_ba = ma.masked_array(rhow_0665p00_fq_box_ba,flags_mask_ba) 
                AOT_0865p50_box_ba     = ma.masked_array(AOT_0865p50_box_ba    ,flags_mask_ba) 
                
                # if nan, change mask
                rhow_0412p50_fq_box_ba = ma.masked_invalid(rhow_0412p50_fq_box_ba)
                rhow_0442p50_fq_box_ba = ma.masked_invalid(rhow_0442p50_fq_box_ba)
                rhow_0490p00_fq_box_ba = ma.masked_invalid(rhow_0490p00_fq_box_ba)
                rhow_0560p00_fq_box_ba = ma.masked_invalid(rhow_0560p00_fq_box_ba)
                rhow_0665p00_fq_box_ba = ma.masked_invalid(rhow_0665p00_fq_box_ba)
                AOT_0865p50_box_ba = ma.masked_invalid(AOT_0865p50_box_ba)

                NGP_rhow_0412p50 = np.count_nonzero(rhow_0412p50_fq_box_ba.mask == 0)
                NGP_rhow_0442p50 = np.count_nonzero(rhow_0442p50_fq_box_ba.mask == 0)
                NGP_rhow_0490p00 = np.count_nonzero(rhow_0490p00_fq_box_ba.mask == 0)
                NGP_rhow_0560p00 = np.count_nonzero(rhow_0560p00_fq_box_ba.mask == 0)
                NGP_rhow_0665p00 = np.count_nonzero(rhow_0665p00_fq_box_ba.mask == 0)
                NGP_AOT_0865p50 = np.count_nonzero(AOT_0865p50_box_ba.mask == 0)

                mean_unfiltered_rhow_0412p50 = rhow_0412p50_fq_box_ba.mean()
                mean_unfiltered_rhow_0442p50 = rhow_0442p50_fq_box_ba.mean()
                mean_unfiltered_rhow_0490p00 = rhow_0490p00_fq_box_ba.mean()
                mean_unfiltered_rhow_0560p00 = rhow_0560p00_fq_box_ba.mean()
                mean_unfiltered_rhow_0665p00 = rhow_0665p00_fq_box_ba.mean()
                mean_unfiltered_AOT_0865p50 = AOT_0865p50_box_ba.mean()

                std_unfiltered_rhow_0412p50 = rhow_0412p50_fq_box_ba.std()
                std_unfiltered_rhow_0442p50 = rhow_0442p50_fq_box_ba.std()
                std_unfiltered_rhow_0490p00 = rhow_0490p00_fq_box_ba.std()
                std_unfiltered_rhow_0560p00 = rhow_0560p00_fq_box_ba.std()
                std_unfiltered_rhow_0665p00 = rhow_0665p00_fq_box_ba.std()
                std_unfiltered_AOT_0865p50 = AOT_0865p50_box_ba.std()

                # mask values that are not within +/- 1.5*std of mean\
                
                rhow_0412p50_fq_box_ba = ma.masked_outside(rhow_0412p50_fq_box_ba,mean_unfiltered_rhow_0412p50\
                    -1.5*std_unfiltered_rhow_0412p50\
                    , mean_unfiltered_rhow_0412p50\
                    +1.5*std_unfiltered_rhow_0412p50)
                rhow_0442p50_fq_box_ba = ma.masked_outside(rhow_0442p50_fq_box_ba,mean_unfiltered_rhow_0442p50\
                    -1.5*std_unfiltered_rhow_0442p50\
                    , mean_unfiltered_rhow_0442p50\
                    +1.5*std_unfiltered_rhow_0442p50)
                rhow_0490p00_fq_box_ba = ma.masked_outside(rhow_0490p00_fq_box_ba,mean_unfiltered_rhow_0490p00\
                    -1.5*std_unfiltered_rhow_0490p00\
                    , mean_unfiltered_rhow_0490p00\
                    +1.5*std_unfiltered_rhow_0490p00)
                rhow_0560p00_fq_box_ba = ma.masked_outside(rhow_0560p00_fq_box_ba,mean_unfiltered_rhow_0560p00\
                    -1.5*std_unfiltered_rhow_0560p00\
                    , mean_unfiltered_rhow_0560p00\
                    +1.5*std_unfiltered_rhow_0560p00)
                rhow_0665p00_fq_box_ba = ma.masked_outside(rhow_0665p00_fq_box_ba,mean_unfiltered_rhow_0665p00\
                    -1.5*std_unfiltered_rhow_0665p00\
                    , mean_unfiltered_rhow_0665p00\
                    +1.5*std_unfiltered_rhow_0665p00)
                AOT_0865p50_box_ba = ma.masked_outside(AOT_0865p50_box_ba,mean_unfiltered_AOT_0865p50\
                    -1.5*std_unfiltered_AOT_0865p50\
                    , mean_unfiltered_AOT_0865p50\
                    +1.5*std_unfiltered_AOT_0865p50)

                mean_filtered_rhow_0412p50 = rhow_0412p50_fq_box_ba.mean()
                mean_filtered_rhow_0442p50 = rhow_0442p50_fq_box_ba.mean()
                mean_filtered_rhow_0490p00 = rhow_0490p00_fq_box_ba.mean()
                mean_filtered_rhow_0560p00 = rhow_0560p00_fq_box_ba.mean()
                mean_filtered_rhow_0665p00 = rhow_0665p00_fq_box_ba.mean()
                mean_filtered_AOT_0865p50  = AOT_0865p50_box_ba.mean()

                std_filtered_rhow_0412p50 = rhow_0412p50_fq_box_ba.std()
                std_filtered_rhow_0442p50 = rhow_0442p50_fq_box_ba.std()
                std_filtered_rhow_0490p00 = rhow_0490p00_fq_box_ba.std()
                std_filtered_rhow_0560p00 = rhow_0560p00_fq_box_ba.std()
                std_filtered_rhow_0665p00 = rhow_0665p00_fq_box_ba.std()
                std_filtered_AOT_0865p50  = AOT_0865p50_box_ba.std()

                CV_filtered_rhow_0412p50 = std_filtered_rhow_0412p50/mean_filtered_rhow_0412p50
                CV_filtered_rhow_0442p50 = std_filtered_rhow_0442p50/mean_filtered_rhow_0442p50
                CV_filtered_rhow_0490p00 = std_filtered_rhow_0490p00/mean_filtered_rhow_0490p00
                CV_filtered_rhow_0560p00 = std_filtered_rhow_0560p00/mean_filtered_rhow_0560p00
                CV_filtered_rhow_0665p00 = std_filtered_rhow_0665p00/mean_filtered_rhow_0665p00
                CV_filtered_AOT_0865p50  = std_filtered_AOT_0865p50/mean_filtered_AOT_0865p50  
                
                CVs = [CV_filtered_rhow_0412p50, CV_filtered_rhow_0442p50,\
                                     CV_filtered_rhow_0490p00, CV_filtered_rhow_0560p00,\
                                     CV_filtered_AOT_0865p50]
                print(CVs)
                MedianCV = np.nanmedian(np.abs(CVs))
                print('Median CV='+str(MedianCV))
                idx_m = np.argmin(np.abs(MedianCV-np.abs(CVs)))

                # to count potential matchups total and per stations
                pot_mu_cnt_ba += 1
                if station_name == 'Venise':
                    pot_mu_cnt_ba_Venise += 1
                elif station_name == 'Gloria':
                    pot_mu_cnt_ba_Gloria += 1
                elif station_name == 'Galata_Platform':
                    pot_mu_cnt_ba_Galata_Platform += 1
                elif station_name == 'Helsinki_Lighthouse':
                    pot_mu_cnt_ba_Helsinki_Lighthouse += 1
                elif station_name == 'Gustav_Dalen_Tower':
                    pot_mu_cnt_ba_Gustav_Dalen_Tower += 1

                # count rejections
                if sza>75 or vza>60:
                    rej_ang_mu_cnt_ba += 1
                    if station_name == 'Venise':
                        rej_ang_mu_cnt_ba_Venise += 1
                    elif station_name == 'Gloria':
                        rej_ang_mu_cnt_ba_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_ang_mu_cnt_ba_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_ang_mu_cnt_ba_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_ang_mu_cnt_ba_Gustav_Dalen_Tower += 1    

                if sza>75:
                    rej_sza_mu_cnt_ba += 1
                    if station_name == 'Venise':
                        rej_sza_mu_cnt_ba_Venise += 1
                    elif station_name == 'Gloria':
                        rej_sza_mu_cnt_ba_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_sza_mu_cnt_ba_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_sza_mu_cnt_ba_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_sza_mu_cnt_ba_Gustav_Dalen_Tower += 1   

                if vza>60:
                    rej_vza_mu_cnt_ba += 1
                    if station_name == 'Venise':
                        rej_vza_mu_cnt_ba_Venise += 1
                    elif station_name == 'Gloria':
                        rej_vza_mu_cnt_ba_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_vza_mu_cnt_ba_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_vza_mu_cnt_ba_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_vza_mu_cnt_ba_Gustav_Dalen_Tower += 1                                                                    

                if NGP<=NTP/2+1 or NGP_rhow_0412p50<=NTP/2+1 \
                        or NGP_rhow_0442p50<=NTP/2+1 or NGP_rhow_0490p00<=NTP/2+1 \
                        or NGP_rhow_0560p00<=NTP/2+1 or NGP_rhow_0665p00<=NTP/2+1:
                    rej_inv_mu_cnt_ba += 1
                    if station_name == 'Venise':
                        rej_inv_mu_cnt_ba_Venise += 1
                    elif station_name == 'Gloria':
                        rej_inv_mu_cnt_ba_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_inv_mu_cnt_ba_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_inv_mu_cnt_ba_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_inv_mu_cnt_ba_Gustav_Dalen_Tower += 1

                if MedianCV > 0.15:
                    rej_cvs_mu_cnt_ba += 1
                    if station_name == 'Venise':
                        rej_cvs_mu_cnt_ba_Venise += 1
                    elif station_name == 'Gloria':
                        rej_cvs_mu_cnt_ba_Gloria += 1
                    elif station_name == 'Galata_Platform':
                        rej_cvs_mu_cnt_ba_Galata_Platform += 1
                    elif station_name == 'Helsinki_Lighthouse':
                        rej_cvs_mu_cnt_ba_Helsinki_Lighthouse += 1
                    elif station_name == 'Gustav_Dalen_Tower':
                        rej_cvs_mu_cnt_ba_Gustav_Dalen_Tower += 1 
                    print(f'rej. idx MedianCV: {idx_m}')    
                    idx_medianCV.append(idx_m)

                # create matchup
                if sza<=75 and vza<=60 and NGP>NTP/2+1 and MedianCV <= 0.15\
                        and (NGP_rhow_0412p50>NTP/2+1)\
                        and (NGP_rhow_0442p50>NTP/2+1)\
                        and (NGP_rhow_0490p00>NTP/2+1)\
                        and (NGP_rhow_0560p00>NTP/2+1)\
                        and (NGP_rhow_0665p00>NTP/2+1):  
                    dt_mu_ba.append(dt_hour[idx_min])
                    
                    number_used_pixels = np.append(number_used_pixels,[[0, 0, 0, 0, 0]],axis=0)
                    BW06_MU = True
                    mu_cnt_ba += 1              
                    # Rrs 0412p50
                    # print('412.5')
                    if NGP_rhow_0412p50>NTP/2+1:
                        mu_cnt2_ba += 1
                        # print('Exceeded: NGP_rhow_0412p50='+str(NGP_rhow_0412p50))
                    # else:
                        mu_Lwn_0412p50_fq_sat_ba.append(mean_filtered_rhow_0412p50*F0_0412p50/np.pi)
                        mu_Lwn_0412p50_fq_ins_ba.append(insitu_Lwn_fonQ_ba[olci_wl_list.index(412.5)]) # 412,
                        mu_Lwn_0412p50_fq_ins_ba_station.append(station_name)
                        mu_Lwn_0412p50_fq_sat_ba_stop_time.append(sat_stop_time)
                        mu_Lwn_0412p50_fq_ins_ba_time.append(ins_time[idx_min])
                        
                        map_valid_pxs = np.ones(rhow_0412p50_fq_box_ba.shape,dtype=int)
                        map_valid_pxs[rhow_0412p50_fq_box_ba.mask==True]=0
                        map_valid_pxs_ba[station_idx,0,:,:] = map_valid_pxs_ba[station_idx,0,:,:] + map_valid_pxs

                        number_used_pixels[-1,0] = rhow_0412p50_fq_box_ba.count()

                    # Rrs 0442p50
                    # print('442.5')
                    if NGP_rhow_0442p50>NTP/2+1:
                        # print('Exceeded: NGP_rhow_0442p50='+str(NGP_rhow_0442p50))
                    # else:
                        mu_Lwn_0442p50_fq_sat_ba.append(mean_filtered_rhow_0442p50*F0_0442p50/np.pi)
                        mu_Lwn_0442p50_fq_ins_ba.append(insitu_Lwn_fonQ_ba[olci_wl_list.index(442.5)]) # 441.8
                        mu_Lwn_0442p50_fq_ins_ba_station.append(station_name)
                        mu_Lwn_0442p50_fq_sat_ba_stop_time.append(sat_stop_time)
                        mu_Lwn_0442p50_fq_ins_ba_time.append(ins_time[idx_min])

                        map_valid_pxs = np.ones(rhow_0442p50_fq_box_ba.shape,dtype=int)
                        map_valid_pxs[rhow_0442p50_fq_box_ba.mask==True]=0
                        map_valid_pxs_ba[station_idx,1,:,:] = map_valid_pxs_ba[station_idx,1,:,:] + map_valid_pxs

                        number_used_pixels[-1,1] = rhow_0442p50_fq_box_ba.count()
                        
                    # Rrs 0490p00
                    # print('490.0')
                    if NGP_rhow_0490p00>NTP/2+1:
                        # print('Exceeded: NGP_rhow_0490p00='+str(NGP_rhow_0490p00))
                    # else:
                        mu_Lwn_0490p00_fq_sat_ba.append(mean_filtered_rhow_0490p00*F0_0490p00/np.pi)
                        mu_Lwn_0490p00_fq_ins_ba.append(insitu_Lwn_fonQ_ba[olci_wl_list.index(490.0)]) # 488.5
                        mu_Lwn_0490p00_fq_ins_ba_station.append(station_name)
                        mu_Lwn_0490p00_fq_sat_ba_stop_time.append(sat_stop_time)
                        mu_Lwn_0490p00_fq_ins_ba_time.append(ins_time[idx_min])

                        map_valid_pxs = np.ones(rhow_0490p00_fq_box_ba.shape,dtype=int)
                        map_valid_pxs[rhow_0490p00_fq_box_ba.mask==True]=0
                        map_valid_pxs_ba[station_idx,2,:,:] = map_valid_pxs_ba[station_idx,2,:,:] + map_valid_pxs

                        number_used_pixels[-1,2] = rhow_0490p00_fq_box_ba.count()
                        
                        # for determining the pixels that are not used for the validation
                        if rhow_0490p00_fq_box_ba.mask[1,1] == True and station_list_main[station_idx] == 'Gloria':
                            data_masked_Gloria.append(line)
                            data_masked_Gloria_value.append(rhow_0490p00_fq_box_ba.data[1,1])
                    # Rrs 0560p00
                    # print('560.0')
                    if NGP_rhow_0560p00>NTP/2+1:
                        # print('Exceeded: NGP_rhow_0560p00='+str(NGP_rhow_0560p00))
                    # else:

                        mu_Lwn_0560p00_fq_sat_ba.append(mean_filtered_rhow_0560p00*F0_0560p00/np.pi)
                        mu_Lwn_0560p00_fq_ins_ba.append(insitu_Lwn_fonQ_ba[olci_wl_list.index(560.0)]) # 551,
                        mu_Lwn_0560p00_fq_ins_ba_station.append(station_name)
                        mu_Lwn_0560p00_fq_sat_ba_stop_time.append(sat_stop_time)
                        mu_Lwn_0560p00_fq_ins_ba_time.append(ins_time[idx_min])

                        map_valid_pxs = np.ones(rhow_0560p00_fq_box_ba.shape,dtype=int)
                        map_valid_pxs[rhow_0560p00_fq_box_ba.mask==True]=0
                        map_valid_pxs_ba[station_idx,3,:,:] = map_valid_pxs_ba[station_idx,3,:,:] + map_valid_pxs

                        number_used_pixels[-1,3] = rhow_0560p00_fq_box_ba.count()

                        
                    # Rrs 0665p00
                    # print('665.0')
                    if NGP_rhow_0665p00>NTP/2+1:
                        # print('Exceeded: NGP_rhow_0665p00='+str(NGP_rhow_0665p00))
                    # else:
                        mu_Lwn_0665p00_fq_sat_ba.append(mean_filtered_rhow_0665p00*F0_0665p00/np.pi)
                        mu_Lwn_0665p00_fq_ins_ba.append(insitu_Lwn_fonQ_ba[olci_wl_list.index(665.0)]) # 667.9    
                        mu_Lwn_0665p00_fq_ins_ba_station.append(station_name)
                        mu_Lwn_0665p00_fq_sat_ba_stop_time.append(sat_stop_time)
                        mu_Lwn_0665p00_fq_ins_ba_time.append(ins_time[idx_min])

                        map_valid_pxs = np.ones(rhow_0665p00_fq_box_ba.shape,dtype=int)
                        map_valid_pxs[rhow_0665p00_fq_box_ba.mask==True]=0
                        map_valid_pxs_ba[station_idx,4,:,:] = map_valid_pxs_ba[station_idx,4,:,:] + map_valid_pxs

                        number_used_pixels[-1,4] = rhow_0665p00_fq_box_ba.count()

                    # to analyze CV
                    df_CVs_ba = df_CVs_ba.append({'station_idx':station_idx,'rhow_412.5':CV_filtered_rhow_0412p50,'rhow_442.5':CV_filtered_rhow_0442p50,\
                        'rhow_490.0':CV_filtered_rhow_0490p00,'rhow_560.0':CV_filtered_rhow_0560p00,'rhow_665.0':CV_filtered_rhow_0665p00,\
                        'aot_865.5':CV_filtered_AOT_0865p50,'MedianCV':MedianCV,'MedianCV_band_idx':idx_m,'sat_datetime':sat_stop_time},\
                        ignore_index=True)
                        
            # create df_matchups
            try:
                rhow_0560p00_fq_box_ba_count = rhow_0560p00_fq_box_ba.count()
            except:
                rhow_0560p00_fq_box_ba_count = np.nan

            try:
                rhow_0412p50_fq_box_zi_mean = rhow_0412p50_fq_box_zi.mean()
            except:
                rhow_0412p50_fq_box_zi_mean = np.nan        

            try:
                rhow_0442p50_fq_box_zi_mean = rhow_0442p50_fq_box_zi.mean()
            except:
                rhow_0442p50_fq_box_zi_mean = np.nan        

            try:
                rhow_0490p00_fq_box_zi_mean = rhow_0490p00_fq_box_zi.mean()
            except:
                rhow_0490p00_fq_box_zi_mean = np.nan        

            try:
                rhow_0560p00_fq_box_zi_mean = rhow_0560p00_fq_box_zi.mean()
            except:
                rhow_0560p00_fq_box_zi_mean = np.nan        

            try:
                rhow_0665p00_fq_box_zi_mean = rhow_0665p00_fq_box_zi.mean()
            except:
                rhow_0665p00_fq_box_zi_mean = np.nan   
                                 
            df_matchups = df_matchups.append({'station':station_list_main[station_idx],'sat_datetime':sat_stop_time,'insitu_datetime':ins_time[idx_min],'vza':vza,'sza':sza,\
                     'BW06_MU':BW06_MU,'BW06_l2_mask':flags_mask_ba,\
                     'BW06: rhow_412_box':rhow_0412p50_fq_box_ba,'BW06: rho_412_filt_mean':mean_filtered_rhow_0412p50,\
                     'BW06: rhow_442_box':rhow_0442p50_fq_box_ba,'BW06: rho_442_filt_mean':mean_filtered_rhow_0442p50,\
                     'BW06: rhow_490_box':rhow_0490p00_fq_box_ba,'BW06: rho_490_filt_mean':mean_filtered_rhow_0490p00,\
                     'BW06: rhow_560_box':rhow_0560p00_fq_box_ba,'BW06: rho_560_filt_mean':mean_filtered_rhow_0560p00,\
                     'BW06: rhow_665_box':rhow_0665p00_fq_box_ba,'BW06: rho_665_filt_mean':mean_filtered_rhow_0665p00,\
                     'BW06: MedianCV':MedianCV,'BW06: Nfilt_560':rhow_0560p00_fq_box_ba_count,'BW06: NGP':NGP,'BW06: NTP':NTP,\
                     'BW06: CV_rhow_412.5':CV_filtered_rhow_0412p50,'BW06: CV_rhow_442.5':CV_filtered_rhow_0442p50,\
                     'BW06: CV_rhow_490.0':CV_filtered_rhow_0490p00,'BW06: CV_rhow_560.0':CV_filtered_rhow_0560p00,\
                     'BW06: CV_rhow_665.0':CV_filtered_rhow_0665p00,'BW06: CV_aot_865.5':CV_filtered_AOT_0865p50,\
                     'BW06: MedianCV_band_idx':idx_m,\
                     'BW06: insitu_Lwn_fonQ':insitu_Lwn_fonQ_ba,\
                     'BW06_Exact_wavelengths_ba':Exact_wavelengths_ba,\
                     'BW06_insitu_Lwn_fonQ_ba_all':insitu_Lwn_fonQ_ba_all,\
                     'Z09_MU':Z09_MU,'Z09_l2_mask':flags_mask_zi,\
                     'Z09: rhow_412_box':rhow_0412p50_fq_box_zi,'Z09: rho_412_mean':rhow_0412p50_fq_box_zi_mean,\
                     'Z09: rhow_442_box':rhow_0442p50_fq_box_zi,'Z09: rho_442_mean':rhow_0442p50_fq_box_zi_mean,\
                     'Z09: rhow_490_box':rhow_0490p00_fq_box_zi,'Z09: rho_490_mean':rhow_0490p00_fq_box_zi_mean,\
                     'Z09: rhow_560_box':rhow_0560p00_fq_box_zi,'Z09: rho_560_mean':rhow_0560p00_fq_box_zi_mean,\
                     'Z09: rhow_665_box':rhow_0665p00_fq_box_zi,'Z09: rho_665_mean':rhow_0665p00_fq_box_zi_mean,\
                     'Z09: CV_560':Lwn_560_CV,'Z09: CV_865p5':AOT_0865p50_CV,\
                     'Z09: insitu_Lwn_fonQ':insitu_Lwn_fonQ_zi,\
                     'Z09_Exact_wavelengths_zi':Exact_wavelengths_zi,\
                     'Z09_insitu_Lwn_fonQ_zi_all':insitu_Lwn_fonQ_zi_all},ignore_index=True)   

  

    #         else:
    #             print('Median CV exceeds criteria: Median[CV]='+str(MedianCV))
    #     else:
    #         print('Angles exceeds criteria: sza='+str(sza)+'; vza='+str(vza)+'; OR NGP='+str(NGP)+'< NTP/2+1='+str(NTP/2+1)+'!')
    # else:
    #     print('Not matchups per '+year_str+' '+doy_str)            



#%% plots 
if plot_flag:  
    prot_name = 'zi'
    sensor_name = 'OLCI'
    rmse_val_0412p50_zi, mean_abs_rel_diff_0412p50_zi, mean_rel_diff_0412p50_zi, mean_bias_0412p50_zi, mean_abs_error_0412p50_zi, r_sqr_0412p50_zi,\
    rmse_val_0412p50_zi_Venise,mean_abs_rel_diff_0412p50_zi_Venise, mean_rel_diff_0412p50_zi_Venise, mean_bias_0412p50_zi_Venise, mean_abs_error_0412p50_zi_Venise, r_sqr_0412p50_zi_Venise,\
    rmse_val_0412p50_zi_Gloria,mean_abs_rel_diff_0412p50_zi_Gloria, mean_rel_diff_0412p50_zi_Gloria, mean_bias_0412p50_zi_Gloria, mean_abs_error_0412p50_zi_Gloria, r_sqr_0412p50_zi_Gloria,\
    rmse_val_0412p50_zi_Galata_Platform,mean_abs_rel_diff_0412p50_zi_Galata_Platform, mean_rel_diff_0412p50_zi_Galata_Platform, mean_bias_0412p50_zi_Galata_Platform, mean_abs_error_0412p50_zi_Galata_Platform, r_sqr_0412p50_zi_Galata_Platform,\
    rmse_val_0412p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0412p50_zi_Helsinki_Lighthouse, mean_rel_diff_0412p50_zi_Helsinki_Lighthouse, mean_bias_0412p50_zi_Helsinki_Lighthouse, mean_abs_error_0412p50_zi_Helsinki_Lighthouse, r_sqr_0412p50_zi_Helsinki_Lighthouse,\
    rmse_val_0412p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0412p50_zi_Gustav_Dalen_Tower, mean_rel_diff_0412p50_zi_Gustav_Dalen_Tower, mean_bias_0412p50_zi_Gustav_Dalen_Tower, mean_abs_error_0412p50_zi_Gustav_Dalen_Tower, r_sqr_0412p50_zi_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0412p50_fq_ins_zi,mu_Lwn_0412p50_fq_sat_zi,'412.5',path_out,prot_name,sensor_name,\
        mu_Lwn_0412p50_fq_ins_zi_station,min_val=-3.00,max_val=5.0)
    
    rmse_val_0442p50_zi, mean_abs_rel_diff_0442p50_zi, mean_rel_diff_0442p50_zi, mean_bias_0442p50_zi, mean_abs_error_0442p50_zi, r_sqr_0442p50_zi,\
    rmse_val_0442p50_zi_Venise,mean_abs_rel_diff_0442p50_zi_Venise, mean_rel_diff_0442p50_zi_Venise, mean_bias_0442p50_zi_Venise, mean_abs_error_0442p50_zi_Venise, r_sqr_0442p50_zi_Venise,\
    rmse_val_0442p50_zi_Gloria,mean_abs_rel_diff_0442p50_zi_Gloria, mean_rel_diff_0442p50_zi_Gloria, mean_bias_0442p50_zi_Gloria, mean_abs_error_0442p50_zi_Gloria, r_sqr_0442p50_zi_Gloria,\
    rmse_val_0442p50_zi_Galata_Platform,mean_abs_rel_diff_0442p50_zi_Galata_Platform, mean_rel_diff_0442p50_zi_Galata_Platform, mean_bias_0442p50_zi_Galata_Platform, mean_abs_error_0442p50_zi_Galata_Platform, r_sqr_0442p50_zi_Galata_Platform,\
    rmse_val_0442p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_zi_Helsinki_Lighthouse, mean_rel_diff_0442p50_zi_Helsinki_Lighthouse, mean_bias_0442p50_zi_Helsinki_Lighthouse, mean_abs_error_0442p50_zi_Helsinki_Lighthouse, r_sqr_0442p50_zi_Helsinki_Lighthouse,\
    rmse_val_0442p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_zi_Gustav_Dalen_Tower, mean_rel_diff_0442p50_zi_Gustav_Dalen_Tower, mean_bias_0442p50_zi_Gustav_Dalen_Tower, mean_abs_error_0442p50_zi_Gustav_Dalen_Tower, r_sqr_0442p50_zi_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0442p50_fq_ins_zi,mu_Lwn_0442p50_fq_sat_zi,'442.5',path_out,prot_name,sensor_name,\
        mu_Lwn_0442p50_fq_ins_zi_station,min_val=-3.00,max_val=6.2)
    
    rmse_val_0490p00_zi, mean_abs_rel_diff_0490p00_zi, mean_rel_diff_0490p00_zi, mean_bias_0490p00_zi, mean_abs_error_0490p00_zi, r_sqr_0490p00_zi,\
    rmse_val_0490p00_zi_Venise,mean_abs_rel_diff_0490p00_zi_Venise, mean_rel_diff_0490p00_zi_Venise, mean_bias_0490p00_zi_Venise, mean_abs_error_0490p00_zi_Venise, r_sqr_0490p00_zi_Venise,\
    rmse_val_0490p00_zi_Gloria,mean_abs_rel_diff_0490p00_zi_Gloria, mean_rel_diff_0490p00_zi_Gloria, mean_bias_0490p00_zi_Gloria, mean_abs_error_0490p00_zi_Gloria, r_sqr_0490p00_zi_Gloria,\
    rmse_val_0490p00_zi_Galata_Platform,mean_abs_rel_diff_0490p00_zi_Galata_Platform, mean_rel_diff_0490p00_zi_Galata_Platform, mean_bias_0490p00_zi_Galata_Platform, mean_abs_error_0490p00_zi_Galata_Platform, r_sqr_0490p00_zi_Galata_Platform,\
    rmse_val_0490p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_zi_Helsinki_Lighthouse, mean_rel_diff_0490p00_zi_Helsinki_Lighthouse, mean_bias_0490p00_zi_Helsinki_Lighthouse, mean_abs_error_0490p00_zi_Helsinki_Lighthouse, r_sqr_0490p00_zi_Helsinki_Lighthouse,\
    rmse_val_0490p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_zi_Gustav_Dalen_Tower, mean_rel_diff_0490p00_zi_Gustav_Dalen_Tower, mean_bias_0490p00_zi_Gustav_Dalen_Tower, mean_abs_error_0490p00_zi_Gustav_Dalen_Tower, r_sqr_0490p00_zi_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0490p00_fq_ins_zi,mu_Lwn_0490p00_fq_sat_zi,'490.0',path_out,prot_name,sensor_name,\
        mu_Lwn_0490p00_fq_ins_zi_station,min_val=-2.00,max_val=8.0)
    
    rmse_val_0560p00_zi, mean_abs_rel_diff_0560p00_zi, mean_rel_diff_0560p00_zi, mean_bias_0560p00_zi, mean_abs_error_0560p00_zi, r_sqr_0560p00_zi,\
    rmse_val_0560p00_zi_Venise,mean_abs_rel_diff_0560p00_zi_Venise, mean_rel_diff_0560p00_zi_Venise, mean_bias_0560p00_zi_Venise, mean_abs_error_0560p00_zi_Venise, r_sqr_0560p00_zi_Venise,\
    rmse_val_0560p00_zi_Gloria,mean_abs_rel_diff_0560p00_zi_Gloria, mean_rel_diff_0560p00_zi_Gloria, mean_bias_0560p00_zi_Gloria, mean_abs_error_0560p00_zi_Gloria, r_sqr_0560p00_zi_Gloria,\
    rmse_val_0560p00_zi_Galata_Platform,mean_abs_rel_diff_0560p00_zi_Galata_Platform, mean_rel_diff_0560p00_zi_Galata_Platform, mean_bias_0560p00_zi_Galata_Platform, mean_abs_error_0560p00_zi_Galata_Platform, r_sqr_0560p00_zi_Galata_Platform,\
    rmse_val_0560p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_zi_Helsinki_Lighthouse, mean_rel_diff_0560p00_zi_Helsinki_Lighthouse, mean_bias_0560p00_zi_Helsinki_Lighthouse, mean_abs_error_0560p00_zi_Helsinki_Lighthouse, r_sqr_0560p00_zi_Helsinki_Lighthouse,\
    rmse_val_0560p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_zi_Gustav_Dalen_Tower, mean_rel_diff_0560p00_zi_Gustav_Dalen_Tower, mean_bias_0560p00_zi_Gustav_Dalen_Tower, mean_abs_error_0560p00_zi_Gustav_Dalen_Tower, r_sqr_0560p00_zi_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0560p00_fq_ins_zi,mu_Lwn_0560p00_fq_sat_zi,'560.0',path_out,prot_name,sensor_name,\
        mu_Lwn_0560p00_fq_ins_zi_station,min_val=-0.50,max_val=6.0)
    
    rmse_val_0665p00_zi, mean_abs_rel_diff_0665p00_zi, mean_rel_diff_0665p00_zi, mean_bias_0665p00_zi, mean_abs_error_0665p00_zi, r_sqr_0665p00_zi,\
    rmse_val_0665p00_zi_Venise,mean_abs_rel_diff_0665p00_zi_Venise, mean_rel_diff_0665p00_zi_Venise, mean_bias_0665p00_zi_Venise, mean_abs_error_0665p00_zi_Venise, r_sqr_0665p00_zi_Venise,\
    rmse_val_0665p00_zi_Gloria,mean_abs_rel_diff_0665p00_zi_Gloria, mean_rel_diff_0665p00_zi_Gloria, mean_bias_0665p00_zi_Gloria, mean_abs_error_0665p00_zi_Gloria, r_sqr_0665p00_zi_Gloria,\
    rmse_val_0665p00_zi_Galata_Platform,mean_abs_rel_diff_0665p00_zi_Galata_Platform, mean_rel_diff_0665p00_zi_Galata_Platform, mean_bias_0665p00_zi_Galata_Platform, mean_abs_error_0665p00_zi_Galata_Platform, r_sqr_0665p00_zi_Galata_Platform,\
    rmse_val_0665p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_zi_Helsinki_Lighthouse, mean_rel_diff_0665p00_zi_Helsinki_Lighthouse, mean_bias_0665p00_zi_Helsinki_Lighthouse, mean_abs_error_0665p00_zi_Helsinki_Lighthouse, r_sqr_0665p00_zi_Helsinki_Lighthouse,\
    rmse_val_0665p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_zi_Gustav_Dalen_Tower, mean_rel_diff_0665p00_zi_Gustav_Dalen_Tower, mean_bias_0665p00_zi_Gustav_Dalen_Tower, mean_abs_error_0665p00_zi_Gustav_Dalen_Tower, r_sqr_0665p00_zi_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0665p00_fq_ins_zi,mu_Lwn_0665p00_fq_sat_zi,'665.0',path_out,prot_name,sensor_name,\
        mu_Lwn_0665p00_fq_ins_zi_station,min_val=-0.60,max_val=4.0)
    
    #% plots  
    prot_name = 'ba' 
    sensor_name = 'OLCI'
    rmse_val_0412p50_ba, mean_abs_rel_diff_0412p50_ba, mean_rel_diff_0412p50_ba, mean_bias_0412p50_ba, mean_abs_error_0412p50_ba, r_sqr_0412p50_ba,\
    rmse_val_0412p50_ba_Venise,mean_abs_rel_diff_0412p50_ba_Venise, mean_rel_diff_0412p50_ba_Venise, mean_bias_0412p50_ba_Venise, mean_abs_error_0412p50_ba_Venise, r_sqr_0412p50_ba_Venise,\
    rmse_val_0412p50_ba_Gloria,mean_abs_rel_diff_0412p50_ba_Gloria, mean_rel_diff_0412p50_ba_Gloria, mean_bias_0412p50_ba_Gloria, mean_abs_error_0412p50_ba_Gloria, r_sqr_0412p50_ba_Gloria,\
    rmse_val_0412p50_ba_Galata_Platform,mean_abs_rel_diff_0412p50_ba_Galata_Platform, mean_rel_diff_0412p50_ba_Galata_Platform, mean_bias_0412p50_ba_Galata_Platform, mean_abs_error_0412p50_ba_Galata_Platform, r_sqr_0412p50_ba_Galata_Platform,\
    rmse_val_0412p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0412p50_ba_Helsinki_Lighthouse, mean_rel_diff_0412p50_ba_Helsinki_Lighthouse, mean_bias_0412p50_ba_Helsinki_Lighthouse, mean_abs_error_0412p50_ba_Helsinki_Lighthouse, r_sqr_0412p50_ba_Helsinki_Lighthouse,\
    rmse_val_0412p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0412p50_ba_Gustav_Dalen_Tower, mean_rel_diff_0412p50_ba_Gustav_Dalen_Tower, mean_bias_0412p50_ba_Gustav_Dalen_Tower, mean_abs_error_0412p50_ba_Gustav_Dalen_Tower, r_sqr_0412p50_ba_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0412p50_fq_ins_ba,mu_Lwn_0412p50_fq_sat_ba,'412.5',path_out,prot_name,sensor_name,\
        mu_Lwn_0412p50_fq_ins_ba_station,min_val=-3.00,max_val=5.0)
    rmse_val_0442p50_ba, mean_abs_rel_diff_0442p50_ba, mean_rel_diff_0442p50_ba, mean_bias_0442p50_ba, mean_abs_error_0442p50_ba, r_sqr_0442p50_ba,\
    rmse_val_0442p50_ba_Venise,mean_abs_rel_diff_0442p50_ba_Venise, mean_rel_diff_0442p50_ba_Venise, mean_bias_0442p50_ba_Venise, mean_abs_error_0442p50_ba_Venise, r_sqr_0442p50_ba_Venise,\
    rmse_val_0442p50_ba_Gloria,mean_abs_rel_diff_0442p50_ba_Gloria, mean_rel_diff_0442p50_ba_Gloria, mean_bias_0442p50_ba_Gloria, mean_abs_error_0442p50_ba_Gloria, r_sqr_0442p50_ba_Gloria,\
    rmse_val_0442p50_ba_Galata_Platform,mean_abs_rel_diff_0442p50_ba_Galata_Platform, mean_rel_diff_0442p50_ba_Galata_Platform, mean_bias_0442p50_ba_Galata_Platform, mean_abs_error_0442p50_ba_Galata_Platform, r_sqr_0442p50_ba_Galata_Platform,\
    rmse_val_0442p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_ba_Helsinki_Lighthouse, mean_rel_diff_0442p50_ba_Helsinki_Lighthouse, mean_bias_0442p50_ba_Helsinki_Lighthouse, mean_abs_error_0442p50_ba_Helsinki_Lighthouse, r_sqr_0442p50_ba_Helsinki_Lighthouse,\
    rmse_val_0442p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_ba_Gustav_Dalen_Tower, mean_rel_diff_0442p50_ba_Gustav_Dalen_Tower, mean_bias_0442p50_ba_Gustav_Dalen_Tower, mean_abs_error_0442p50_ba_Gustav_Dalen_Tower, r_sqr_0442p50_ba_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0442p50_fq_ins_ba,mu_Lwn_0442p50_fq_sat_ba,'442.5',path_out,prot_name,sensor_name,\
        mu_Lwn_0442p50_fq_ins_ba_station,min_val=-3.00,max_val=6.2)
    rmse_val_0490p00_ba, mean_abs_rel_diff_0490p00_ba, mean_rel_diff_0490p00_ba, mean_bias_0490p00_ba, mean_abs_error_0490p00_ba, r_sqr_0490p00_ba,\
    rmse_val_0490p00_ba_Venise,mean_abs_rel_diff_0490p00_ba_Venise, mean_rel_diff_0490p00_ba_Venise, mean_bias_0490p00_ba_Venise, mean_abs_error_0490p00_ba_Venise, r_sqr_0490p00_ba_Venise,\
    rmse_val_0490p00_ba_Gloria,mean_abs_rel_diff_0490p00_ba_Gloria, mean_rel_diff_0490p00_ba_Gloria, mean_bias_0490p00_ba_Gloria, mean_abs_error_0490p00_ba_Gloria, r_sqr_0490p00_ba_Gloria,\
    rmse_val_0490p00_ba_Galata_Platform,mean_abs_rel_diff_0490p00_ba_Galata_Platform, mean_rel_diff_0490p00_ba_Galata_Platform, mean_bias_0490p00_ba_Galata_Platform, mean_abs_error_0490p00_ba_Galata_Platform, r_sqr_0490p00_ba_Galata_Platform,\
    rmse_val_0490p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_ba_Helsinki_Lighthouse, mean_rel_diff_0490p00_ba_Helsinki_Lighthouse, mean_bias_0490p00_ba_Helsinki_Lighthouse, mean_abs_error_0490p00_ba_Helsinki_Lighthouse, r_sqr_0490p00_ba_Helsinki_Lighthouse,\
    rmse_val_0490p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_ba_Gustav_Dalen_Tower, mean_rel_diff_0490p00_ba_Gustav_Dalen_Tower, mean_bias_0490p00_ba_Gustav_Dalen_Tower, mean_abs_error_0490p00_ba_Gustav_Dalen_Tower, r_sqr_0490p00_ba_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0490p00_fq_ins_ba,mu_Lwn_0490p00_fq_sat_ba,'490.0',path_out,prot_name,sensor_name,\
        mu_Lwn_0490p00_fq_ins_ba_station,min_val=-2.00,max_val=8.0)
    rmse_val_0560p00_ba, mean_abs_rel_diff_0560p00_ba, mean_rel_diff_0560p00_ba, mean_bias_0560p00_ba, mean_abs_error_0560p00_ba, r_sqr_0560p00_ba,\
    rmse_val_0560p00_ba_Venise,mean_abs_rel_diff_0560p00_ba_Venise, mean_rel_diff_0560p00_ba_Venise, mean_bias_0560p00_ba_Venise, mean_abs_error_0560p00_ba_Venise, r_sqr_0560p00_ba_Venise,\
    rmse_val_0560p00_ba_Gloria,mean_abs_rel_diff_0560p00_ba_Gloria, mean_rel_diff_0560p00_ba_Gloria, mean_bias_0560p00_ba_Gloria, mean_abs_error_0560p00_ba_Gloria, r_sqr_0560p00_ba_Gloria,\
    rmse_val_0560p00_ba_Galata_Platform,mean_abs_rel_diff_0560p00_ba_Galata_Platform, mean_rel_diff_0560p00_ba_Galata_Platform, mean_bias_0560p00_ba_Galata_Platform, mean_abs_error_0560p00_ba_Galata_Platform, r_sqr_0560p00_ba_Galata_Platform,\
    rmse_val_0560p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_ba_Helsinki_Lighthouse, mean_rel_diff_0560p00_ba_Helsinki_Lighthouse, mean_bias_0560p00_ba_Helsinki_Lighthouse, mean_abs_error_0560p00_ba_Helsinki_Lighthouse, r_sqr_0560p00_ba_Helsinki_Lighthouse,\
    rmse_val_0560p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_ba_Gustav_Dalen_Tower, mean_rel_diff_0560p00_ba_Gustav_Dalen_Tower, mean_bias_0560p00_ba_Gustav_Dalen_Tower, mean_abs_error_0560p00_ba_Gustav_Dalen_Tower, r_sqr_0560p00_ba_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0560p00_fq_ins_ba,mu_Lwn_0560p00_fq_sat_ba,'560.0',path_out,prot_name,sensor_name,\
        mu_Lwn_0560p00_fq_ins_ba_station,min_val=-0.50,max_val=6.0)
    rmse_val_0665p00_ba, mean_abs_rel_diff_0665p00_ba, mean_rel_diff_0665p00_ba, mean_bias_0665p00_ba, mean_abs_error_0665p00_ba, r_sqr_0665p00_ba,\
    rmse_val_0665p00_ba_Venise,mean_abs_rel_diff_0665p00_ba_Venise, mean_rel_diff_0665p00_ba_Venise, mean_bias_0665p00_ba_Venise, mean_abs_error_0665p00_ba_Venise, r_sqr_0665p00_ba_Venise,\
    rmse_val_0665p00_ba_Gloria,mean_abs_rel_diff_0665p00_ba_Gloria, mean_rel_diff_0665p00_ba_Gloria, mean_bias_0665p00_ba_Gloria, mean_abs_error_0665p00_ba_Gloria, r_sqr_0665p00_ba_Gloria,\
    rmse_val_0665p00_ba_Galata_Platform,mean_abs_rel_diff_0665p00_ba_Galata_Platform, mean_rel_diff_0665p00_ba_Galata_Platform, mean_bias_0665p00_ba_Galata_Platform, mean_abs_error_0665p00_ba_Galata_Platform, r_sqr_0665p00_ba_Galata_Platform,\
    rmse_val_0665p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_ba_Helsinki_Lighthouse, mean_rel_diff_0665p00_ba_Helsinki_Lighthouse, mean_bias_0665p00_ba_Helsinki_Lighthouse, mean_abs_error_0665p00_ba_Helsinki_Lighthouse, r_sqr_0665p00_ba_Helsinki_Lighthouse,\
    rmse_val_0665p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_ba_Gustav_Dalen_Tower, mean_rel_diff_0665p00_ba_Gustav_Dalen_Tower, mean_bias_0665p00_ba_Gustav_Dalen_Tower, mean_abs_error_0665p00_ba_Gustav_Dalen_Tower, r_sqr_0665p00_ba_Gustav_Dalen_Tower\
    = plot_scatter(\
        mu_Lwn_0665p00_fq_ins_ba,mu_Lwn_0665p00_fq_sat_ba,'665.0',path_out,prot_name,sensor_name,\
        mu_Lwn_0665p00_fq_ins_ba_station,min_val=-0.60,max_val=4.0)
    
    #%%
    # rmse
    rmse_zi = [rmse_val_0412p50_zi,rmse_val_0442p50_zi,rmse_val_0490p00_zi,rmse_val_0560p00_zi,rmse_val_0665p00_zi] 
    rmse_ba = [rmse_val_0412p50_ba,rmse_val_0442p50_ba,rmse_val_0490p00_ba,rmse_val_0560p00_ba,rmse_val_0665p00_ba]
    rmse_zi_Venise = [rmse_val_0412p50_zi_Venise,rmse_val_0442p50_zi_Venise,rmse_val_0490p00_zi_Venise,rmse_val_0560p00_zi_Venise,rmse_val_0665p00_zi_Venise] 
    rmse_ba_Venise = [rmse_val_0412p50_ba_Venise,rmse_val_0442p50_ba_Venise,rmse_val_0490p00_ba_Venise,rmse_val_0560p00_ba_Venise,rmse_val_0665p00_ba_Venise]
    rmse_zi_Gloria = [rmse_val_0412p50_zi_Gloria,rmse_val_0442p50_zi_Gloria,rmse_val_0490p00_zi_Gloria,rmse_val_0560p00_zi_Gloria,rmse_val_0665p00_zi_Gloria] 
    rmse_ba_Gloria = [rmse_val_0412p50_ba_Gloria,rmse_val_0442p50_ba_Gloria,rmse_val_0490p00_ba_Gloria,rmse_val_0560p00_ba_Gloria,rmse_val_0665p00_ba_Gloria]
    rmse_zi_Galata_Platform = [rmse_val_0412p50_zi_Galata_Platform,rmse_val_0442p50_zi_Galata_Platform,rmse_val_0490p00_zi_Galata_Platform,rmse_val_0560p00_zi_Galata_Platform,rmse_val_0665p00_zi_Galata_Platform] 
    rmse_ba_Galata_Platform = [rmse_val_0412p50_ba_Galata_Platform,rmse_val_0442p50_ba_Galata_Platform,rmse_val_0490p00_ba_Galata_Platform,rmse_val_0560p00_ba_Galata_Platform,rmse_val_0665p00_ba_Galata_Platform]
    rmse_zi_Helsinki_Lighthouse = [rmse_val_0412p50_zi_Helsinki_Lighthouse,rmse_val_0442p50_zi_Helsinki_Lighthouse,rmse_val_0490p00_zi_Helsinki_Lighthouse,rmse_val_0560p00_zi_Helsinki_Lighthouse,rmse_val_0665p00_zi_Helsinki_Lighthouse] 
    rmse_ba_Helsinki_Lighthouse = [rmse_val_0412p50_ba_Helsinki_Lighthouse,rmse_val_0442p50_ba_Helsinki_Lighthouse,rmse_val_0490p00_ba_Helsinki_Lighthouse,rmse_val_0560p00_ba_Helsinki_Lighthouse,rmse_val_0665p00_ba_Helsinki_Lighthouse]
    rmse_zi_Gustav_Dalen_Tower = [rmse_val_0412p50_zi_Gustav_Dalen_Tower,rmse_val_0442p50_zi_Gustav_Dalen_Tower,rmse_val_0490p00_zi_Gustav_Dalen_Tower,rmse_val_0560p00_zi_Gustav_Dalen_Tower,rmse_val_0665p00_zi_Gustav_Dalen_Tower] 
    rmse_ba_Gustav_Dalen_Tower = [rmse_val_0412p50_ba_Gustav_Dalen_Tower,rmse_val_0442p50_ba_Gustav_Dalen_Tower,rmse_val_0490p00_ba_Gustav_Dalen_Tower,rmse_val_0560p00_ba_Gustav_Dalen_Tower,rmse_val_0665p00_ba_Gustav_Dalen_Tower]
    wv = [412.5,442.5,490.0,560.0,665.0]
    plt.figure()
    kwargs = dict(linewidth=1, markersize=10,markeredgewidth=2)
    kwargs2 = dict(linewidth=2, markersize=10,markeredgewidth=2)
    plt.plot(wv,rmse_zi_Venise,'-+r',**kwargs)
    plt.plot(wv,rmse_ba_Venise,'-xr',**kwargs)
    plt.plot(wv,rmse_zi_Gloria,'-+g',**kwargs)
    plt.plot(wv,rmse_ba_Gloria,'-xg',**kwargs)
    plt.plot(wv,rmse_zi_Galata_Platform,'-+b',**kwargs)
    plt.plot(wv,rmse_ba_Galata_Platform,'-xb',**kwargs)
    plt.plot(wv,rmse_zi_Helsinki_Lighthouse,'-+m',**kwargs)
    plt.plot(wv,rmse_ba_Helsinki_Lighthouse,'-xm',**kwargs)
    plt.plot(wv,rmse_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
    plt.plot(wv,rmse_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
    plt.plot(wv,rmse_zi,'--+k',**kwargs2)
    plt.plot(wv,rmse_ba,'--xk',**kwargs2)
    plt.xlabel('Wavelength [nm]',fontsize=12)
    plt.ylabel('$RMSD$',fontsize=12)
    # plt.legend(['Zibordi, Mèlin and Berthon (2018)','Bailey and Werdell (2006)'])
    plt.show()
    
    ofname = 'OLCI_rmse.pdf'
    ofname = os.path.join(path_out,'source',ofname)   
    plt.savefig(ofname, dpi=300)
    
    #% mean_abs_rel_diff
    mean_abs_rel_diff_zi = [mean_abs_rel_diff_0412p50_zi,mean_abs_rel_diff_0442p50_zi,mean_abs_rel_diff_0490p00_zi,mean_abs_rel_diff_0560p00_zi,mean_abs_rel_diff_0665p00_zi]
    mean_abs_rel_diff_ba = [mean_abs_rel_diff_0412p50_ba,mean_abs_rel_diff_0442p50_ba,mean_abs_rel_diff_0490p00_ba,mean_abs_rel_diff_0560p00_ba,mean_abs_rel_diff_0665p00_ba]
    mean_abs_rel_diff_zi_Venise = [mean_abs_rel_diff_0412p50_zi_Venise,mean_abs_rel_diff_0442p50_zi_Venise,mean_abs_rel_diff_0490p00_zi_Venise,mean_abs_rel_diff_0560p00_zi_Venise,mean_abs_rel_diff_0665p00_zi_Venise] 
    mean_abs_rel_diff_ba_Venise = [mean_abs_rel_diff_0412p50_ba_Venise,mean_abs_rel_diff_0442p50_ba_Venise,mean_abs_rel_diff_0490p00_ba_Venise,mean_abs_rel_diff_0560p00_ba_Venise,mean_abs_rel_diff_0665p00_ba_Venise]
    mean_abs_rel_diff_zi_Gloria = [mean_abs_rel_diff_0412p50_zi_Gloria,mean_abs_rel_diff_0442p50_zi_Gloria,mean_abs_rel_diff_0490p00_zi_Gloria,mean_abs_rel_diff_0560p00_zi_Gloria,mean_abs_rel_diff_0665p00_zi_Gloria] 
    mean_abs_rel_diff_ba_Gloria = [mean_abs_rel_diff_0412p50_ba_Gloria,mean_abs_rel_diff_0442p50_ba_Gloria,mean_abs_rel_diff_0490p00_ba_Gloria,mean_abs_rel_diff_0560p00_ba_Gloria,mean_abs_rel_diff_0665p00_ba_Gloria]
    mean_abs_rel_diff_zi_Galata_Platform = [mean_abs_rel_diff_0412p50_zi_Galata_Platform,mean_abs_rel_diff_0442p50_zi_Galata_Platform,mean_abs_rel_diff_0490p00_zi_Galata_Platform,mean_abs_rel_diff_0560p00_zi_Galata_Platform,mean_abs_rel_diff_0665p00_zi_Galata_Platform] 
    mean_abs_rel_diff_ba_Galata_Platform = [mean_abs_rel_diff_0412p50_ba_Galata_Platform,mean_abs_rel_diff_0442p50_ba_Galata_Platform,mean_abs_rel_diff_0490p00_ba_Galata_Platform,mean_abs_rel_diff_0560p00_ba_Galata_Platform,mean_abs_rel_diff_0665p00_ba_Galata_Platform]
    mean_abs_rel_diff_zi_Helsinki_Lighthouse = [mean_abs_rel_diff_0412p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_zi_Helsinki_Lighthouse] 
    mean_abs_rel_diff_ba_Helsinki_Lighthouse = [mean_abs_rel_diff_0412p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_ba_Helsinki_Lighthouse]
    mean_abs_rel_diff_zi_Gustav_Dalen_Tower = [mean_abs_rel_diff_0412p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_zi_Gustav_Dalen_Tower] 
    mean_abs_rel_diff_ba_Gustav_Dalen_Tower = [mean_abs_rel_diff_0412p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_ba_Gustav_Dalen_Tower]    
    wv = [412.5,442.5,490.0,560.0,665.0]
    plt.figure()
    plt.plot(wv,mean_abs_rel_diff_zi_Venise,'-+r',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_ba_Venise,'-xr',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_zi_Gloria,'-+g',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_ba_Gloria,'-xg',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_zi_Galata_Platform,'-+b',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_ba_Galata_Platform,'-xb',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_zi_Helsinki_Lighthouse,'-+m',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_ba_Helsinki_Lighthouse,'-xm',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
    plt.plot(wv,mean_abs_rel_diff_zi,'--+k',**kwargs2)
    plt.plot(wv,mean_abs_rel_diff_ba,'--xk',**kwargs2)
    plt.xlabel('Wavelength [nm]',fontsize=12)
    plt.ylabel('MAPD [%]',fontsize=12)
    # plt.legend(['Zibordi','Bailey and Werdell'])
    plt.show()
    
    ofname = 'OLCI_mean_abs_rel_diff.pdf'
    ofname = os.path.join(path_out,'source',ofname)   
    plt.savefig(ofname, dpi=300)
    
    # mean_rel_diff
    mean_rel_diff_zi = [mean_rel_diff_0412p50_zi,mean_rel_diff_0442p50_zi,mean_rel_diff_0490p00_zi,\
        mean_rel_diff_0560p00_zi,mean_rel_diff_0665p00_zi]
    mean_rel_diff_ba = [mean_rel_diff_0412p50_ba,mean_rel_diff_0442p50_ba,mean_rel_diff_0490p00_ba,\
        mean_rel_diff_0560p00_ba,mean_rel_diff_0665p00_ba]
    mean_rel_diff_zi_Venise = [mean_rel_diff_0412p50_zi_Venise,mean_rel_diff_0442p50_zi_Venise,mean_rel_diff_0490p00_zi_Venise,mean_rel_diff_0560p00_zi_Venise,mean_rel_diff_0665p00_zi_Venise] 
    mean_rel_diff_ba_Venise = [mean_rel_diff_0412p50_ba_Venise,mean_rel_diff_0442p50_ba_Venise,mean_rel_diff_0490p00_ba_Venise,mean_rel_diff_0560p00_ba_Venise,mean_rel_diff_0665p00_ba_Venise]
    mean_rel_diff_zi_Gloria = [mean_rel_diff_0412p50_zi_Gloria,mean_rel_diff_0442p50_zi_Gloria,mean_rel_diff_0490p00_zi_Gloria,mean_rel_diff_0560p00_zi_Gloria,mean_rel_diff_0665p00_zi_Gloria] 
    mean_rel_diff_ba_Gloria = [mean_rel_diff_0412p50_ba_Gloria,mean_rel_diff_0442p50_ba_Gloria,mean_rel_diff_0490p00_ba_Gloria,mean_rel_diff_0560p00_ba_Gloria,mean_rel_diff_0665p00_ba_Gloria]
    mean_rel_diff_zi_Galata_Platform = [mean_rel_diff_0412p50_zi_Galata_Platform,mean_rel_diff_0442p50_zi_Galata_Platform,mean_rel_diff_0490p00_zi_Galata_Platform,mean_rel_diff_0560p00_zi_Galata_Platform,mean_rel_diff_0665p00_zi_Galata_Platform] 
    mean_rel_diff_ba_Galata_Platform = [mean_rel_diff_0412p50_ba_Galata_Platform,mean_rel_diff_0442p50_ba_Galata_Platform,mean_rel_diff_0490p00_ba_Galata_Platform,mean_rel_diff_0560p00_ba_Galata_Platform,mean_rel_diff_0665p00_ba_Galata_Platform]
    mean_rel_diff_zi_Helsinki_Lighthouse = [mean_rel_diff_0412p50_zi_Helsinki_Lighthouse,mean_rel_diff_0442p50_zi_Helsinki_Lighthouse,mean_rel_diff_0490p00_zi_Helsinki_Lighthouse,mean_rel_diff_0560p00_zi_Helsinki_Lighthouse,mean_rel_diff_0665p00_zi_Helsinki_Lighthouse] 
    mean_rel_diff_ba_Helsinki_Lighthouse = [mean_rel_diff_0412p50_ba_Helsinki_Lighthouse,mean_rel_diff_0442p50_ba_Helsinki_Lighthouse,mean_rel_diff_0490p00_ba_Helsinki_Lighthouse,mean_rel_diff_0560p00_ba_Helsinki_Lighthouse,mean_rel_diff_0665p00_ba_Helsinki_Lighthouse]
    mean_rel_diff_zi_Gustav_Dalen_Tower = [mean_rel_diff_0412p50_zi_Gustav_Dalen_Tower,mean_rel_diff_0442p50_zi_Gustav_Dalen_Tower,mean_rel_diff_0490p00_zi_Gustav_Dalen_Tower,mean_rel_diff_0560p00_zi_Gustav_Dalen_Tower,mean_rel_diff_0665p00_zi_Gustav_Dalen_Tower] 
    mean_rel_diff_ba_Gustav_Dalen_Tower = [mean_rel_diff_0412p50_ba_Gustav_Dalen_Tower,mean_rel_diff_0442p50_ba_Gustav_Dalen_Tower,mean_rel_diff_0490p00_ba_Gustav_Dalen_Tower,mean_rel_diff_0560p00_ba_Gustav_Dalen_Tower,mean_rel_diff_0665p00_ba_Gustav_Dalen_Tower]    
    
    wv = [412.5,442.5,490.0,560.0,665.0]
    plt.figure()
    plt.plot(wv,mean_rel_diff_zi_Venise,'-+r',**kwargs)
    plt.plot(wv,mean_rel_diff_ba_Venise,'-xr',**kwargs)
    plt.plot(wv,mean_rel_diff_zi_Gloria,'-+g',**kwargs)
    plt.plot(wv,mean_rel_diff_ba_Gloria,'-xg',**kwargs)
    plt.plot(wv,mean_rel_diff_zi_Galata_Platform,'-+b',**kwargs)
    plt.plot(wv,mean_rel_diff_ba_Galata_Platform,'-xb',**kwargs)
    plt.plot(wv,mean_rel_diff_zi_Helsinki_Lighthouse,'-+m',**kwargs)
    plt.plot(wv,mean_rel_diff_ba_Helsinki_Lighthouse,'-xm',**kwargs)
    plt.plot(wv,mean_rel_diff_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
    plt.plot(wv,mean_rel_diff_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
    plt.plot(wv,mean_rel_diff_zi,'--+k',**kwargs2)
    plt.plot(wv,mean_rel_diff_ba,'--xk',**kwargs2)
    plt.xlabel('Wavelength [nm]',fontsize=12)
    plt.ylabel('MPD [%]',fontsize=12)
    # plt.legend(['Zibordi','Bailey and Werdell'])
    plt.show()    
    
    ofname = 'OLCI_mean_rel_diff.pdf'
    ofname = os.path.join(path_out,'source',ofname)   
    plt.savefig(ofname, dpi=300)
    
    # r_sqr
    r_sqr_zi = [r_sqr_0412p50_zi,r_sqr_0442p50_zi,r_sqr_0490p00_zi,\
        r_sqr_0560p00_zi,r_sqr_0665p00_zi]
    r_sqr_ba = [r_sqr_0412p50_ba,r_sqr_0442p50_ba,r_sqr_0490p00_ba,\
        r_sqr_0560p00_ba,r_sqr_0665p00_ba]
    r_sqr_zi_Venise = [r_sqr_0412p50_zi_Venise,r_sqr_0442p50_zi_Venise,r_sqr_0490p00_zi_Venise,r_sqr_0560p00_zi_Venise,r_sqr_0665p00_zi_Venise] 
    r_sqr_ba_Venise = [r_sqr_0412p50_ba_Venise,r_sqr_0442p50_ba_Venise,r_sqr_0490p00_ba_Venise,r_sqr_0560p00_ba_Venise,r_sqr_0665p00_ba_Venise]
    r_sqr_zi_Gloria = [r_sqr_0412p50_zi_Gloria,r_sqr_0442p50_zi_Gloria,r_sqr_0490p00_zi_Gloria,r_sqr_0560p00_zi_Gloria,r_sqr_0665p00_zi_Gloria] 
    r_sqr_ba_Gloria = [r_sqr_0412p50_ba_Gloria,r_sqr_0442p50_ba_Gloria,r_sqr_0490p00_ba_Gloria,r_sqr_0560p00_ba_Gloria,r_sqr_0665p00_ba_Gloria]
    r_sqr_zi_Galata_Platform = [r_sqr_0412p50_zi_Galata_Platform,r_sqr_0442p50_zi_Galata_Platform,r_sqr_0490p00_zi_Galata_Platform,r_sqr_0560p00_zi_Galata_Platform,r_sqr_0665p00_zi_Galata_Platform] 
    r_sqr_ba_Galata_Platform = [r_sqr_0412p50_ba_Galata_Platform,r_sqr_0442p50_ba_Galata_Platform,r_sqr_0490p00_ba_Galata_Platform,r_sqr_0560p00_ba_Galata_Platform,r_sqr_0665p00_ba_Galata_Platform]
    r_sqr_zi_Helsinki_Lighthouse = [r_sqr_0412p50_zi_Helsinki_Lighthouse,r_sqr_0442p50_zi_Helsinki_Lighthouse,r_sqr_0490p00_zi_Helsinki_Lighthouse,r_sqr_0560p00_zi_Helsinki_Lighthouse,r_sqr_0665p00_zi_Helsinki_Lighthouse] 
    r_sqr_ba_Helsinki_Lighthouse = [r_sqr_0412p50_ba_Helsinki_Lighthouse,r_sqr_0442p50_ba_Helsinki_Lighthouse,r_sqr_0490p00_ba_Helsinki_Lighthouse,r_sqr_0560p00_ba_Helsinki_Lighthouse,r_sqr_0665p00_ba_Helsinki_Lighthouse]
    r_sqr_zi_Gustav_Dalen_Tower = [r_sqr_0412p50_zi_Gustav_Dalen_Tower,r_sqr_0442p50_zi_Gustav_Dalen_Tower,r_sqr_0490p00_zi_Gustav_Dalen_Tower,r_sqr_0560p00_zi_Gustav_Dalen_Tower,r_sqr_0665p00_zi_Gustav_Dalen_Tower] 
    r_sqr_ba_Gustav_Dalen_Tower = [r_sqr_0412p50_ba_Gustav_Dalen_Tower,r_sqr_0442p50_ba_Gustav_Dalen_Tower,r_sqr_0490p00_ba_Gustav_Dalen_Tower,r_sqr_0560p00_ba_Gustav_Dalen_Tower,r_sqr_0665p00_ba_Gustav_Dalen_Tower]    
    
    wv = [412.5,442.5,490.0,560.0,665.0]
    plt.figure()
    plt.plot(wv,r_sqr_zi_Venise,'-+r',**kwargs)
    plt.plot(wv,r_sqr_ba_Venise,'-xr',**kwargs)
    plt.plot(wv,r_sqr_zi_Gloria,'-+g',**kwargs)
    plt.plot(wv,r_sqr_ba_Gloria,'-xg',**kwargs)
    plt.plot(wv,r_sqr_zi_Galata_Platform,'-+b',**kwargs)
    plt.plot(wv,r_sqr_ba_Galata_Platform,'-xb',**kwargs)
    plt.plot(wv,r_sqr_zi_Helsinki_Lighthouse,'-+m',**kwargs)
    plt.plot(wv,r_sqr_ba_Helsinki_Lighthouse,'-xm',**kwargs)
    plt.plot(wv,r_sqr_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
    plt.plot(wv,r_sqr_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
    plt.plot(wv,r_sqr_zi,'--+k',**kwargs2)
    plt.plot(wv,r_sqr_ba,'--xk',**kwargs2)
    plt.xlabel('Wavelength [nm]',fontsize=12)
    plt.ylabel('$r^2$',fontsize=12)
    # plt.legend(['Z09','BW06'],fontsize=12)
    plt.show()    
    
    ofname = 'OLCI_r_sqr.pdf'
    ofname = os.path.join(path_out,'source',ofname)   
    plt.savefig(ofname, dpi=300)   
    
    # mean_bias
    mean_bias_zi = [mean_bias_0412p50_zi,mean_bias_0442p50_zi,mean_bias_0490p00_zi,\
        mean_bias_0560p00_zi,mean_bias_0665p00_zi]
    mean_bias_ba = [mean_bias_0412p50_ba,mean_bias_0442p50_ba,mean_bias_0490p00_ba,\
        mean_bias_0560p00_ba,mean_bias_0665p00_ba]
    mean_bias_zi_Venise = [mean_bias_0412p50_zi_Venise,mean_bias_0442p50_zi_Venise,mean_bias_0490p00_zi_Venise,mean_bias_0560p00_zi_Venise,mean_bias_0665p00_zi_Venise] 
    mean_bias_ba_Venise = [mean_bias_0412p50_ba_Venise,mean_bias_0442p50_ba_Venise,mean_bias_0490p00_ba_Venise,mean_bias_0560p00_ba_Venise,mean_bias_0665p00_ba_Venise]
    mean_bias_zi_Gloria = [mean_bias_0412p50_zi_Gloria,mean_bias_0442p50_zi_Gloria,mean_bias_0490p00_zi_Gloria,mean_bias_0560p00_zi_Gloria,mean_bias_0665p00_zi_Gloria] 
    mean_bias_ba_Gloria = [mean_bias_0412p50_ba_Gloria,mean_bias_0442p50_ba_Gloria,mean_bias_0490p00_ba_Gloria,mean_bias_0560p00_ba_Gloria,mean_bias_0665p00_ba_Gloria]
    mean_bias_zi_Galata_Platform = [mean_bias_0412p50_zi_Galata_Platform,mean_bias_0442p50_zi_Galata_Platform,mean_bias_0490p00_zi_Galata_Platform,mean_bias_0560p00_zi_Galata_Platform,mean_bias_0665p00_zi_Galata_Platform] 
    mean_bias_ba_Galata_Platform = [mean_bias_0412p50_ba_Galata_Platform,mean_bias_0442p50_ba_Galata_Platform,mean_bias_0490p00_ba_Galata_Platform,mean_bias_0560p00_ba_Galata_Platform,mean_bias_0665p00_ba_Galata_Platform]
    mean_bias_zi_Helsinki_Lighthouse = [mean_bias_0412p50_zi_Helsinki_Lighthouse,mean_bias_0442p50_zi_Helsinki_Lighthouse,mean_bias_0490p00_zi_Helsinki_Lighthouse,mean_bias_0560p00_zi_Helsinki_Lighthouse,mean_bias_0665p00_zi_Helsinki_Lighthouse] 
    mean_bias_ba_Helsinki_Lighthouse = [mean_bias_0412p50_ba_Helsinki_Lighthouse,mean_bias_0442p50_ba_Helsinki_Lighthouse,mean_bias_0490p00_ba_Helsinki_Lighthouse,mean_bias_0560p00_ba_Helsinki_Lighthouse,mean_bias_0665p00_ba_Helsinki_Lighthouse]
    mean_bias_zi_Gustav_Dalen_Tower = [mean_bias_0412p50_zi_Gustav_Dalen_Tower,mean_bias_0442p50_zi_Gustav_Dalen_Tower,mean_bias_0490p00_zi_Gustav_Dalen_Tower,mean_bias_0560p00_zi_Gustav_Dalen_Tower,mean_bias_0665p00_zi_Gustav_Dalen_Tower] 
    mean_bias_ba_Gustav_Dalen_Tower = [mean_bias_0412p50_ba_Gustav_Dalen_Tower,mean_bias_0442p50_ba_Gustav_Dalen_Tower,mean_bias_0490p00_ba_Gustav_Dalen_Tower,mean_bias_0560p00_ba_Gustav_Dalen_Tower,mean_bias_0665p00_ba_Gustav_Dalen_Tower]    
    
    wv = [412.5,442.5,490.0,560.0,665.0]
    plt.figure()
    plt.plot(wv,mean_bias_zi_Venise,'-+r',**kwargs)
    plt.plot(wv,mean_bias_ba_Venise,'-xr',**kwargs)
    plt.plot(wv,mean_bias_zi_Gloria,'-+g',**kwargs)
    plt.plot(wv,mean_bias_ba_Gloria,'-xg',**kwargs)
    plt.plot(wv,mean_bias_zi_Galata_Platform,'-+b',**kwargs)
    plt.plot(wv,mean_bias_ba_Galata_Platform,'-xb',**kwargs)
    plt.plot(wv,mean_bias_zi_Helsinki_Lighthouse,'-+m',**kwargs)
    plt.plot(wv,mean_bias_ba_Helsinki_Lighthouse,'-xm',**kwargs)
    plt.plot(wv,mean_bias_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
    plt.plot(wv,mean_bias_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
    plt.plot(wv,mean_bias_zi,'--+k',**kwargs2)
    plt.plot(wv,mean_bias_ba,'--xk',**kwargs2)
    plt.xlabel('Wavelength [nm]',fontsize=12)
    plt.ylabel('MB',fontsize=12)
    # plt.legend(['Z09','BW06'],fontsize=12)
    plt.show()    
    
    ofname = 'OLCI_mean_bias.pdf'
    ofname = os.path.join(path_out,'source',ofname)   
    plt.savefig(ofname, dpi=300) 
    
    # mean_abs_error
    mean_abs_error_zi = [mean_abs_error_0412p50_zi,mean_abs_error_0442p50_zi,mean_abs_error_0490p00_zi,\
        mean_abs_error_0560p00_zi,mean_abs_error_0665p00_zi]
    mean_abs_error_ba = [mean_abs_error_0412p50_ba,mean_abs_error_0442p50_ba,mean_abs_error_0490p00_ba,\
        mean_abs_error_0560p00_ba,mean_abs_error_0665p00_ba]
    mean_abs_error_zi_Venise = [mean_abs_error_0412p50_zi_Venise,mean_abs_error_0442p50_zi_Venise,mean_abs_error_0490p00_zi_Venise,mean_abs_error_0560p00_zi_Venise,mean_abs_error_0665p00_zi_Venise] 
    mean_abs_error_ba_Venise = [mean_abs_error_0412p50_ba_Venise,mean_abs_error_0442p50_ba_Venise,mean_abs_error_0490p00_ba_Venise,mean_abs_error_0560p00_ba_Venise,mean_abs_error_0665p00_ba_Venise]
    mean_abs_error_zi_Gloria = [mean_abs_error_0412p50_zi_Gloria,mean_abs_error_0442p50_zi_Gloria,mean_abs_error_0490p00_zi_Gloria,mean_abs_error_0560p00_zi_Gloria,mean_abs_error_0665p00_zi_Gloria] 
    mean_abs_error_ba_Gloria = [mean_abs_error_0412p50_ba_Gloria,mean_abs_error_0442p50_ba_Gloria,mean_abs_error_0490p00_ba_Gloria,mean_abs_error_0560p00_ba_Gloria,mean_abs_error_0665p00_ba_Gloria]
    mean_abs_error_zi_Galata_Platform = [mean_abs_error_0412p50_zi_Galata_Platform,mean_abs_error_0442p50_zi_Galata_Platform,mean_abs_error_0490p00_zi_Galata_Platform,mean_abs_error_0560p00_zi_Galata_Platform,mean_abs_error_0665p00_zi_Galata_Platform] 
    mean_abs_error_ba_Galata_Platform = [mean_abs_error_0412p50_ba_Galata_Platform,mean_abs_error_0442p50_ba_Galata_Platform,mean_abs_error_0490p00_ba_Galata_Platform,mean_abs_error_0560p00_ba_Galata_Platform,mean_abs_error_0665p00_ba_Galata_Platform]
    mean_abs_error_zi_Helsinki_Lighthouse = [mean_abs_error_0412p50_zi_Helsinki_Lighthouse,mean_abs_error_0442p50_zi_Helsinki_Lighthouse,mean_abs_error_0490p00_zi_Helsinki_Lighthouse,mean_abs_error_0560p00_zi_Helsinki_Lighthouse,mean_abs_error_0665p00_zi_Helsinki_Lighthouse] 
    mean_abs_error_ba_Helsinki_Lighthouse = [mean_abs_error_0412p50_ba_Helsinki_Lighthouse,mean_abs_error_0442p50_ba_Helsinki_Lighthouse,mean_abs_error_0490p00_ba_Helsinki_Lighthouse,mean_abs_error_0560p00_ba_Helsinki_Lighthouse,mean_abs_error_0665p00_ba_Helsinki_Lighthouse]
    mean_abs_error_zi_Gustav_Dalen_Tower = [mean_abs_error_0412p50_zi_Gustav_Dalen_Tower,mean_abs_error_0442p50_zi_Gustav_Dalen_Tower,mean_abs_error_0490p00_zi_Gustav_Dalen_Tower,mean_abs_error_0560p00_zi_Gustav_Dalen_Tower,mean_abs_error_0665p00_zi_Gustav_Dalen_Tower] 
    mean_abs_error_ba_Gustav_Dalen_Tower = [mean_abs_error_0412p50_ba_Gustav_Dalen_Tower,mean_abs_error_0442p50_ba_Gustav_Dalen_Tower,mean_abs_error_0490p00_ba_Gustav_Dalen_Tower,mean_abs_error_0560p00_ba_Gustav_Dalen_Tower,mean_abs_error_0665p00_ba_Gustav_Dalen_Tower]    
    
    wv = [412.5,442.5,490.0,560.0,665.0]
    plt.figure()
    plt.plot(wv,mean_abs_error_zi_Venise,'-+r',**kwargs)
    plt.plot(wv,mean_abs_error_ba_Venise,'-xr',**kwargs)
    plt.plot(wv,mean_abs_error_zi_Gloria,'-+g',**kwargs)
    plt.plot(wv,mean_abs_error_ba_Gloria,'-xg',**kwargs)
    plt.plot(wv,mean_abs_error_zi_Galata_Platform,'-+b',**kwargs)
    plt.plot(wv,mean_abs_error_ba_Galata_Platform,'-xb',**kwargs)
    plt.plot(wv,mean_abs_error_zi_Helsinki_Lighthouse,'-+m',**kwargs)
    plt.plot(wv,mean_abs_error_ba_Helsinki_Lighthouse,'-xm',**kwargs)
    plt.plot(wv,mean_abs_error_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
    plt.plot(wv,mean_abs_error_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
    plt.plot(wv,mean_abs_error_zi,'--+k',**kwargs2)
    plt.plot(wv,mean_abs_error_ba,'--xk',**kwargs2)
    plt.xlabel('Wavelength [nm]',fontsize=12)
    plt.ylabel('$MAD$',fontsize=12)
    # plt.legend(['Z09','BW06'],fontsize=12)
    plt.show()    
    
    ofname = 'OLCI_mean_abs_error.pdf'
    ofname = os.path.join(path_out,'source',ofname)   
    plt.savefig(ofname, dpi=300) 
#%%  table of validation
if plot_flag:
    data = {'Protocol':['BW06','Z09','BW06','Z09','BW06','Z09','BW06','Z09','BW06','Z09']}
    df4 = pd.DataFrame(data)
    df4['Wavelength'] = ['412.5','412.5','442.5','442.5','490.0','490.0','560.0','560.0','665.0','665.0']
    df4['N'] = [len(mu_Lwn_0412p50_fq_sat_ba),len(mu_Lwn_0412p50_fq_sat_zi),\
                    len(mu_Lwn_0442p50_fq_sat_ba),len(mu_Lwn_0442p50_fq_sat_zi),\
                    len(mu_Lwn_0490p00_fq_sat_ba),len(mu_Lwn_0490p00_fq_sat_zi),\
                    len(mu_Lwn_0560p00_fq_sat_ba),len(mu_Lwn_0560p00_fq_sat_zi),\
                    len(mu_Lwn_0665p00_fq_sat_ba),len(mu_Lwn_0665p00_fq_sat_zi)]
    df4['MPD'] = [mean_rel_diff_0412p50_ba,mean_rel_diff_0412p50_zi,mean_rel_diff_0442p50_ba,\
                    mean_rel_diff_0442p50_zi,mean_rel_diff_0490p00_ba,mean_rel_diff_0490p00_zi,\
                    mean_rel_diff_0560p00_ba,mean_rel_diff_0560p00_zi,mean_rel_diff_0665p00_ba,\
                    mean_rel_diff_0665p00_zi]
    df4['MAPD'] = [mean_abs_rel_diff_0412p50_ba,mean_abs_rel_diff_0412p50_zi,mean_abs_rel_diff_0442p50_ba,\
                    mean_abs_rel_diff_0442p50_zi,mean_abs_rel_diff_0490p00_ba,mean_abs_rel_diff_0490p00_zi,\
                    mean_abs_rel_diff_0560p00_ba,mean_abs_rel_diff_0560p00_zi,mean_abs_rel_diff_0665p00_ba,\
                    mean_abs_rel_diff_0665p00_zi]     
    df4['MB'] = [mean_bias_0412p50_ba,mean_bias_0412p50_zi,mean_bias_0442p50_ba,\
                    mean_bias_0442p50_zi,mean_bias_0490p00_ba,mean_bias_0490p00_zi,\
                    mean_bias_0560p00_ba,mean_bias_0560p00_zi,mean_bias_0665p00_ba,\
                    mean_bias_0665p00_zi]             
    df4['MAD'] = [mean_abs_error_0412p50_ba,mean_abs_error_0412p50_zi,mean_abs_error_0442p50_ba,\
                    mean_abs_error_0442p50_zi,mean_abs_error_0490p00_ba,mean_abs_error_0490p00_zi,\
                    mean_abs_error_0560p00_ba,mean_abs_error_0560p00_zi,mean_abs_error_0665p00_ba,\
                    mean_abs_error_0665p00_zi]
    df4['RMSD'] = [rmse_val_0412p50_ba,rmse_val_0412p50_zi,rmse_val_0442p50_ba,\
                    rmse_val_0442p50_zi,rmse_val_0490p00_ba,rmse_val_0490p00_zi,\
                    rmse_val_0560p00_ba,rmse_val_0560p00_zi,rmse_val_0665p00_ba,\
                    rmse_val_0665p00_zi]
    df4['r_sqr'] = [r_sqr_0412p50_ba,r_sqr_0412p50_zi,r_sqr_0442p50_ba,\
                    r_sqr_0442p50_zi,r_sqr_0490p00_ba,r_sqr_0490p00_zi,\
                    r_sqr_0560p00_ba,r_sqr_0560p00_zi,r_sqr_0665p00_ba,\
                    r_sqr_0665p00_zi]     
    
    print(tabulate(df4, tablefmt='pipe', headers='keys',showindex=False))

#%% plot both methods
if plot_flag:
    sat_same_zi_412p5 = sat_same_ba_412p5 = ins_same_zi_412p5 = ins_same_ba_412p5 = ins_same_station_412p5 = []
    sat_same_zi_442p5 = sat_same_ba_442p5 = ins_same_zi_442p5 = ins_same_ba_442p5 = ins_same_station_442p5 = []
    sat_same_zi_490p0 = sat_same_ba_490p0 = ins_same_zi_490p0 = ins_same_ba_490p0 = ins_same_station_490p0 = []
    sat_same_zi_560p0 = sat_same_ba_560p0 = ins_same_zi_560p0 = ins_same_ba_560p0 = ins_same_station_560p0 = []
    sat_same_zi_665p0 = sat_same_ba_665p0 = ins_same_zi_665p0 = ins_same_ba_665p0 = ins_same_station_665p0 = []
    notation_flag = 0 # to display percentage difference in the plot
    sat_same_zi_412p5, sat_same_ba_412p5, ins_same_zi_412p5, ins_same_ba_412p5, ins_same_station_412p5 = plot_both_methods('412.5',notation_flag,path_out,min_val=-3.00,max_val=5.0)
    sat_same_zi_442p5, sat_same_ba_442p5, ins_same_zi_442p5, ins_same_ba_442p5, ins_same_station_442p5 = plot_both_methods('442.5',notation_flag,path_out,min_val=-3.00,max_val=6.2)
    sat_same_zi_490p0, sat_same_ba_490p0, ins_same_zi_490p0, ins_same_ba_490p0, ins_same_station_490p0 = plot_both_methods('490.0',notation_flag,path_out,min_val=-2.00,max_val=8.0)
    sat_same_zi_560p0, sat_same_ba_560p0, ins_same_zi_560p0, ins_same_ba_560p0, ins_same_station_560p0 = plot_both_methods('560.0',notation_flag,path_out,min_val=-0.50,max_val=6.0)
    sat_same_zi_665p0, sat_same_ba_665p0, ins_same_zi_665p0, ins_same_ba_665p0, ins_same_station_665p0 = plot_both_methods('665.0',notation_flag,path_out,min_val=-0.60,max_val=4.0)

#%% plots for the common matchups
# prot_name = 'zi_same'
# sensor_name = 'OLCI'
# rmse_val_0412p50_zi, mean_abs_rel_diff_0412p50_zi, mean_rel_diff_0412p50_zi, mean_bias_0412p50_zi, mean_abs_error_0412p50_zi, r_sqr_0412p50_zi,\
# rmse_val_0412p50_zi_Venise,mean_abs_rel_diff_0412p50_zi_Venise, mean_rel_diff_0412p50_zi_Venise, mean_bias_0412p50_zi_Venise, mean_abs_error_0412p50_zi_Venise, r_sqr_0412p50_zi_Venise,\
# rmse_val_0412p50_zi_Gloria,mean_abs_rel_diff_0412p50_zi_Gloria, mean_rel_diff_0412p50_zi_Gloria, mean_bias_0412p50_zi_Gloria, mean_abs_error_0412p50_zi_Gloria, r_sqr_0412p50_zi_Gloria,\
# rmse_val_0412p50_zi_Galata_Platform,mean_abs_rel_diff_0412p50_zi_Galata_Platform, mean_rel_diff_0412p50_zi_Galata_Platform, mean_bias_0412p50_zi_Galata_Platform, mean_abs_error_0412p50_zi_Galata_Platform, r_sqr_0412p50_zi_Galata_Platform,\
# rmse_val_0412p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0412p50_zi_Helsinki_Lighthouse, mean_rel_diff_0412p50_zi_Helsinki_Lighthouse, mean_bias_0412p50_zi_Helsinki_Lighthouse, mean_abs_error_0412p50_zi_Helsinki_Lighthouse, r_sqr_0412p50_zi_Helsinki_Lighthouse,\
# rmse_val_0412p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0412p50_zi_Gustav_Dalen_Tower, mean_rel_diff_0412p50_zi_Gustav_Dalen_Tower, mean_bias_0412p50_zi_Gustav_Dalen_Tower, mean_abs_error_0412p50_zi_Gustav_Dalen_Tower, r_sqr_0412p50_zi_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_zi_412p5,ins_same_zi_412p5,'412.5',path_out,prot_name,sensor_name,\
#     ins_same_station_412p5,min_val=-3.00,max_val=5.0)

# rmse_val_0442p50_zi, mean_abs_rel_diff_0442p50_zi, mean_rel_diff_0442p50_zi, mean_bias_0442p50_zi, mean_abs_error_0442p50_zi, r_sqr_0442p50_zi,\
# rmse_val_0442p50_zi_Venise,mean_abs_rel_diff_0442p50_zi_Venise, mean_rel_diff_0442p50_zi_Venise, mean_bias_0442p50_zi_Venise, mean_abs_error_0442p50_zi_Venise, r_sqr_0442p50_zi_Venise,\
# rmse_val_0442p50_zi_Gloria,mean_abs_rel_diff_0442p50_zi_Gloria, mean_rel_diff_0442p50_zi_Gloria, mean_bias_0442p50_zi_Gloria, mean_abs_error_0442p50_zi_Gloria, r_sqr_0442p50_zi_Gloria,\
# rmse_val_0442p50_zi_Galata_Platform,mean_abs_rel_diff_0442p50_zi_Galata_Platform, mean_rel_diff_0442p50_zi_Galata_Platform, mean_bias_0442p50_zi_Galata_Platform, mean_abs_error_0442p50_zi_Galata_Platform, r_sqr_0442p50_zi_Galata_Platform,\
# rmse_val_0442p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_zi_Helsinki_Lighthouse, mean_rel_diff_0442p50_zi_Helsinki_Lighthouse, mean_bias_0442p50_zi_Helsinki_Lighthouse, mean_abs_error_0442p50_zi_Helsinki_Lighthouse, r_sqr_0442p50_zi_Helsinki_Lighthouse,\
# rmse_val_0442p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_zi_Gustav_Dalen_Tower, mean_rel_diff_0442p50_zi_Gustav_Dalen_Tower, mean_bias_0442p50_zi_Gustav_Dalen_Tower, mean_abs_error_0442p50_zi_Gustav_Dalen_Tower, r_sqr_0442p50_zi_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_zi_442p5,ins_same_zi_442p5,'442.5',path_out,prot_name,sensor_name,\
#     ins_same_station_442p5,min_val=-3.00,max_val=6.2)

# rmse_val_0490p00_zi, mean_abs_rel_diff_0490p00_zi, mean_rel_diff_0490p00_zi, mean_bias_0490p00_zi, mean_abs_error_0490p00_zi, r_sqr_0490p00_zi,\
# rmse_val_0490p00_zi_Venise,mean_abs_rel_diff_0490p00_zi_Venise, mean_rel_diff_0490p00_zi_Venise, mean_bias_0490p00_zi_Venise, mean_abs_error_0490p00_zi_Venise, r_sqr_0490p00_zi_Venise,\
# rmse_val_0490p00_zi_Gloria,mean_abs_rel_diff_0490p00_zi_Gloria, mean_rel_diff_0490p00_zi_Gloria, mean_bias_0490p00_zi_Gloria, mean_abs_error_0490p00_zi_Gloria, r_sqr_0490p00_zi_Gloria,\
# rmse_val_0490p00_zi_Galata_Platform,mean_abs_rel_diff_0490p00_zi_Galata_Platform, mean_rel_diff_0490p00_zi_Galata_Platform, mean_bias_0490p00_zi_Galata_Platform, mean_abs_error_0490p00_zi_Galata_Platform, r_sqr_0490p00_zi_Galata_Platform,\
# rmse_val_0490p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_zi_Helsinki_Lighthouse, mean_rel_diff_0490p00_zi_Helsinki_Lighthouse, mean_bias_0490p00_zi_Helsinki_Lighthouse, mean_abs_error_0490p00_zi_Helsinki_Lighthouse, r_sqr_0490p00_zi_Helsinki_Lighthouse,\
# rmse_val_0490p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_zi_Gustav_Dalen_Tower, mean_rel_diff_0490p00_zi_Gustav_Dalen_Tower, mean_bias_0490p00_zi_Gustav_Dalen_Tower, mean_abs_error_0490p00_zi_Gustav_Dalen_Tower, r_sqr_0490p00_zi_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_zi_490p0,ins_same_zi_490p0,'490.0',path_out,prot_name,sensor_name,\
#     ins_same_station_490p0,min_val=-2.00,max_val=8.0)

# rmse_val_0560p00_zi, mean_abs_rel_diff_0560p00_zi, mean_rel_diff_0560p00_zi, mean_bias_0560p00_zi, mean_abs_error_0560p00_zi, r_sqr_0560p00_zi,\
# rmse_val_0560p00_zi_Venise,mean_abs_rel_diff_0560p00_zi_Venise, mean_rel_diff_0560p00_zi_Venise, mean_bias_0560p00_zi_Venise, mean_abs_error_0560p00_zi_Venise, r_sqr_0560p00_zi_Venise,\
# rmse_val_0560p00_zi_Gloria,mean_abs_rel_diff_0560p00_zi_Gloria, mean_rel_diff_0560p00_zi_Gloria, mean_bias_0560p00_zi_Gloria, mean_abs_error_0560p00_zi_Gloria, r_sqr_0560p00_zi_Gloria,\
# rmse_val_0560p00_zi_Galata_Platform,mean_abs_rel_diff_0560p00_zi_Galata_Platform, mean_rel_diff_0560p00_zi_Galata_Platform, mean_bias_0560p00_zi_Galata_Platform, mean_abs_error_0560p00_zi_Galata_Platform, r_sqr_0560p00_zi_Galata_Platform,\
# rmse_val_0560p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_zi_Helsinki_Lighthouse, mean_rel_diff_0560p00_zi_Helsinki_Lighthouse, mean_bias_0560p00_zi_Helsinki_Lighthouse, mean_abs_error_0560p00_zi_Helsinki_Lighthouse, r_sqr_0560p00_zi_Helsinki_Lighthouse,\
# rmse_val_0560p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_zi_Gustav_Dalen_Tower, mean_rel_diff_0560p00_zi_Gustav_Dalen_Tower, mean_bias_0560p00_zi_Gustav_Dalen_Tower, mean_abs_error_0560p00_zi_Gustav_Dalen_Tower, r_sqr_0560p00_zi_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_zi_560p0,ins_same_zi_560p0,'560.0',path_out,prot_name,sensor_name,\
#     ins_same_station_560p0,min_val=-0.50,max_val=6.0)

# rmse_val_0665p00_zi, mean_abs_rel_diff_0665p00_zi, mean_rel_diff_0665p00_zi, mean_bias_0665p00_zi, mean_abs_error_0665p00_zi, r_sqr_0665p00_zi,\
# rmse_val_0665p00_zi_Venise,mean_abs_rel_diff_0665p00_zi_Venise, mean_rel_diff_0665p00_zi_Venise, mean_bias_0665p00_zi_Venise, mean_abs_error_0665p00_zi_Venise, r_sqr_0665p00_zi_Venise,\
# rmse_val_0665p00_zi_Gloria,mean_abs_rel_diff_0665p00_zi_Gloria, mean_rel_diff_0665p00_zi_Gloria, mean_bias_0665p00_zi_Gloria, mean_abs_error_0665p00_zi_Gloria, r_sqr_0665p00_zi_Gloria,\
# rmse_val_0665p00_zi_Galata_Platform,mean_abs_rel_diff_0665p00_zi_Galata_Platform, mean_rel_diff_0665p00_zi_Galata_Platform, mean_bias_0665p00_zi_Galata_Platform, mean_abs_error_0665p00_zi_Galata_Platform, r_sqr_0665p00_zi_Galata_Platform,\
# rmse_val_0665p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_zi_Helsinki_Lighthouse, mean_rel_diff_0665p00_zi_Helsinki_Lighthouse, mean_bias_0665p00_zi_Helsinki_Lighthouse, mean_abs_error_0665p00_zi_Helsinki_Lighthouse, r_sqr_0665p00_zi_Helsinki_Lighthouse,\
# rmse_val_0665p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_zi_Gustav_Dalen_Tower, mean_rel_diff_0665p00_zi_Gustav_Dalen_Tower, mean_bias_0665p00_zi_Gustav_Dalen_Tower, mean_abs_error_0665p00_zi_Gustav_Dalen_Tower, r_sqr_0665p00_zi_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_zi_665p0,ins_same_zi_665p0,'665.0',path_out,prot_name,sensor_name,\
#     ins_same_station_665p0,min_val=-0.60,max_val=4.0)

# #% plots  
# prot_name = 'ba_same' 
# sensor_name = 'OLCI'
# rmse_val_0412p50_ba, mean_abs_rel_diff_0412p50_ba, mean_rel_diff_0412p50_ba, mean_bias_0412p50_ba, mean_abs_error_0412p50_ba, r_sqr_0412p50_ba,\
# rmse_val_0412p50_ba_Venise,mean_abs_rel_diff_0412p50_ba_Venise, mean_rel_diff_0412p50_ba_Venise, mean_bias_0412p50_ba_Venise, mean_abs_error_0412p50_ba_Venise, r_sqr_0412p50_ba_Venise,\
# rmse_val_0412p50_ba_Gloria,mean_abs_rel_diff_0412p50_ba_Gloria, mean_rel_diff_0412p50_ba_Gloria, mean_bias_0412p50_ba_Gloria, mean_abs_error_0412p50_ba_Gloria, r_sqr_0412p50_ba_Gloria,\
# rmse_val_0412p50_ba_Galata_Platform,mean_abs_rel_diff_0412p50_ba_Galata_Platform, mean_rel_diff_0412p50_ba_Galata_Platform, mean_bias_0412p50_ba_Galata_Platform, mean_abs_error_0412p50_ba_Galata_Platform, r_sqr_0412p50_ba_Galata_Platform,\
# rmse_val_0412p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0412p50_ba_Helsinki_Lighthouse, mean_rel_diff_0412p50_ba_Helsinki_Lighthouse, mean_bias_0412p50_ba_Helsinki_Lighthouse, mean_abs_error_0412p50_ba_Helsinki_Lighthouse, r_sqr_0412p50_ba_Helsinki_Lighthouse,\
# rmse_val_0412p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0412p50_ba_Gustav_Dalen_Tower, mean_rel_diff_0412p50_ba_Gustav_Dalen_Tower, mean_bias_0412p50_ba_Gustav_Dalen_Tower, mean_abs_error_0412p50_ba_Gustav_Dalen_Tower, r_sqr_0412p50_ba_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_ba_412p5,ins_same_ba_412p5,'412.5',path_out,prot_name,sensor_name,\
#     ins_same_station_412p5,min_val=-3.00,max_val=5.0)

# rmse_val_0442p50_ba, mean_abs_rel_diff_0442p50_ba, mean_rel_diff_0442p50_ba, mean_bias_0442p50_ba, mean_abs_error_0442p50_ba, r_sqr_0442p50_ba,\
# rmse_val_0442p50_ba_Venise,mean_abs_rel_diff_0442p50_ba_Venise, mean_rel_diff_0442p50_ba_Venise, mean_bias_0442p50_ba_Venise, mean_abs_error_0442p50_ba_Venise, r_sqr_0442p50_ba_Venise,\
# rmse_val_0442p50_ba_Gloria,mean_abs_rel_diff_0442p50_ba_Gloria, mean_rel_diff_0442p50_ba_Gloria, mean_bias_0442p50_ba_Gloria, mean_abs_error_0442p50_ba_Gloria, r_sqr_0442p50_ba_Gloria,\
# rmse_val_0442p50_ba_Galata_Platform,mean_abs_rel_diff_0442p50_ba_Galata_Platform, mean_rel_diff_0442p50_ba_Galata_Platform, mean_bias_0442p50_ba_Galata_Platform, mean_abs_error_0442p50_ba_Galata_Platform, r_sqr_0442p50_ba_Galata_Platform,\
# rmse_val_0442p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_ba_Helsinki_Lighthouse, mean_rel_diff_0442p50_ba_Helsinki_Lighthouse, mean_bias_0442p50_ba_Helsinki_Lighthouse, mean_abs_error_0442p50_ba_Helsinki_Lighthouse, r_sqr_0442p50_ba_Helsinki_Lighthouse,\
# rmse_val_0442p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_ba_Gustav_Dalen_Tower, mean_rel_diff_0442p50_ba_Gustav_Dalen_Tower, mean_bias_0442p50_ba_Gustav_Dalen_Tower, mean_abs_error_0442p50_ba_Gustav_Dalen_Tower, r_sqr_0442p50_ba_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_ba_442p5,ins_same_ba_442p5,'442.5',path_out,prot_name,sensor_name,\
#     ins_same_station_442p5,min_val=-3.00,max_val=6.2)

# rmse_val_0490p00_ba, mean_abs_rel_diff_0490p00_ba, mean_rel_diff_0490p00_ba, mean_bias_0490p00_ba, mean_abs_error_0490p00_ba, r_sqr_0490p00_ba,\
# rmse_val_0490p00_ba_Venise,mean_abs_rel_diff_0490p00_ba_Venise, mean_rel_diff_0490p00_ba_Venise, mean_bias_0490p00_ba_Venise, mean_abs_error_0490p00_ba_Venise, r_sqr_0490p00_ba_Venise,\
# rmse_val_0490p00_ba_Gloria,mean_abs_rel_diff_0490p00_ba_Gloria, mean_rel_diff_0490p00_ba_Gloria, mean_bias_0490p00_ba_Gloria, mean_abs_error_0490p00_ba_Gloria, r_sqr_0490p00_ba_Gloria,\
# rmse_val_0490p00_ba_Galata_Platform,mean_abs_rel_diff_0490p00_ba_Galata_Platform, mean_rel_diff_0490p00_ba_Galata_Platform, mean_bias_0490p00_ba_Galata_Platform, mean_abs_error_0490p00_ba_Galata_Platform, r_sqr_0490p00_ba_Galata_Platform,\
# rmse_val_0490p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_ba_Helsinki_Lighthouse, mean_rel_diff_0490p00_ba_Helsinki_Lighthouse, mean_bias_0490p00_ba_Helsinki_Lighthouse, mean_abs_error_0490p00_ba_Helsinki_Lighthouse, r_sqr_0490p00_ba_Helsinki_Lighthouse,\
# rmse_val_0490p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_ba_Gustav_Dalen_Tower, mean_rel_diff_0490p00_ba_Gustav_Dalen_Tower, mean_bias_0490p00_ba_Gustav_Dalen_Tower, mean_abs_error_0490p00_ba_Gustav_Dalen_Tower, r_sqr_0490p00_ba_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_ba_490p0,ins_same_ba_490p0,'490.0',path_out,prot_name,sensor_name,\
#     ins_same_station_490p0,min_val=-2.00,max_val=8.0)

# rmse_val_0560p00_ba, mean_abs_rel_diff_0560p00_ba, mean_rel_diff_0560p00_ba, mean_bias_0560p00_ba, mean_abs_error_0560p00_ba, r_sqr_0560p00_ba,\
# rmse_val_0560p00_ba_Venise,mean_abs_rel_diff_0560p00_ba_Venise, mean_rel_diff_0560p00_ba_Venise, mean_bias_0560p00_ba_Venise, mean_abs_error_0560p00_ba_Venise, r_sqr_0560p00_ba_Venise,\
# rmse_val_0560p00_ba_Gloria,mean_abs_rel_diff_0560p00_ba_Gloria, mean_rel_diff_0560p00_ba_Gloria, mean_bias_0560p00_ba_Gloria, mean_abs_error_0560p00_ba_Gloria, r_sqr_0560p00_ba_Gloria,\
# rmse_val_0560p00_ba_Galata_Platform,mean_abs_rel_diff_0560p00_ba_Galata_Platform, mean_rel_diff_0560p00_ba_Galata_Platform, mean_bias_0560p00_ba_Galata_Platform, mean_abs_error_0560p00_ba_Galata_Platform, r_sqr_0560p00_ba_Galata_Platform,\
# rmse_val_0560p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_ba_Helsinki_Lighthouse, mean_rel_diff_0560p00_ba_Helsinki_Lighthouse, mean_bias_0560p00_ba_Helsinki_Lighthouse, mean_abs_error_0560p00_ba_Helsinki_Lighthouse, r_sqr_0560p00_ba_Helsinki_Lighthouse,\
# rmse_val_0560p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_ba_Gustav_Dalen_Tower, mean_rel_diff_0560p00_ba_Gustav_Dalen_Tower, mean_bias_0560p00_ba_Gustav_Dalen_Tower, mean_abs_error_0560p00_ba_Gustav_Dalen_Tower, r_sqr_0560p00_ba_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_ba_560p0,ins_same_ba_560p0,'560.0',path_out,prot_name,sensor_name,\
#     ins_same_station_560p0,min_val=-0.50,max_val=6.0)

# rmse_val_0665p00_ba, mean_abs_rel_diff_0665p00_ba, mean_rel_diff_0665p00_ba, mean_bias_0665p00_ba, mean_abs_error_0665p00_ba, r_sqr_0665p00_ba,\
# rmse_val_0665p00_ba_Venise,mean_abs_rel_diff_0665p00_ba_Venise, mean_rel_diff_0665p00_ba_Venise, mean_bias_0665p00_ba_Venise, mean_abs_error_0665p00_ba_Venise, r_sqr_0665p00_ba_Venise,\
# rmse_val_0665p00_ba_Gloria,mean_abs_rel_diff_0665p00_ba_Gloria, mean_rel_diff_0665p00_ba_Gloria, mean_bias_0665p00_ba_Gloria, mean_abs_error_0665p00_ba_Gloria, r_sqr_0665p00_ba_Gloria,\
# rmse_val_0665p00_ba_Galata_Platform,mean_abs_rel_diff_0665p00_ba_Galata_Platform, mean_rel_diff_0665p00_ba_Galata_Platform, mean_bias_0665p00_ba_Galata_Platform, mean_abs_error_0665p00_ba_Galata_Platform, r_sqr_0665p00_ba_Galata_Platform,\
# rmse_val_0665p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_ba_Helsinki_Lighthouse, mean_rel_diff_0665p00_ba_Helsinki_Lighthouse, mean_bias_0665p00_ba_Helsinki_Lighthouse, mean_abs_error_0665p00_ba_Helsinki_Lighthouse, r_sqr_0665p00_ba_Helsinki_Lighthouse,\
# rmse_val_0665p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_ba_Gustav_Dalen_Tower, mean_rel_diff_0665p00_ba_Gustav_Dalen_Tower, mean_bias_0665p00_ba_Gustav_Dalen_Tower, mean_abs_error_0665p00_ba_Gustav_Dalen_Tower, r_sqr_0665p00_ba_Gustav_Dalen_Tower\
# = plot_scatter(\
#     sat_same_ba_665p0,ins_same_ba_665p0,'665.0',path_out,prot_name,sensor_name,\
#     ins_same_station_665p0,min_val=-0.60,max_val=4.0)


# #%%
# # rmse
# rmse_zi = [rmse_val_0412p50_zi,rmse_val_0442p50_zi,rmse_val_0490p00_zi,rmse_val_0560p00_zi,rmse_val_0665p00_zi] 
# rmse_ba = [rmse_val_0412p50_ba,rmse_val_0442p50_ba,rmse_val_0490p00_ba,rmse_val_0560p00_ba,rmse_val_0665p00_ba]
# rmse_zi_Venise = [rmse_val_0412p50_zi_Venise,rmse_val_0442p50_zi_Venise,rmse_val_0490p00_zi_Venise,rmse_val_0560p00_zi_Venise,rmse_val_0665p00_zi_Venise] 
# rmse_ba_Venise = [rmse_val_0412p50_ba_Venise,rmse_val_0442p50_ba_Venise,rmse_val_0490p00_ba_Venise,rmse_val_0560p00_ba_Venise,rmse_val_0665p00_ba_Venise]
# rmse_zi_Gloria = [rmse_val_0412p50_zi_Gloria,rmse_val_0442p50_zi_Gloria,rmse_val_0490p00_zi_Gloria,rmse_val_0560p00_zi_Gloria,rmse_val_0665p00_zi_Gloria] 
# rmse_ba_Gloria = [rmse_val_0412p50_ba_Gloria,rmse_val_0442p50_ba_Gloria,rmse_val_0490p00_ba_Gloria,rmse_val_0560p00_ba_Gloria,rmse_val_0665p00_ba_Gloria]
# rmse_zi_Galata_Platform = [rmse_val_0412p50_zi_Galata_Platform,rmse_val_0442p50_zi_Galata_Platform,rmse_val_0490p00_zi_Galata_Platform,rmse_val_0560p00_zi_Galata_Platform,rmse_val_0665p00_zi_Galata_Platform] 
# rmse_ba_Galata_Platform = [rmse_val_0412p50_ba_Galata_Platform,rmse_val_0442p50_ba_Galata_Platform,rmse_val_0490p00_ba_Galata_Platform,rmse_val_0560p00_ba_Galata_Platform,rmse_val_0665p00_ba_Galata_Platform]
# rmse_zi_Helsinki_Lighthouse = [rmse_val_0412p50_zi_Helsinki_Lighthouse,rmse_val_0442p50_zi_Helsinki_Lighthouse,rmse_val_0490p00_zi_Helsinki_Lighthouse,rmse_val_0560p00_zi_Helsinki_Lighthouse,rmse_val_0665p00_zi_Helsinki_Lighthouse] 
# rmse_ba_Helsinki_Lighthouse = [rmse_val_0412p50_ba_Helsinki_Lighthouse,rmse_val_0442p50_ba_Helsinki_Lighthouse,rmse_val_0490p00_ba_Helsinki_Lighthouse,rmse_val_0560p00_ba_Helsinki_Lighthouse,rmse_val_0665p00_ba_Helsinki_Lighthouse]
# rmse_zi_Gustav_Dalen_Tower = [rmse_val_0412p50_zi_Gustav_Dalen_Tower,rmse_val_0442p50_zi_Gustav_Dalen_Tower,rmse_val_0490p00_zi_Gustav_Dalen_Tower,rmse_val_0560p00_zi_Gustav_Dalen_Tower,rmse_val_0665p00_zi_Gustav_Dalen_Tower] 
# rmse_ba_Gustav_Dalen_Tower = [rmse_val_0412p50_ba_Gustav_Dalen_Tower,rmse_val_0442p50_ba_Gustav_Dalen_Tower,rmse_val_0490p00_ba_Gustav_Dalen_Tower,rmse_val_0560p00_ba_Gustav_Dalen_Tower,rmse_val_0665p00_ba_Gustav_Dalen_Tower]
# wv = [412.5,442.5,490.0,560.0,665.0]

# #%%
# plt.figure()
# kwargs = dict(linewidth=1, markersize=10,markeredgewidth=2)
# kwargs2 = dict(linewidth=2, markersize=10,markeredgewidth=2)
# plt.plot(wv,rmse_zi_Venise,'-+r',**kwargs)
# plt.plot(wv,rmse_ba_Venise,'-xr',**kwargs)
# plt.plot(wv,rmse_zi_Gloria,'-+g',**kwargs)
# plt.plot(wv,rmse_ba_Gloria,'-xg',**kwargs)
# plt.plot(wv,rmse_zi_Galata_Platform,'-+b',**kwargs)
# plt.plot(wv,rmse_ba_Galata_Platform,'-xb',**kwargs)
# plt.plot(wv,rmse_zi_Helsinki_Lighthouse,'-+m',**kwargs)
# plt.plot(wv,rmse_ba_Helsinki_Lighthouse,'-xm',**kwargs)
# plt.plot(wv,rmse_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
# plt.plot(wv,rmse_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
# plt.plot(wv,rmse_zi,'--+k',**kwargs2)
# plt.plot(wv,rmse_ba,'--xk',**kwargs2)
# plt.xlabel('Wavelength [nm]',fontsize=12)
# plt.ylabel('$RMSD$',fontsize=12)
# # plt.legend(['Zibordi, Mèlin and Berthon (2018)','Bailey and Werdell (2006)'])
# plt.show()

# ofname = 'OLCI_rmse_same.pdf'
# ofname = os.path.join(path_out,'source',ofname)   
# plt.savefig(ofname, dpi=300)

# #% mean_abs_rel_diff
# mean_abs_rel_diff_zi = [mean_abs_rel_diff_0412p50_zi,mean_abs_rel_diff_0442p50_zi,mean_abs_rel_diff_0490p00_zi,mean_abs_rel_diff_0560p00_zi,mean_abs_rel_diff_0665p00_zi]
# mean_abs_rel_diff_ba = [mean_abs_rel_diff_0412p50_ba,mean_abs_rel_diff_0442p50_ba,mean_abs_rel_diff_0490p00_ba,mean_abs_rel_diff_0560p00_ba,mean_abs_rel_diff_0665p00_ba]
# mean_abs_rel_diff_zi_Venise = [mean_abs_rel_diff_0412p50_zi_Venise,mean_abs_rel_diff_0442p50_zi_Venise,mean_abs_rel_diff_0490p00_zi_Venise,mean_abs_rel_diff_0560p00_zi_Venise,mean_abs_rel_diff_0665p00_zi_Venise] 
# mean_abs_rel_diff_ba_Venise = [mean_abs_rel_diff_0412p50_ba_Venise,mean_abs_rel_diff_0442p50_ba_Venise,mean_abs_rel_diff_0490p00_ba_Venise,mean_abs_rel_diff_0560p00_ba_Venise,mean_abs_rel_diff_0665p00_ba_Venise]
# mean_abs_rel_diff_zi_Gloria = [mean_abs_rel_diff_0412p50_zi_Gloria,mean_abs_rel_diff_0442p50_zi_Gloria,mean_abs_rel_diff_0490p00_zi_Gloria,mean_abs_rel_diff_0560p00_zi_Gloria,mean_abs_rel_diff_0665p00_zi_Gloria] 
# mean_abs_rel_diff_ba_Gloria = [mean_abs_rel_diff_0412p50_ba_Gloria,mean_abs_rel_diff_0442p50_ba_Gloria,mean_abs_rel_diff_0490p00_ba_Gloria,mean_abs_rel_diff_0560p00_ba_Gloria,mean_abs_rel_diff_0665p00_ba_Gloria]
# mean_abs_rel_diff_zi_Galata_Platform = [mean_abs_rel_diff_0412p50_zi_Galata_Platform,mean_abs_rel_diff_0442p50_zi_Galata_Platform,mean_abs_rel_diff_0490p00_zi_Galata_Platform,mean_abs_rel_diff_0560p00_zi_Galata_Platform,mean_abs_rel_diff_0665p00_zi_Galata_Platform] 
# mean_abs_rel_diff_ba_Galata_Platform = [mean_abs_rel_diff_0412p50_ba_Galata_Platform,mean_abs_rel_diff_0442p50_ba_Galata_Platform,mean_abs_rel_diff_0490p00_ba_Galata_Platform,mean_abs_rel_diff_0560p00_ba_Galata_Platform,mean_abs_rel_diff_0665p00_ba_Galata_Platform]
# mean_abs_rel_diff_zi_Helsinki_Lighthouse = [mean_abs_rel_diff_0412p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_zi_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_zi_Helsinki_Lighthouse] 
# mean_abs_rel_diff_ba_Helsinki_Lighthouse = [mean_abs_rel_diff_0412p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0442p50_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0490p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0560p00_ba_Helsinki_Lighthouse,mean_abs_rel_diff_0665p00_ba_Helsinki_Lighthouse]
# mean_abs_rel_diff_zi_Gustav_Dalen_Tower = [mean_abs_rel_diff_0412p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_zi_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_zi_Gustav_Dalen_Tower] 
# mean_abs_rel_diff_ba_Gustav_Dalen_Tower = [mean_abs_rel_diff_0412p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0442p50_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0490p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0560p00_ba_Gustav_Dalen_Tower,mean_abs_rel_diff_0665p00_ba_Gustav_Dalen_Tower]    
# wv = [412.5,442.5,490.0,560.0,665.0]
# plt.figure()
# plt.plot(wv,mean_abs_rel_diff_zi_Venise,'-+r',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_ba_Venise,'-xr',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_zi_Gloria,'-+g',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_ba_Gloria,'-xg',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_zi_Galata_Platform,'-+b',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_ba_Galata_Platform,'-xb',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_zi_Helsinki_Lighthouse,'-+m',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_ba_Helsinki_Lighthouse,'-xm',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
# plt.plot(wv,mean_abs_rel_diff_zi,'--+k',**kwargs2)
# plt.plot(wv,mean_abs_rel_diff_ba,'--xk',**kwargs2)
# plt.xlabel('Wavelength [nm]',fontsize=12)
# plt.ylabel('MAPD [%]',fontsize=12)
# # plt.legend(['Zibordi','Bailey and Werdell'])
# plt.show()

# ofname = 'OLCI_mean_abs_rel_diff_same.pdf'
# ofname = os.path.join(path_out,'source',ofname)   
# plt.savefig(ofname, dpi=300)

# # mean_rel_diff
# mean_rel_diff_zi = [mean_rel_diff_0412p50_zi,mean_rel_diff_0442p50_zi,mean_rel_diff_0490p00_zi,\
#     mean_rel_diff_0560p00_zi,mean_rel_diff_0665p00_zi]
# mean_rel_diff_ba = [mean_rel_diff_0412p50_ba,mean_rel_diff_0442p50_ba,mean_rel_diff_0490p00_ba,\
#     mean_rel_diff_0560p00_ba,mean_rel_diff_0665p00_ba]
# mean_rel_diff_zi_Venise = [mean_rel_diff_0412p50_zi_Venise,mean_rel_diff_0442p50_zi_Venise,mean_rel_diff_0490p00_zi_Venise,mean_rel_diff_0560p00_zi_Venise,mean_rel_diff_0665p00_zi_Venise] 
# mean_rel_diff_ba_Venise = [mean_rel_diff_0412p50_ba_Venise,mean_rel_diff_0442p50_ba_Venise,mean_rel_diff_0490p00_ba_Venise,mean_rel_diff_0560p00_ba_Venise,mean_rel_diff_0665p00_ba_Venise]
# mean_rel_diff_zi_Gloria = [mean_rel_diff_0412p50_zi_Gloria,mean_rel_diff_0442p50_zi_Gloria,mean_rel_diff_0490p00_zi_Gloria,mean_rel_diff_0560p00_zi_Gloria,mean_rel_diff_0665p00_zi_Gloria] 
# mean_rel_diff_ba_Gloria = [mean_rel_diff_0412p50_ba_Gloria,mean_rel_diff_0442p50_ba_Gloria,mean_rel_diff_0490p00_ba_Gloria,mean_rel_diff_0560p00_ba_Gloria,mean_rel_diff_0665p00_ba_Gloria]
# mean_rel_diff_zi_Galata_Platform = [mean_rel_diff_0412p50_zi_Galata_Platform,mean_rel_diff_0442p50_zi_Galata_Platform,mean_rel_diff_0490p00_zi_Galata_Platform,mean_rel_diff_0560p00_zi_Galata_Platform,mean_rel_diff_0665p00_zi_Galata_Platform] 
# mean_rel_diff_ba_Galata_Platform = [mean_rel_diff_0412p50_ba_Galata_Platform,mean_rel_diff_0442p50_ba_Galata_Platform,mean_rel_diff_0490p00_ba_Galata_Platform,mean_rel_diff_0560p00_ba_Galata_Platform,mean_rel_diff_0665p00_ba_Galata_Platform]
# mean_rel_diff_zi_Helsinki_Lighthouse = [mean_rel_diff_0412p50_zi_Helsinki_Lighthouse,mean_rel_diff_0442p50_zi_Helsinki_Lighthouse,mean_rel_diff_0490p00_zi_Helsinki_Lighthouse,mean_rel_diff_0560p00_zi_Helsinki_Lighthouse,mean_rel_diff_0665p00_zi_Helsinki_Lighthouse] 
# mean_rel_diff_ba_Helsinki_Lighthouse = [mean_rel_diff_0412p50_ba_Helsinki_Lighthouse,mean_rel_diff_0442p50_ba_Helsinki_Lighthouse,mean_rel_diff_0490p00_ba_Helsinki_Lighthouse,mean_rel_diff_0560p00_ba_Helsinki_Lighthouse,mean_rel_diff_0665p00_ba_Helsinki_Lighthouse]
# mean_rel_diff_zi_Gustav_Dalen_Tower = [mean_rel_diff_0412p50_zi_Gustav_Dalen_Tower,mean_rel_diff_0442p50_zi_Gustav_Dalen_Tower,mean_rel_diff_0490p00_zi_Gustav_Dalen_Tower,mean_rel_diff_0560p00_zi_Gustav_Dalen_Tower,mean_rel_diff_0665p00_zi_Gustav_Dalen_Tower] 
# mean_rel_diff_ba_Gustav_Dalen_Tower = [mean_rel_diff_0412p50_ba_Gustav_Dalen_Tower,mean_rel_diff_0442p50_ba_Gustav_Dalen_Tower,mean_rel_diff_0490p00_ba_Gustav_Dalen_Tower,mean_rel_diff_0560p00_ba_Gustav_Dalen_Tower,mean_rel_diff_0665p00_ba_Gustav_Dalen_Tower]    

# wv = [412.5,442.5,490.0,560.0,665.0]
# plt.figure()
# plt.plot(wv,mean_rel_diff_zi_Venise,'-+r',**kwargs)
# plt.plot(wv,mean_rel_diff_ba_Venise,'-xr',**kwargs)
# plt.plot(wv,mean_rel_diff_zi_Gloria,'-+g',**kwargs)
# plt.plot(wv,mean_rel_diff_ba_Gloria,'-xg',**kwargs)
# plt.plot(wv,mean_rel_diff_zi_Galata_Platform,'-+b',**kwargs)
# plt.plot(wv,mean_rel_diff_ba_Galata_Platform,'-xb',**kwargs)
# plt.plot(wv,mean_rel_diff_zi_Helsinki_Lighthouse,'-+m',**kwargs)
# plt.plot(wv,mean_rel_diff_ba_Helsinki_Lighthouse,'-xm',**kwargs)
# plt.plot(wv,mean_rel_diff_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
# plt.plot(wv,mean_rel_diff_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
# plt.plot(wv,mean_rel_diff_zi,'--+k',**kwargs2)
# plt.plot(wv,mean_rel_diff_ba,'--xk',**kwargs2)
# plt.xlabel('Wavelength [nm]',fontsize=12)
# plt.ylabel('MPD [%]',fontsize=12)
# # plt.legend(['Zibordi','Bailey and Werdell'])
# plt.show()    

# ofname = 'OLCI_mean_rel_diff_same.pdf'
# ofname = os.path.join(path_out,'source',ofname)   
# plt.savefig(ofname, dpi=300)

# # r_sqr
# r_sqr_zi = [r_sqr_0412p50_zi,r_sqr_0442p50_zi,r_sqr_0490p00_zi,\
#     r_sqr_0560p00_zi,r_sqr_0665p00_zi]
# r_sqr_ba = [r_sqr_0412p50_ba,r_sqr_0442p50_ba,r_sqr_0490p00_ba,\
#     r_sqr_0560p00_ba,r_sqr_0665p00_ba]
# r_sqr_zi_Venise = [r_sqr_0412p50_zi_Venise,r_sqr_0442p50_zi_Venise,r_sqr_0490p00_zi_Venise,r_sqr_0560p00_zi_Venise,r_sqr_0665p00_zi_Venise] 
# r_sqr_ba_Venise = [r_sqr_0412p50_ba_Venise,r_sqr_0442p50_ba_Venise,r_sqr_0490p00_ba_Venise,r_sqr_0560p00_ba_Venise,r_sqr_0665p00_ba_Venise]
# r_sqr_zi_Gloria = [r_sqr_0412p50_zi_Gloria,r_sqr_0442p50_zi_Gloria,r_sqr_0490p00_zi_Gloria,r_sqr_0560p00_zi_Gloria,r_sqr_0665p00_zi_Gloria] 
# r_sqr_ba_Gloria = [r_sqr_0412p50_ba_Gloria,r_sqr_0442p50_ba_Gloria,r_sqr_0490p00_ba_Gloria,r_sqr_0560p00_ba_Gloria,r_sqr_0665p00_ba_Gloria]
# r_sqr_zi_Galata_Platform = [r_sqr_0412p50_zi_Galata_Platform,r_sqr_0442p50_zi_Galata_Platform,r_sqr_0490p00_zi_Galata_Platform,r_sqr_0560p00_zi_Galata_Platform,r_sqr_0665p00_zi_Galata_Platform] 
# r_sqr_ba_Galata_Platform = [r_sqr_0412p50_ba_Galata_Platform,r_sqr_0442p50_ba_Galata_Platform,r_sqr_0490p00_ba_Galata_Platform,r_sqr_0560p00_ba_Galata_Platform,r_sqr_0665p00_ba_Galata_Platform]
# r_sqr_zi_Helsinki_Lighthouse = [r_sqr_0412p50_zi_Helsinki_Lighthouse,r_sqr_0442p50_zi_Helsinki_Lighthouse,r_sqr_0490p00_zi_Helsinki_Lighthouse,r_sqr_0560p00_zi_Helsinki_Lighthouse,r_sqr_0665p00_zi_Helsinki_Lighthouse] 
# r_sqr_ba_Helsinki_Lighthouse = [r_sqr_0412p50_ba_Helsinki_Lighthouse,r_sqr_0442p50_ba_Helsinki_Lighthouse,r_sqr_0490p00_ba_Helsinki_Lighthouse,r_sqr_0560p00_ba_Helsinki_Lighthouse,r_sqr_0665p00_ba_Helsinki_Lighthouse]
# r_sqr_zi_Gustav_Dalen_Tower = [r_sqr_0412p50_zi_Gustav_Dalen_Tower,r_sqr_0442p50_zi_Gustav_Dalen_Tower,r_sqr_0490p00_zi_Gustav_Dalen_Tower,r_sqr_0560p00_zi_Gustav_Dalen_Tower,r_sqr_0665p00_zi_Gustav_Dalen_Tower] 
# r_sqr_ba_Gustav_Dalen_Tower = [r_sqr_0412p50_ba_Gustav_Dalen_Tower,r_sqr_0442p50_ba_Gustav_Dalen_Tower,r_sqr_0490p00_ba_Gustav_Dalen_Tower,r_sqr_0560p00_ba_Gustav_Dalen_Tower,r_sqr_0665p00_ba_Gustav_Dalen_Tower]    

# wv = [412.5,442.5,490.0,560.0,665.0]
# plt.figure()
# plt.plot(wv,r_sqr_zi_Venise,'-+r',**kwargs)
# plt.plot(wv,r_sqr_ba_Venise,'-xr',**kwargs)
# plt.plot(wv,r_sqr_zi_Gloria,'-+g',**kwargs)
# plt.plot(wv,r_sqr_ba_Gloria,'-xg',**kwargs)
# plt.plot(wv,r_sqr_zi_Galata_Platform,'-+b',**kwargs)
# plt.plot(wv,r_sqr_ba_Galata_Platform,'-xb',**kwargs)
# plt.plot(wv,r_sqr_zi_Helsinki_Lighthouse,'-+m',**kwargs)
# plt.plot(wv,r_sqr_ba_Helsinki_Lighthouse,'-xm',**kwargs)
# plt.plot(wv,r_sqr_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
# plt.plot(wv,r_sqr_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
# plt.plot(wv,r_sqr_zi,'--+k',**kwargs2)
# plt.plot(wv,r_sqr_ba,'--xk',**kwargs2)
# plt.xlabel('Wavelength [nm]',fontsize=12)
# plt.ylabel('$r^2$',fontsize=12)
# # plt.legend(['Z09','BW06'],fontsize=12)
# plt.show()    

# ofname = 'OLCI_r_sqr_same.pdf'
# ofname = os.path.join(path_out,'source',ofname)   
# plt.savefig(ofname, dpi=300)   

# # mean_bias
# mean_bias_zi = [mean_bias_0412p50_zi,mean_bias_0442p50_zi,mean_bias_0490p00_zi,\
#     mean_bias_0560p00_zi,mean_bias_0665p00_zi]
# mean_bias_ba = [mean_bias_0412p50_ba,mean_bias_0442p50_ba,mean_bias_0490p00_ba,\
#     mean_bias_0560p00_ba,mean_bias_0665p00_ba]
# mean_bias_zi_Venise = [mean_bias_0412p50_zi_Venise,mean_bias_0442p50_zi_Venise,mean_bias_0490p00_zi_Venise,mean_bias_0560p00_zi_Venise,mean_bias_0665p00_zi_Venise] 
# mean_bias_ba_Venise = [mean_bias_0412p50_ba_Venise,mean_bias_0442p50_ba_Venise,mean_bias_0490p00_ba_Venise,mean_bias_0560p00_ba_Venise,mean_bias_0665p00_ba_Venise]
# mean_bias_zi_Gloria = [mean_bias_0412p50_zi_Gloria,mean_bias_0442p50_zi_Gloria,mean_bias_0490p00_zi_Gloria,mean_bias_0560p00_zi_Gloria,mean_bias_0665p00_zi_Gloria] 
# mean_bias_ba_Gloria = [mean_bias_0412p50_ba_Gloria,mean_bias_0442p50_ba_Gloria,mean_bias_0490p00_ba_Gloria,mean_bias_0560p00_ba_Gloria,mean_bias_0665p00_ba_Gloria]
# mean_bias_zi_Galata_Platform = [mean_bias_0412p50_zi_Galata_Platform,mean_bias_0442p50_zi_Galata_Platform,mean_bias_0490p00_zi_Galata_Platform,mean_bias_0560p00_zi_Galata_Platform,mean_bias_0665p00_zi_Galata_Platform] 
# mean_bias_ba_Galata_Platform = [mean_bias_0412p50_ba_Galata_Platform,mean_bias_0442p50_ba_Galata_Platform,mean_bias_0490p00_ba_Galata_Platform,mean_bias_0560p00_ba_Galata_Platform,mean_bias_0665p00_ba_Galata_Platform]
# mean_bias_zi_Helsinki_Lighthouse = [mean_bias_0412p50_zi_Helsinki_Lighthouse,mean_bias_0442p50_zi_Helsinki_Lighthouse,mean_bias_0490p00_zi_Helsinki_Lighthouse,mean_bias_0560p00_zi_Helsinki_Lighthouse,mean_bias_0665p00_zi_Helsinki_Lighthouse] 
# mean_bias_ba_Helsinki_Lighthouse = [mean_bias_0412p50_ba_Helsinki_Lighthouse,mean_bias_0442p50_ba_Helsinki_Lighthouse,mean_bias_0490p00_ba_Helsinki_Lighthouse,mean_bias_0560p00_ba_Helsinki_Lighthouse,mean_bias_0665p00_ba_Helsinki_Lighthouse]
# mean_bias_zi_Gustav_Dalen_Tower = [mean_bias_0412p50_zi_Gustav_Dalen_Tower,mean_bias_0442p50_zi_Gustav_Dalen_Tower,mean_bias_0490p00_zi_Gustav_Dalen_Tower,mean_bias_0560p00_zi_Gustav_Dalen_Tower,mean_bias_0665p00_zi_Gustav_Dalen_Tower] 
# mean_bias_ba_Gustav_Dalen_Tower = [mean_bias_0412p50_ba_Gustav_Dalen_Tower,mean_bias_0442p50_ba_Gustav_Dalen_Tower,mean_bias_0490p00_ba_Gustav_Dalen_Tower,mean_bias_0560p00_ba_Gustav_Dalen_Tower,mean_bias_0665p00_ba_Gustav_Dalen_Tower]    

# wv = [412.5,442.5,490.0,560.0,665.0]
# plt.figure()
# plt.plot(wv,mean_bias_zi_Venise,'-+r',**kwargs)
# plt.plot(wv,mean_bias_ba_Venise,'-xr',**kwargs)
# plt.plot(wv,mean_bias_zi_Gloria,'-+g',**kwargs)
# plt.plot(wv,mean_bias_ba_Gloria,'-xg',**kwargs)
# plt.plot(wv,mean_bias_zi_Galata_Platform,'-+b',**kwargs)
# plt.plot(wv,mean_bias_ba_Galata_Platform,'-xb',**kwargs)
# plt.plot(wv,mean_bias_zi_Helsinki_Lighthouse,'-+m',**kwargs)
# plt.plot(wv,mean_bias_ba_Helsinki_Lighthouse,'-xm',**kwargs)
# plt.plot(wv,mean_bias_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
# plt.plot(wv,mean_bias_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
# plt.plot(wv,mean_bias_zi,'--+k',**kwargs2)
# plt.plot(wv,mean_bias_ba,'--xk',**kwargs2)
# plt.xlabel('Wavelength [nm]',fontsize=12)
# plt.ylabel('MB',fontsize=12)
# # plt.legend(['Z09','BW06'],fontsize=12)
# plt.show()    

# ofname = 'OLCI_mean_bias_same.pdf'
# ofname = os.path.join(path_out,'source',ofname)   
# plt.savefig(ofname, dpi=300) 

# # mean_abs_error
# mean_abs_error_zi = [mean_abs_error_0412p50_zi,mean_abs_error_0442p50_zi,mean_abs_error_0490p00_zi,\
#     mean_abs_error_0560p00_zi,mean_abs_error_0665p00_zi]
# mean_abs_error_ba = [mean_abs_error_0412p50_ba,mean_abs_error_0442p50_ba,mean_abs_error_0490p00_ba,\
#     mean_abs_error_0560p00_ba,mean_abs_error_0665p00_ba]
# mean_abs_error_zi_Venise = [mean_abs_error_0412p50_zi_Venise,mean_abs_error_0442p50_zi_Venise,mean_abs_error_0490p00_zi_Venise,mean_abs_error_0560p00_zi_Venise,mean_abs_error_0665p00_zi_Venise] 
# mean_abs_error_ba_Venise = [mean_abs_error_0412p50_ba_Venise,mean_abs_error_0442p50_ba_Venise,mean_abs_error_0490p00_ba_Venise,mean_abs_error_0560p00_ba_Venise,mean_abs_error_0665p00_ba_Venise]
# mean_abs_error_zi_Gloria = [mean_abs_error_0412p50_zi_Gloria,mean_abs_error_0442p50_zi_Gloria,mean_abs_error_0490p00_zi_Gloria,mean_abs_error_0560p00_zi_Gloria,mean_abs_error_0665p00_zi_Gloria] 
# mean_abs_error_ba_Gloria = [mean_abs_error_0412p50_ba_Gloria,mean_abs_error_0442p50_ba_Gloria,mean_abs_error_0490p00_ba_Gloria,mean_abs_error_0560p00_ba_Gloria,mean_abs_error_0665p00_ba_Gloria]
# mean_abs_error_zi_Galata_Platform = [mean_abs_error_0412p50_zi_Galata_Platform,mean_abs_error_0442p50_zi_Galata_Platform,mean_abs_error_0490p00_zi_Galata_Platform,mean_abs_error_0560p00_zi_Galata_Platform,mean_abs_error_0665p00_zi_Galata_Platform] 
# mean_abs_error_ba_Galata_Platform = [mean_abs_error_0412p50_ba_Galata_Platform,mean_abs_error_0442p50_ba_Galata_Platform,mean_abs_error_0490p00_ba_Galata_Platform,mean_abs_error_0560p00_ba_Galata_Platform,mean_abs_error_0665p00_ba_Galata_Platform]
# mean_abs_error_zi_Helsinki_Lighthouse = [mean_abs_error_0412p50_zi_Helsinki_Lighthouse,mean_abs_error_0442p50_zi_Helsinki_Lighthouse,mean_abs_error_0490p00_zi_Helsinki_Lighthouse,mean_abs_error_0560p00_zi_Helsinki_Lighthouse,mean_abs_error_0665p00_zi_Helsinki_Lighthouse] 
# mean_abs_error_ba_Helsinki_Lighthouse = [mean_abs_error_0412p50_ba_Helsinki_Lighthouse,mean_abs_error_0442p50_ba_Helsinki_Lighthouse,mean_abs_error_0490p00_ba_Helsinki_Lighthouse,mean_abs_error_0560p00_ba_Helsinki_Lighthouse,mean_abs_error_0665p00_ba_Helsinki_Lighthouse]
# mean_abs_error_zi_Gustav_Dalen_Tower = [mean_abs_error_0412p50_zi_Gustav_Dalen_Tower,mean_abs_error_0442p50_zi_Gustav_Dalen_Tower,mean_abs_error_0490p00_zi_Gustav_Dalen_Tower,mean_abs_error_0560p00_zi_Gustav_Dalen_Tower,mean_abs_error_0665p00_zi_Gustav_Dalen_Tower] 
# mean_abs_error_ba_Gustav_Dalen_Tower = [mean_abs_error_0412p50_ba_Gustav_Dalen_Tower,mean_abs_error_0442p50_ba_Gustav_Dalen_Tower,mean_abs_error_0490p00_ba_Gustav_Dalen_Tower,mean_abs_error_0560p00_ba_Gustav_Dalen_Tower,mean_abs_error_0665p00_ba_Gustav_Dalen_Tower]    

# wv = [412.5,442.5,490.0,560.0,665.0]
# plt.figure()
# plt.plot(wv,mean_abs_error_zi_Venise,'-+r',**kwargs)
# plt.plot(wv,mean_abs_error_ba_Venise,'-xr',**kwargs)
# plt.plot(wv,mean_abs_error_zi_Gloria,'-+g',**kwargs)
# plt.plot(wv,mean_abs_error_ba_Gloria,'-xg',**kwargs)
# plt.plot(wv,mean_abs_error_zi_Galata_Platform,'-+b',**kwargs)
# plt.plot(wv,mean_abs_error_ba_Galata_Platform,'-xb',**kwargs)
# plt.plot(wv,mean_abs_error_zi_Helsinki_Lighthouse,'-+m',**kwargs)
# plt.plot(wv,mean_abs_error_ba_Helsinki_Lighthouse,'-xm',**kwargs)
# plt.plot(wv,mean_abs_error_zi_Gustav_Dalen_Tower,'-+c',**kwargs)
# plt.plot(wv,mean_abs_error_ba_Gustav_Dalen_Tower,'-xc',**kwargs)
# plt.plot(wv,mean_abs_error_zi,'--+k',**kwargs2)
# plt.plot(wv,mean_abs_error_ba,'--xk',**kwargs2)
# plt.xlabel('Wavelength [nm]',fontsize=12)
# plt.ylabel('$MAD$',fontsize=12)
# # plt.legend(['Z09','BW06'],fontsize=12)
# plt.show()    

# ofname = 'OLCI_mean_abs_error_same.pdf'
# ofname = os.path.join(path_out,'source',ofname)   
# plt.savefig(ofname, dpi=300) 
    
#%% time series base on conditions
# cond1 = np.array(mu_Lwn_0412p50_fq_ins_zi_station) == 'Venise' 
# idx=np.where(cond1)
# time_vec = np.array(mu_Lwn_0412p50_fq_sat_zi_stop_time)
# sat_vec = np.array(mu_Lwn_0412p50_fq_ins_zi)
# station_vec = np.array(mu_Lwn_0412p50_fq_ins_zi_station)
# plt.figure()
# plt.plot(time_vec,sat_vec,'o',mfc='none')
# plt.plot(time_vec[idx],sat_vec[idx],'ro',mfc='none')
#%%
def plot_hist_delta(station,n_matchups,df0):
    
    # if protocol_name == 'zi':
    #     protocol_str = 'Z09'
    # elif protocol_name == 'ba':
    #     protocol_str = 'BW06'
    
    protocol_list = ['zi','ba']
    
    for protocol_name in protocol_list:    
    
        time_vec = np.array(globals()['mu_Lwn_0412p50_fq_sat_'+protocol_name+'_stop_time'])
        date_vec = [dt.date() for dt in time_vec]
        date_vec = np.array(date_vec)
        
        station_vec = np.array(globals()['mu_Lwn_0412p50_fq_ins_'+protocol_name+'_station'])
    
        cond1 = station_vec == station
        
        date_vec = date_vec[cond1]
        date_vec = np.sort(date_vec)
        
        i = 0
        
        date_diff = []
        
        while i < len(date_vec)-n_matchups:
            date_diff.append((date_vec[i+n_matchups]-date_vec[i]).days)
            i += 1
        globals()['date_diff_'+protocol_name]=date_diff
        
    # histograms of both dataset: zi and ba
    kwargs2 = dict(bins='auto', histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w',figsize=(5,6))
    ax1.hist(date_diff_zi,color='red', **kwargs2)
    ax1.hist(date_diff_ba,color='black', **kwargs2)
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim([x0,x0+1*(x1-x0)])

    ax1.set_ylabel('Frequency (counts)',fontsize=12)
    ax1.set_xlabel('Days needed to have '+str(n_matchups)+' matchups.',fontsize=12)
    plt.title(f'{station.replace("_"," ")} Station')
    
    str1 = 'Z09\nmin: {:,.0f}\nmax: {:,.0f}\nmedian: {:,.0f}\nmean: {:,.0f}\nN: {:,.0f}'\
    .format(np.nanmin(date_diff_zi),
            np.nanmax(date_diff_zi),
            np.nanmedian(date_diff_zi),
            np.nanmean(date_diff_zi),
            len(date_diff_zi))

    str2 = 'BW06\nmin: {:,.0f}\nmax: {:,.0f}\nmedian: {:,.0f}\nmean: {:,.0f}\nN: {:,.0f}'\
    .format(np.nanmin(date_diff_ba),
            np.nanmax(date_diff_ba),
            np.nanmedian(date_diff_ba),
            np.nanmean(date_diff_ba),
            len(date_diff_ba))

    bottom, top = ax1.get_ylim()
    left, right = ax1.get_xlim()
    ypos = bottom+0.75*(top-bottom)
    ax1.text(left+0.45*(right-left),ypos, str2, fontsize=11,color='black')
    ax1.text(left+0.73*(right-left),ypos, str1, fontsize=11,color='red')
    
    

    # save fig
    ofname = sensor_name+'_hist_ba_zi_goal_'+station+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()
    
    df0 = df0.append(dict(mininum=np.nanmin(date_diff_zi), \
                          maximum=np.nanmax(date_diff_zi),\
                              median=np.nanmedian(date_diff_zi),\
                                  mean=np.nanmean(date_diff_zi),\
                                      N=len(date_diff_zi),\
                                          station=station,\
                                              protocol='Z09',\
                                                  median_diff=0),
                     ignore_index=True) 
    df0 = df0.append(dict(mininum=np.nanmin(date_diff_ba),\
                          maximum=np.nanmax(date_diff_ba),\
                              median=np.nanmedian(date_diff_ba),\
                                  mean=np.nanmean(date_diff_ba),\
                                      N=len(date_diff_ba),\
                                          station=station,\
                                              protocol='BW06',\
                                                  median_diff=(np.nanmedian(date_diff_zi)-np.nanmedian(date_diff_ba))),\
                     ignore_index=True)
    return df0

df0 = pd.DataFrame()
station_list = ['Venise','Galata_Platform','Gloria']
n_matchups = 30
if plot_flag:
    for station in station_list:
        df0 = plot_hist_delta(station,n_matchups,df0)
        
    df = df0[['station','protocol','mean','median','median_diff']]    
    print(tabulate(df, tablefmt='pipe', headers='keys'))
    
    print(tabulate(df, tablefmt='latex', headers='keys'))

#%%
def plot_histogram_and_qq(points, mu, sigma, distribution_type="norm"):
  # Plot histogram of the 1000 points
  plt.figure(figsize=(12,6))
  ax = plt.subplot(1,2,1)
  count, bins, ignored = plt.hist(points, 30, normed=True)
  ax.set_title('Histogram')
  ax.set_xlabel('Value bin')
  ax.set_ylabel('Frequency')

  # Overlay the bell curve (normal distribution) on the bins data
  bell_curve = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
  plt.plot(bins, bell_curve, linewidth=2, color='r')

  # Q-Q plot
  plt.subplot(1,2,2)
  res = stats.probplot(points, dist=distribution_type, plot=plt)
  (osm, osr) = res[0]
  (slope, intercept, r) = res[1]
  # For details see: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.probplot.html
  print("slope, intercept, r:", slope, intercept, r)
  print("r is the square root of the coefficient of determination")

  plt.show()

#%% print potential matchups numbers
print('Station & BW06 & Z09')
print(f'Venise & {pot_mu_cnt_ba_Venise} & {pot_mu_cnt_zi_Venise}')
print(f'Gloria & {pot_mu_cnt_ba_Gloria} & {pot_mu_cnt_zi_Gloria}')
print(f'Galata_Platform & {pot_mu_cnt_ba_Galata_Platform} & {pot_mu_cnt_zi_Galata_Platform}')
print(f'Helsinki_Lighthouse & {pot_mu_cnt_ba_Helsinki_Lighthouse} & {pot_mu_cnt_zi_Helsinki_Lighthouse}')
print(f'Gustav_Dalen_Tower & {pot_mu_cnt_ba_Gustav_Dalen_Tower} & {pot_mu_cnt_zi_Gustav_Dalen_Tower}')
print(f'Total & {pot_mu_cnt_ba} & {pot_mu_cnt_zi}')


data = {'stations':['Venise','Gloria','Galata_Platform','Helsinki_Lighthouse','Gustav_Dalen_Tower','Total']}
df3 = pd.DataFrame(data)
df3['cvs_ba'] = [rej_cvs_mu_cnt_ba_Venise,rej_cvs_mu_cnt_ba_Gloria,rej_cvs_mu_cnt_ba_Galata_Platform,rej_cvs_mu_cnt_ba_Helsinki_Lighthouse,rej_cvs_mu_cnt_ba_Gustav_Dalen_Tower,rej_cvs_mu_cnt_ba]
df3['ang_ba'] = [rej_ang_mu_cnt_ba_Venise,rej_ang_mu_cnt_ba_Gloria,rej_ang_mu_cnt_ba_Galata_Platform,rej_ang_mu_cnt_ba_Helsinki_Lighthouse,rej_ang_mu_cnt_ba_Gustav_Dalen_Tower,rej_ang_mu_cnt_ba]
df3['sza_ba'] = [rej_sza_mu_cnt_ba_Venise,rej_sza_mu_cnt_ba_Gloria,rej_sza_mu_cnt_ba_Galata_Platform,rej_sza_mu_cnt_ba_Helsinki_Lighthouse,rej_sza_mu_cnt_ba_Gustav_Dalen_Tower,rej_sza_mu_cnt_ba]
df3['vza_ba'] = [rej_vza_mu_cnt_ba_Venise,rej_vza_mu_cnt_ba_Gloria,rej_vza_mu_cnt_ba_Galata_Platform,rej_vza_mu_cnt_ba_Helsinki_Lighthouse,rej_vza_mu_cnt_ba_Gustav_Dalen_Tower,rej_vza_mu_cnt_ba]
df3['inv_ba'] = [rej_inv_mu_cnt_ba_Venise,rej_inv_mu_cnt_ba_Gloria,rej_inv_mu_cnt_ba_Galata_Platform,rej_inv_mu_cnt_ba_Helsinki_Lighthouse,rej_inv_mu_cnt_ba_Gustav_Dalen_Tower,rej_inv_mu_cnt_ba]
df3['cvs_zi'] = [rej_cvs_mu_cnt_zi_Venise,rej_cvs_mu_cnt_zi_Gloria,rej_cvs_mu_cnt_zi_Galata_Platform,rej_cvs_mu_cnt_zi_Helsinki_Lighthouse,rej_cvs_mu_cnt_zi_Gustav_Dalen_Tower,rej_cvs_mu_cnt_zi]
df3['cvL_zi'] = [rej_cvL_mu_cnt_zi_Venise,rej_cvL_mu_cnt_zi_Gloria,rej_cvL_mu_cnt_zi_Galata_Platform,rej_cvL_mu_cnt_zi_Helsinki_Lighthouse,rej_cvL_mu_cnt_zi_Gustav_Dalen_Tower,rej_cvL_mu_cnt_zi]
df3['cvA_zi'] = [rej_cvA_mu_cnt_zi_Venise,rej_cvA_mu_cnt_zi_Gloria,rej_cvA_mu_cnt_zi_Galata_Platform,rej_cvA_mu_cnt_zi_Helsinki_Lighthouse,rej_cvA_mu_cnt_zi_Gustav_Dalen_Tower,rej_cvA_mu_cnt_zi]
df3['ang_zi'] = [rej_ang_mu_cnt_zi_Venise,rej_ang_mu_cnt_zi_Gloria,rej_ang_mu_cnt_zi_Galata_Platform,rej_ang_mu_cnt_zi_Helsinki_Lighthouse,rej_ang_mu_cnt_zi_Gustav_Dalen_Tower,rej_ang_mu_cnt_zi]
df3['sza_zi'] = [rej_sza_mu_cnt_zi_Venise,rej_sza_mu_cnt_zi_Gloria,rej_sza_mu_cnt_zi_Galata_Platform,rej_sza_mu_cnt_zi_Helsinki_Lighthouse,rej_sza_mu_cnt_zi_Gustav_Dalen_Tower,rej_sza_mu_cnt_zi]
df3['vza_zi'] = [rej_vza_mu_cnt_zi_Venise,rej_vza_mu_cnt_zi_Gloria,rej_vza_mu_cnt_zi_Galata_Platform,rej_vza_mu_cnt_zi_Helsinki_Lighthouse,rej_vza_mu_cnt_zi_Gustav_Dalen_Tower,rej_vza_mu_cnt_zi]
df3['inv_zi'] = [rej_inv_mu_cnt_zi_Venise,rej_inv_mu_cnt_zi_Gloria,rej_inv_mu_cnt_zi_Galata_Platform,rej_inv_mu_cnt_zi_Helsinki_Lighthouse,rej_inv_mu_cnt_zi_Gustav_Dalen_Tower,rej_inv_mu_cnt_zi]

print(tabulate(df3, tablefmt='latex', headers='keys',showindex=False))
   
#%% band triggered
if plot_flag:
    fig, ax = plt.subplots()
    x = np.arange(5)
    y = [sum(np.array(idx_medianCV)==0),sum(np.array(idx_medianCV)==1),sum(np.array(idx_medianCV)==2),sum(np.array(idx_medianCV)==3),sum(np.array(idx_medianCV)==4)]
    plt.bar(x,y)
    plt.xticks(x, ('$L_{WN}(412.5)$', '$L_{WN}(442.5)$', '$L_{WN}(490)$', '$L_{WN}(560)$', '$A_{OT}(865.5)$'))
    plt.title('Band that triggered the Median(CV) criteria for BW06')
    plt.xlabel('OLCI Bands')
    plt.ylabel('Frequency (counts)')
    plt.show()    
#%%    
# data = stats.norm.rvs(loc=5, scale=3, size=(450,))
# cdf_plot(data)
# x = np.arange(np.min(data),np.max(data),(np.max(data)-np.min(data))/100)
# norm_dist = stats.norm.pdf(x,np.mean(data),np.std(data))
# plt.figure()
# plt.plot(x,norm_dist)
# cdf_plot(norm_dist)
#%% plotly example
if plot_flag:
    from plotly.offline import plot
    import plotly.graph_objs as go
    
    fig = go.Figure(data=[go.Bar(y=[1, 3, 2])])
    plot(fig, auto_open=True)
    
#%% x and y given as array_like objects
if plot_flag:
    from plotly.offline import plot
    import plotly.express as px
    
     
    
    time_vec = np.array(mu_Lwn_0412p50_fq_sat_zi_stop_time)
    sat_vec = np.array(mu_Lwn_0412p50_fq_sat_zi)
    station_vec = np.array(mu_Lwn_0412p50_fq_ins_zi_station)
    df1 = pd.DataFrame(dict(time=time_vec, sat=sat_vec, station=station_vec,protocol='Z09'))    
        
        
    time_vec = np.array(mu_Lwn_0412p50_fq_sat_ba_stop_time)
    sat_vec = np.array(mu_Lwn_0412p50_fq_sat_ba)
    station_vec = np.array(mu_Lwn_0412p50_fq_ins_ba_station)
    df2 = pd.DataFrame(dict(time=time_vec, sat=sat_vec, station=station_vec,protocol='BW06'))
    df = pd.concat([df1,df2])
    
    fig = px.scatter(df1,x=df.time,y=df.sat,color=df.station,symbol=df.protocol)
    fig.update_traces(mode="markers", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    plot(fig, auto_open=True)

#%% histograms of both delta time: zi and ba
# delta time  => time_diff = ins_time - sat_stop_time
if plot_flag:
    kwargs2 = dict(histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w',figsize=(8,6))
    
    # hist, bins = np.histogram(np.array(dt_mu_zi)*60)
    # ax1.bar(bins[:-1], 100*hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='red')
    # hist, bins = np.histogram(np.array(dt_mu_ba)*60)
    # ax1.bar(bins[:-1], 100*hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='black')
    bins_Dt = [-180,-165,-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90,105,120,135,150.65,180]
    
    counts_zi, bins_zi = np.histogram(np.array(dt_mu_zi)*60,bins=bins_Dt)
    ax1.hist(bins_zi[:-1], bins_zi, weights=100*counts_zi/counts_zi.sum(),color='red', **kwargs2)
    
    counts_ba, bins_ba = np.histogram(np.array(dt_mu_ba)*60,bins=bins_Dt)
    ax1.hist(bins_ba[:-1], bins_ba, weights=100*counts_ba/counts_ba.sum(),color='black', **kwargs2)
    
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim([x0,x0+1*(x1-x0)])
    
    ax1.set_ylabel('Frequency (%)',fontsize=12)
    ax1.set_xlabel('Delta time (minutes)',fontsize=12)
    plt.xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])
    plt.legend(['Z09','BW06'],fontsize=12)
# plt.title(f'{station.replace("_"," ")} Station')
#%% maps valid pixels
if plot_flag:
    for station in station_list_main:
        station_idx = station_list_main.index(station)
        fig = plt.figure()
        plt.subplot(2,3,1)
        plt.imshow(map_valid_pxs_ba[station_idx,0,:,:],interpolation='none')
        plt.colorbar()
        plt.title('BW06: 412.5 nm')
        
        plt.subplot(2,3,2)
        plt.imshow(map_valid_pxs_ba[station_idx,1,:,:],interpolation='none')
        plt.colorbar()
        plt.title('BW06: 442.5 nm')
        
        plt.subplot(2,3,3)
        plt.imshow(map_valid_pxs_ba[station_idx,2,:,:],interpolation='none')
        plt.colorbar()
        plt.title('BW06: 490 nm')
        
        plt.subplot(2,3,4)
        plt.imshow(map_valid_pxs_ba[station_idx,3,:,:],interpolation='none')
        plt.colorbar()
        plt.title('BW06: 560 nm')
        
        plt.subplot(2,3,5)
        plt.imshow(map_valid_pxs_ba[station_idx,4,:,:],interpolation='none')
        plt.colorbar()
        plt.title('BW06: 665 nm')
        
        title_str = f'OLCI Valid Pixels Maps - {station}'
        fig.suptitle(title_str)
        
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()    
        
        plt.savefig(os.path.join(path_out,title_str.replace(' ','_')+'.png'))
        
        plt.close()
#%% plot number of used pixels for BW06
if plot_flag:
    plt.figure()
    
    for sat_band_index in range(number_used_pixels.shape[1]):
        plt.plot(number_used_pixels[:,sat_band_index],c=color_dict[f'{olci_wl_list[sat_band_index]:.2f}'])
    
    wl_str = ['412.5','442.5','490.0','560.0','665.0']
    plt.legend(wl_str,fontsize=12)
    plt.xlabel('Match-up Number',fontsize=12)
    plt.ylabel('Number of pixels used',fontsize=12)
    
    
    for idx in range(number_used_pixels.shape[1]):
    
        kwargs2 = dict(histtype='step')
        fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w',figsize=(8,6))
        bins = [12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    
    
    
        counts, bins = np.histogram(number_used_pixels[:,idx],bins=bins)
        ax1.hist(bins[:-1], bins, weights=100*counts/counts.sum(),color='red', **kwargs2)
        plt.title(f'{olci_wl_list[idx]} nm',fontsize=12)
        plt.xlabel('Number of Pixels Used',fontsize=12)
        plt.ylabel('Frequency (%)',fontsize=12)
#%% plot CVS
if plot_flag:
    CVs_ba_bands_list =[ 'rhow_412.5','rhow_442.5','rhow_490.0','rhow_560.0','rhow_665.0','aot_865.5']
    
    kwargs2 = dict(histtype='step')
    bins = [0.0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20]
    
    dataset_name = 'BW06_notZ09' # all, all_MUs, common, notBW06_Z09, BW06_notZ09  ((df_matchups['BW06_MU']==True) | (df_matchups['Z09_MU']==True))
    savefig_flag = False
    for station_idx in range(len(station_list_main)):
        station = station_list_main[station_idx]
        if dataset_name == 'BW06_notZ09':
            df = df_matchups.loc[(df_matchups['station']==station) & ((df_matchups['BW06_MU']==True) & (df_matchups['Z09_MU']==False))]
        elif dataset_name == 'notBW06_Z09':
            df = df_matchups.loc[(df_matchups['station']==station) & ((df_matchups['BW06_MU']==False) & (df_matchups['Z09_MU']==True))]
        elif dataset_name == 'all_MUs':
            df = df_matchups.loc[(df_matchups['station']==station) & ((df_matchups['BW06_MU']==True) | (df_matchups['Z09_MU']==True))]
        elif dataset_name == 'common':
            df = df_matchups.loc[(df_matchups['station']==station) & ((df_matchups['BW06_MU']==True) & (df_matchups['Z09_MU']==True))]
        elif dataset_name == 'all':
            df = df_matchups.loc[(df_matchups['station']==station)]        
            
        plt.figure(figsize=(20,8))
        for band in CVs_ba_bands_list:
            plt.plot(df['sat_datetime'],df['BW06: CV_'+band],c=color_dict[f'{float(band[-5:]):.2f}'],linestyle='none',marker='.')
        plt.plot(df['sat_datetime'],df['BW06: MedianCV'],'k')
        
        plt.plot(df['sat_datetime'],df['Z09: CV_560'],c=color_dict[f'{float(560.0):.2f}'],linestyle='-',marker='o',fillstyle='none')
        plt.plot(df['sat_datetime'],df['Z09: CV_865p5'],c=color_dict[f'{float(865.5):.2f}'],linestyle='-',marker='o',fillstyle='none')
        plt.ylim([-0.2,0.2])
        legend_items =['BW06: rhow_412.5','BW06: rhow_442.5','BW06: rhow_490.0','BW06: rhow_560.0','BW06: rhow_665.0','BW06: aot_865.5',\
                    'BW06: MedianCV','Z09: Lwn_560.0','Z09: aot_865.5']
        plt.legend(legend_items)
        plt.title(f'{station_list_main[station_idx]}; Dataset: {dataset_name}')
        plt.xlabel('time')
        plt.ylabel('CV')
        if savefig_flag:
            plt.savefig(os.path.join(path_out,f'CVs_ba_zi_timeseriet_{station_list_main[station_idx]}_{dataset_name}.png'))
            plt.close()
     
        fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w',figsize=(8,6))
        counts, bins2 = np.histogram(df['BW06: MedianCV'],bins=bins)
        ax1.hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='red', **kwargs2)
        str1 = f'BW06: MedianCV (N={counts.sum()})'
        counts, bins2 = np.histogram(df['Z09: CV_560'],bins=bins)
        ax1.hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='green', **kwargs2)
        counts, bins2 = np.histogram(df['Z09: CV_865p5'],bins=bins)
        str2 = f'Z09: Lwn_560.0 (N={counts.sum()})'
        ax1.hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='blue', **kwargs2)
        str3 = f'Z09: aot_865.5 (N={counts.sum()})'
        plt.title(f'{station_list_main[station_idx]}; Dataset: {dataset_name}')
        plt.legend([str1,str2,str3])
        plt.xlim([0,0.2])
        plt.xlabel('CV')
        plt.ylabel('Frequency (%)',fontsize=12)
        if savefig_flag:
            plt.savefig(os.path.join(path_out,f'CVs_ba_zi_hist_{station_list_main[station_idx]}_{dataset_name}.png'))
            plt.close()
        
        fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w',figsize=(8,6))
        counts, bins2 = np.histogram(df['BW06: CV_rhow_560.0'],bins=bins)
        ax1.hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='red', **kwargs2)
        str1 = f'BW06: rhow_560.0 (N={counts.sum()})'
        counts, bins2 = np.histogram(df['Z09: CV_560'],bins=bins)
        ax1.hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='green', **kwargs2)
        str2 = f'Z09: Lwn_560.0 (N={counts.sum()})'
        plt.title(f'{station_list_main[station_idx]}; Dataset: {dataset_name}')
        plt.legend([str1,str2])  
        plt.xlim([0,0.2])
        plt.xlabel('CV')
        plt.ylabel('Frequency (%)',fontsize=12)
        if savefig_flag:
            plt.savefig(os.path.join(path_out,f'CVs_ba_zi_560_hist_{station_list_main[station_idx]}_{dataset_name}.png'))           
            plt.close()
#%% plot matchups
# df_matchups = df_matchups.append({'station':station_list_main[station_idx],'sat_datetime':sat_stop_time,'insitu_datetime':ins_time[idx_min],'vza':vza,'sza':sza,\
#          'BW06_MU':BW06_MU,'BW06_l2_mask':flags_mask_ba,\
#          'BW06: rhow_412_box':rhow_0412p50_fq_box_ba,'BW06: rho_412_filt_mean':mean_filtered_rhow_0412p50,\
#          'BW06: rhow_442_box':rhow_0442p50_fq_box_ba,'BW06: rho_442_filt_mean':mean_filtered_rhow_0442p50,\
#          'BW06: rhow_490_box':rhow_0490p00_fq_box_ba,'BW06: rho_490_filt_mean':mean_filtered_rhow_0490p00,\
#          'BW06: rhow_560_box':rhow_0560p00_fq_box_ba,'BW06: rho_560_filt_mean':mean_filtered_rhow_0560p00,\
#          'BW06: rhow_665_box':rhow_0665p00_fq_box_ba,'BW06: rho_665_filt_mean':mean_filtered_rhow_0665p00,\
#          'BW06: MedianCV':MedianCV,'BW06: Nfilt_560':rhow_0560p00_fq_box_ba.count(),'BW06: NGP':NGP,'BW06: NTP':NTP,\
#          'BW06: CV_rhow_412.5':CV_filtered_rhow_0412p50,'BW06: CV_rhow_442.5':CV_filtered_rhow_0442p50,\
#          'BW06: CV_rhow_490.0':CV_filtered_rhow_0490p00,'BW06: CV_rhow_560.0':CV_filtered_rhow_0560p00,\
#          'BW06: CV_rhow_665.0':CV_filtered_rhow_0665p00,'BW06: CV_aot_865.5':CV_filtered_AOT_0865p50,\
#          'BW06: MedianCV_band_idx':idx_m,\
#          'Z09_MU':Z09_MU,'Z09_l2_mask':flags_mask_zi,\
#          'Z09: rhow_412_box':rhow_0412p50_fq_box_zi,'Z09: rho_412_mean':rhow_0412p50_fq_box_zi.mean(),\
#          'Z09: rhow_442_box':rhow_0442p50_fq_box_zi,'Z09: rho_442_mean':rhow_0442p50_fq_box_zi.mean(),\
#          'Z09: rhow_490_box':rhow_0490p00_fq_box_zi,'Z09: rho_490_mean':rhow_0490p00_fq_box_zi.mean(),\
#          'Z09: rhow_560_box':rhow_0560p00_fq_box_zi,'Z09: rho_560_mean':rhow_0560p00_fq_box_zi.mean(),\
#          'Z09: rhow_665_box':rhow_0665p00_fq_box_zi,'Z09: rho_665_mean':rhow_0665p00_fq_box_zi.mean(),\
#          'Z09: CV_560':Lwn_560_CV,'Z09: CV_865p5':AOT_0865p50_CV},ignore_index=True) 

# for idx, row in df_matchups.iterrows()[0]:
#     if row['BW06: MU'] == True:
#         print(row['BW06: rhow_560_box'])

core_pxs_count_412 = []
core_pxs_count_442 = []
core_pxs_count_490 = []
core_pxs_count_560 = []
core_pxs_count_665 = []

if True or plot_flag:
    rhow_412_core_incl_cnt = 0
    rhow_442_core_incl_cnt = 0
    rhow_490_core_incl_cnt = 0
    rhow_560_core_incl_cnt = 0
    rhow_665_core_incl_cnt = 0
            
    for idx in range(len(df_matchups)): 
    # for idx in range(3):
        if df_matchups.loc[idx,'BW06_MU'] == True and df_matchups.loc[idx,'Z09_MU'] == True:
            print('---------------')
            # print(df_matchups.loc[idx,'BW06: rhow_560_box'])
    
            rhow_412_core_incl = False
            rhow_442_core_incl = False
            rhow_490_core_incl = False
            rhow_560_core_incl = False
            rhow_665_core_incl = False
            
            # to know how many pixels of Z09 are included in the calculation of BW09. Only if (df_matchups.loc[idx,'BW06_MU'] == True and df_matchups.loc[idx,'Z09_MU'] == True)
            core_pxs_count_412.append(df_matchups.loc[idx,'BW06: rhow_412_box'][1:4,1:4].count())
            core_pxs_count_442.append(df_matchups.loc[idx,'BW06: rhow_442_box'][1:4,1:4].count())
            core_pxs_count_490.append(df_matchups.loc[idx,'BW06: rhow_490_box'][1:4,1:4].count())
            core_pxs_count_560.append(df_matchups.loc[idx,'BW06: rhow_560_box'][1:4,1:4].count())
            core_pxs_count_665.append(df_matchups.loc[idx,'BW06: rhow_665_box'][1:4,1:4].count())
            
            # to know how many time the whole core of BW06 is included in the calculation
            if (not df_matchups.loc[idx,'BW06: rhow_412_box'][1:4,1:4].mask.any()) \
                and (df_matchups.loc[idx,'BW06_MU']==True and (df_matchups.loc[idx,'Z09_MU'])):
                rhow_412_core_incl = True
                rhow_412_core_incl_cnt += 1
            if (not df_matchups.loc[idx,'BW06: rhow_442_box'][1:4,1:4].mask.any()) \
                and (df_matchups.loc[idx,'BW06_MU']==True and (df_matchups.loc[idx,'Z09_MU'])):
                rhow_442_core_incl = True
                rhow_442_core_incl_cnt += 1
            if (not df_matchups.loc[idx,'BW06: rhow_490_box'][1:4,1:4].mask.any()) \
                and (df_matchups.loc[idx,'BW06_MU']==True and (df_matchups.loc[idx,'Z09_MU'])):
                rhow_490_core_incl = True
                rhow_490_core_incl_cnt += 1
            if (not df_matchups.loc[idx,'BW06: rhow_560_box'][1:4,1:4].mask.any()) \
                and (df_matchups.loc[idx,'BW06_MU']==True and (df_matchups.loc[idx,'Z09_MU'])):
                rhow_560_core_incl = True
                rhow_560_core_incl_cnt += 1
            if (not df_matchups.loc[idx,'BW06: rhow_665_box'][1:4,1:4].mask.any()) \
                and (df_matchups.loc[idx,'BW06_MU']==True and (df_matchups.loc[idx,'Z09_MU'])):
                rhow_665_core_incl = True
                rhow_665_core_incl_cnt += 1
           
            if plot_flag:
                fig = plt.figure(figsize=(20,8))
                date_str = str(df_matchups.loc[idx,'sat_datetime'].date())
                print(df_matchups.loc[idx,"station"])
                print(date_str)
                plt.suptitle(f'{date_str}, {df_matchups.loc[idx,"station"]}')
        
                # Z09
                ax0 = plt.subplot(4,5,1)
                ax0.imshow(df_matchups.loc[idx,'Z09: rhow_412_box'])
                ax0.set_xlim([-1.5,3.5])
                ax0.set_ylim([3.5,-1.5])
                plt.axis('off')
                plt.title('Z09: rhow_412')
                ax0.annotate((f'\nCore incl: {str(rhow_412_core_incl)}'), (-1.0,3.0), textcoords='data', size=10)
        
                ax0 = plt.subplot(4,5,2)
                ax0.imshow(df_matchups.loc[idx,'Z09: rhow_442_box'])
                ax0.set_xlim([-1.5,3.5])
                ax0.set_ylim([3.5,-1.5])
                plt.axis('off')
                plt.title('Z09: rhow_442')
                ax0.annotate((f'\nCore incl: {str(rhow_442_core_incl)}'), (-1.0,3.0), textcoords='data', size=10)
        
                ax0 = plt.subplot(4,5,6)
                ax0.imshow(df_matchups.loc[idx,'Z09: rhow_490_box'])
                ax0.set_xlim([-1.5,3.5])
                ax0.set_ylim([3.5,-1.5])
                plt.axis('off')
                plt.title('Z09: rhow_490')
                ax0.annotate((f'\nCore incl: {str(rhow_490_core_incl)}'), (-1.0,3.0), textcoords='data', size=10)
        
                ax0 = plt.subplot(4,5,7)
                ax0.imshow(df_matchups.loc[idx,'Z09: rhow_560_box'])
                ax0.set_xlim([-1.5,3.5])
                ax0.set_ylim([3.5,-1.5])
                plt.axis('off')
                plt.title('Z09: rhow_560')
                ax0.annotate((f'\nCore incl: {str(rhow_560_core_incl)}'), (-1.0,3.0), textcoords='data', size=10)
        
                ax0 = plt.subplot(4,5,11)
                ax0.imshow(df_matchups.loc[idx,'Z09: rhow_665_box'])
                ax0.set_xlim([-1.5,3.5])
                ax0.set_ylim([3.5,-1.5])
                plt.axis('off')
                plt.title('Z09: rhow_665')
                ax0.annotate((f'\nCore incl: {str(rhow_665_core_incl)}'), (-1.0,3.0), textcoords='data', size=10)
        
                ax0 = plt.subplot(4,5,12)
                ax0.imshow(df_matchups.loc[idx,'Z09_l2_mask'])
                ax0.set_xlim([-1.5,3.5])
                ax0.set_ylim([3.5,-1.5])
                plt.axis('off')
                plt.title('Z09: L2 mask')
        
                # BW06
                plt.subplot(4,5,3)
                plt.imshow(df_matchups.loc[idx,'BW06: rhow_412_box'])
                plt.axis('off')
                plt.title('BW06: rhow_412')
                    
                plt.subplot(4,5,4)
                plt.imshow(df_matchups.loc[idx,'BW06: rhow_442_box'])
                plt.axis('off')
                plt.title('BW06: rhow_442')
                    
                plt.subplot(4,5,8)
                plt.imshow(df_matchups.loc[idx,'BW06: rhow_490_box'])
                plt.axis('off')
                plt.title('BW06: rhow_490')
                    
                plt.subplot(4,5,9)
                plt.imshow(df_matchups.loc[idx,'BW06: rhow_560_box'])
                plt.axis('off')
                plt.title('BW06: rhow_560')
                    
                plt.subplot(4,5,13)
                plt.imshow(df_matchups.loc[idx,'BW06: rhow_665_box'])
                plt.axis('off')
                plt.title('BW06: rhow_665')
                    
                plt.subplot(4,5,14)
                plt.imshow(df_matchups.loc[idx,'BW06_l2_mask'])
                plt.axis('off')
                plt.title('BW06: L2 mask')
        
                ax1 = plt.subplot(2,5,5)
                ba_MU = zi_MU = 'Rejected'
                if df_matchups.loc[idx,"BW06_MU"]: ba_MU = 'Passed'
                if df_matchups.loc[idx,"Z09_MU"]: zi_MU = 'Passed'
                panel_ba = f'BW06\n{ba_MU}\nNGP: {int(df_matchups.loc[idx,"BW06: NGP"])}\nMedianCV = {df_matchups.loc[idx,"BW06: MedianCV"]:0.3f}\nmean={df_matchups.loc[idx,"BW06: rho_560_filt_mean"]:0.4f}' 
                panel_zi = f'Z09\n{zi_MU}\nCV_560 = {df_matchups.loc[idx,"Z09: CV_560"]:0.3f}\nCV_865.5 = {df_matchups.loc[idx,"Z09: CV_865p5"]:0.3f}\nmean={df_matchups.loc[idx,"Z09: rho_560_mean"]:0.4f}'
                Dt = df_matchups.loc[idx,"sat_datetime"]-df_matchups.loc[idx,"insitu_datetime"]
                panel_str = f'Delta t={abs(Dt).total_seconds()/60/60:0.2f} hours\nvza: {float(df_matchups.loc[idx,"vza"]):0.2f}; sza: {float(df_matchups.loc[idx,"sza"]):0.2f}'
                
                # core_incl_str = \
                # f'\nrhow_412 core incl.: {str(rhow_412_core_incl)}'+\
                # f'\nrhow_442 core incl.: {str(rhow_442_core_incl)}'+\
                # f'\nrhow_490 core incl.: {str(rhow_490_core_incl)}'+\
                # f'\nrhow_560 core incl.: {str(rhow_560_core_incl)}'+\
                # f'\nrhow_665 core incl.: {str(rhow_665_core_incl)}'
        
                ax1.annotate(panel_str, (0.01, 0.95), textcoords='data', size=12)
                ax1.annotate(panel_ba, (0.01, 0.50), textcoords='data', size=12)
                ax1.annotate(panel_zi, (0.01, 0.01), textcoords='data', size=12)
                plt.axis('off')
        
                # individual pixels
                plt.subplot(4,1,4)
                plt.plot(df_matchups.loc[idx,'BW06: rhow_560_box'].ravel(),'r+')
                plt.plot([6,7,8,11,12,13,16,17,18],df_matchups.loc[idx,'Z09: rhow_560_box'].ravel(),'bx')
                plt.plot([0,24],[df_matchups.loc[idx,'BW06: rho_560_filt_mean'],df_matchups.loc[idx,'BW06: rho_560_filt_mean']],'r--')
                plt.plot([0,24],[df_matchups.loc[idx,'Z09: rho_560_mean'],df_matchups.loc[idx,'Z09: rho_560_mean']],'b--')
                plt.xlabel('Pixel Number')
                plt.ylabel(r'$\rho_{W}(560)$')
                plt.legend([r'BW06: $\rho_{W}(560)$',r'Z09: $\rho_{W}(560)$',r'BW06: filt. mean $\rho_{W}(560)$',r'Z09: mean $\rho_{W}(560)$'],\
                           ncol=4,bbox_to_anchor=(0.15, 0.93, 1., .4), loc='lower left',frameon=False)
                
                ofname = f'MU_report_{df_matchups.loc[idx,"station"]}_{date_str}.png'
        
                ofname = os.path.join(path_out,ofname)
                plt.savefig(ofname, dpi=100)
            
            # plt.close()
#%% histogram of how many pixels of Z09 are included in the calculation of BW06  
bins = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
kwargs2 = dict(histtype='step')         
if True or plot_flag:
    fig,axs = plt.subplots(2,3, facecolor='w',figsize=(20,8))
    counts, bins2 = np.histogram(core_pxs_count_412,bins=bins)
    axs[0,0].hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='blue', **kwargs2)
    axs[0,0].set_title(412,x=0.5,y=0.9)
    axs[0,0].set_xlabel('Number of pixels')
    axs[0,0].set_ylabel('Frequency (%)')   

    counts, bins2 = np.histogram(core_pxs_count_442,bins=bins)
    axs[0,1].hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='blue', **kwargs2)
    axs[0,1].set_title(442,x=0.5,y=0.9)
    axs[0,1].set_xlabel('Number of pixels')
    axs[0,1].set_ylabel('Frequency (%)') 

    counts, bins2 = np.histogram(core_pxs_count_490,bins=bins)
    axs[0,2].hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='blue', **kwargs2)
    axs[0,2].set_title(490,x=0.5,y=0.9)
    axs[0,2].set_xlabel('Number of pixels')
    axs[0,2].set_ylabel('Frequency (%)') 

    counts, bins2 = np.histogram(core_pxs_count_560,bins=bins)
    axs[1,0].hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='blue', **kwargs2)
    axs[1,0].set_title(560,x=0.5,y=0.9)
    axs[1,0].set_xlabel('Number of pixels')
    axs[1,0].set_ylabel('Frequency (%)') 

    counts, bins2 = np.histogram(core_pxs_count_665,bins=bins)
    axs[1,1].hist(bins2[:-1], bins2, weights=100*counts/counts.sum(),color='blue', **kwargs2)
    axs[1,1].set_title(665,x=0.5,y=0.9)
    axs[1,1].set_xlabel('Number of pixels')
    axs[1,1].set_ylabel('Frequency (%)')    

    axs[1,2].axis('off')

#%% mean in situ spectra per station
if plot_flag:
    fs = 24
    plt.rc('xtick',labelsize=fs)
    plt.rc('ytick',labelsize=fs)
    for station_idx in range(len(station_list_main)):
        station = station_list_main[station_idx]
        Time, Level, Julian_day, Exact_wavelengths, Lwn_fonQ = \
                open_insitu(station)
        # mean spectra per station
        Exact_wavelengths.mask = Exact_wavelengths==-999
        Exact_wavelengths_mean = Exact_wavelengths.mean(axis=0)
        Lwn_fonQ.mask = Lwn_fonQ==-999
        Lwn_fonQ_mean = Lwn_fonQ.mean(axis=0)
        
        plt.figure(figsize=(12,3.5))
        plt.plot(Exact_wavelengths_mean[~Lwn_fonQ_mean.mask],Lwn_fonQ_mean[~Lwn_fonQ_mean.mask],'k',linewidth=4)
        # plt.xlabel('Wavelength (nm)',fontsize=fs)
        plt.ylabel('$L^{PRS}_{WN}$',fontsize=fs)
        plt.xlim([400,1050])
        plt.ylim([0,3])
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.title(f"{station_n[station]} {station.replace('_',' ')}",x=0.5,y=0.8,fontsize=fs+6)
        
        ofname = os.path.join(path_out,'source',f'spectra_olci_{station}.pdf')
        plt.savefig(ofname)
        plt.show()
        # plt.show()
        # plt.close()
#%%
# if __name__ == '__main__':
#     main()   
