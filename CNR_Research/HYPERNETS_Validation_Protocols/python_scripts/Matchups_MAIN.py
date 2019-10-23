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

# to import apply_flags_OLCI.py
path_main = '/Users/javier.concha/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'
sys.path.insert(0,path_main)
import apply_flags_OLCI as OLCI_flags

create_list_flag = 0
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

    count_Venise = 0
    count_Gloria = 0
    count_Galata_Platform = 0
    count_Helsinki_Lighthouse = 0
    count_Gustav_Dalen_Tower = 0

    plt.figure()
    #plt.errorbar(x, y, xerr=e_x, yerr=e_y, fmt='or')
    for cnt, line in enumerate(y):
        if station_vec[cnt] == 'Venise':
            mrk_style = 'or'
            count_Venise = count_Venise+1
        elif station_vec[cnt] == 'Gloria':
            mrk_style = 'og'
            count_Gloria = count_Gloria+1
        elif station_vec[cnt] == 'Galata_Platform':
            mrk_style = 'ob'
            count_Galata_Platform = count_Galata_Platform+1
        elif station_vec[cnt] == 'Helsinki_Lighthouse':
            mrk_style = 'om'
            count_Helsinki_Lighthouse = count_Helsinki_Lighthouse+1
        elif station_vec[cnt] == 'Gustav_Dalen_Tower':
            mrk_style = 'oc'
            count_Gustav_Dalen_Tower = count_Gustav_Dalen_Tower+1
        plt.plot(x[cnt], y[cnt],mrk_style)
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
    rmse_val = rmse(y,x)
    ref_obs = np.asarray(x)
    sat_obs = np.asarray(y)
    
        # the mean of relative (signed) percent differences
    rel_diff = 100*(ref_obs-sat_obs)/ref_obs
    mean_rel_diff = np.mean(rel_diff)
        
        #  the mean of absolute (unsigned) percent differences
    mean_abs_rel_diff = np.mean(np.abs(rel_diff))
    
    str2 = str1
    # to print without .0
    if str1[-2:]=='.0':
        str2 = str2[:-2]
        
    
    str0 = '{}nm\nN={:d}\nrmse={:,.2f}\nMAPD={:,.0f}%\nMPD={:,.0f}%\n$r^2$={:,.2f}'\
    .format(str2,\
            N,\
            rmse_val,\
            mean_abs_rel_diff,\
            mean_rel_diff,\
            r_value**2)
        
    plt.text(0.05, 0.65, str0,horizontalalignment='left', fontsize=12,transform=plt.gca().transAxes)
    
    ofname = sensor_name+'_scatter_matchups_'+str1.replace(".","p")+'_'+prot_name+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    
    plt.savefig(ofname, dpi=300)

    # latex table
    if str1 == '412.5':
        print('proto & nm & N & rmse & MAPD & MPD & $r^2$\n')
    str_table = '{} & {} & {:d} & {:,.2f} & {:,.0f} & {:,.0f} & {:,.2f}\\\\'\
    .format(prot_name,\
            str2,\
            N,\
            rmse_val,\
            mean_abs_rel_diff,\
            mean_rel_diff,\
            r_value**2)

    print(str_table)
 
    print('count_Venise: '+str(count_Venise))
    print('count_Gloria: '+str(count_Gloria))
    print('count_Galata_Platform: '+str(count_Galata_Platform))
    print('count_Helsinki_Lighthouse: '+str(count_Helsinki_Lighthouse))
    print('count_Gustav_Dalen_Tower: '+str(count_Gustav_Dalen_Tower))

    # plt.show()   
    return rmse_val, mean_abs_rel_diff, mean_rel_diff, r_value**2
#%%
def plot_both_methods(vl_str,notation_flag,path_out,min_val,max_val):
    print('=====================================')
    print(vl_str)
    if vl_str == '412.5':
        str0 = '0412p50'
        str3 = '412.5'
    elif vl_str == '442.5':
        str0 = '0442p50'
        str3 = '442.5'
    elif vl_str == '490.0':
        str0 = '0490p00'
        str3 = '490'
    elif vl_str == '560.0':
        str0 = '0560p00'
        str3 = '560'
    elif vl_str == '665.0':
        str0 = '0665p00'
        str3 = '665'
    
    ins_zi_station = globals()['matchups_Lwn_'+str0+'_fq_ins_zi_station']
    ins_ba_station = globals()['matchups_Lwn_'+str0+'_fq_ins_ba_station']

    sat_zi_stop_time = globals()['matchups_Lwn_'+str0+'_fq_sat_zi_stop_time']
    sat_ba_stop_time = globals()['matchups_Lwn_'+str0+'_fq_sat_ba_stop_time']

    sat_zi = globals()['matchups_Lwn_'+str0+'_fq_sat_zi']
    sat_ba = globals()['matchups_Lwn_'+str0+'_fq_sat_ba']

    ins_zi = globals()['matchups_Lwn_'+str0+'_fq_ins_zi']
    ins_ba = globals()['matchups_Lwn_'+str0+'_fq_ins_ba']

    count_both = 0
    count_zi = len(ins_zi_station)
    count_ba = len(ins_ba_station)

    diff = []

    # time series with two methods
    plt.figure(figsize=(16,4))
    for cnt, line in enumerate(ins_zi_station):
            if ins_zi_station[cnt] == 'Venise':
                mrk_style = 'xr'
            elif ins_zi_station[cnt] == 'Gloria':
                mrk_style = 'xg'
            elif ins_zi_station[cnt] == 'Galata_Platform':
                mrk_style = 'xb'
            elif ins_zi_station[cnt] == 'Helsinki_Lighthouse':
                mrk_style = 'xm'
            elif ins_zi_station[cnt] == 'Gustav_Dalen_Tower':
                mrk_style = 'xc'
                
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
                
                diff.append(sat_zi[cnt]-sat_ba[idx[0][0]])

                percent_change = 100*np.abs(sat_zi[cnt]-sat_ba[idx[0][0]])/max([sat_zi[cnt],sat_ba[idx[0][0]]])
                str2 = '{:,.2f}%'.format(percent_change)
                # print(str2)
                if notation_flag:
                    plt.text(sat_zi_stop_time[cnt], sat_zi[cnt],str2)
                
            else:
                # print(idx[0])
                # print(sat_zi_stop_time[cnt])
                plt.plot(sat_zi_stop_time[cnt], sat_zi[cnt],mrk_style)
                
    for cnt, line in enumerate(ins_ba_station):
            if ins_ba_station[cnt] == 'Venise':
                mrk_style = '+r'
            elif ins_ba_station[cnt] == 'Gloria':
                mrk_style = '+g'
            elif ins_ba_station[cnt] == 'Galata_Platform':
                mrk_style = '+b'
            elif ins_ba_station[cnt] == 'Helsinki_Lighthouse':
                mrk_style = '+m'
            elif ins_ba_station[cnt] == 'Gustav_Dalen_Tower':
                mrk_style = '+c'
            plt.plot(sat_ba_stop_time[cnt], sat_ba[cnt],mrk_style)
    plt.xlabel('Time',fontsize=12)
    sensor_name = 'OLCI'
    plt.ylabel('$L^{'+sensor_name+'}_{WN}$',fontsize=12)
    # zero line
    xmin, xmax = plt.gca().get_xlim()
    plt.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)  
    plt.text(0.05, 0.95, str3+'nm',horizontalalignment='left',fontsize=12,transform=plt.gca().transAxes)

    # save fig
    ofname = sensor_name+'_timeseries_diff_ba_zi_'+vl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    # histograms of both dataset: zi and ba
    kwargs2 = dict(bins='auto', histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w')
    ax1.hist(sat_zi,color='red', **kwargs2)
    ax1.hist(sat_ba,color='black', **kwargs2)
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim([x0,x0+1*(x1-x0)])

    ax1.set_ylabel('Frequency',fontsize=12)

    str1 = 'ZMB18\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.2f}\nmean: {:,.2f}\nN: {:,.0f}'\
    .format(np.nanmin(sat_zi),
            np.nanmax(sat_zi),
            np.nanstd(sat_zi),
            np.nanmedian(sat_zi),
            np.nanmean(sat_zi),
            len(sat_zi))

    str2 = 'BW06\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.2f}\nmean: {:,.2f}\nN: {:,.0f}'\
    .format(np.nanmin(sat_ba),
            np.nanmax(sat_ba),
            np.nanstd(sat_ba),
            np.nanmedian(sat_ba),
            np.nanmean(sat_ba),
            len(sat_ba))

    bottom, top = ax1.get_ylim()
    left, right = ax1.get_xlim()
    xpos = left+0.65*(right-left)
    ax1.text(left+0.01*(right-left),bottom+0.95*(top-bottom), '{}nm'.format(str3), fontsize=12,color='black')
    ax1.text(xpos,bottom+0.55*(top-bottom), str1, fontsize=12,color='red')
    ax1.text(xpos,bottom+0.15*(top-bottom), str2, fontsize=12,color='black')

    fig.text(0.5,0.01,'$L^{'+sensor_name+'}_{WN}$',ha='center',fontsize=12)

    # save fig
    ofname = sensor_name+'_hist_ba_zi_'+vl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    #latex table
    if vl_str == '412.5':
        print('proto & nm & min & max & std & median & mean & N\\\\')
    str_table = 'ZMB18 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(vl_str,\
            np.nanmin(sat_zi),
            np.nanmax(sat_zi),
            np.nanstd(sat_zi),
            np.nanmedian(sat_zi),
            np.nanmean(sat_zi),
            len(sat_zi))
    print(str_table)
    str_table = 'BW06 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(vl_str,\
            np.nanmin(sat_ba),
            np.nanmax(sat_ba),
            np.nanstd(sat_ba),
            np.nanmedian(sat_ba),
            np.nanmean(sat_ba),
            len(sat_ba))
    print(str_table)
    
    # histogram of the difference
    kwargs2 = dict(bins='auto', histtype='step')
    fig, (ax1,ax2)=plt.subplots(1,2,sharey=True, facecolor='w')
    ax1.hist(diff, **kwargs2)
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim([x0,x0+0.15*(x1-x0)])

    ax1.set_ylabel('Frequency',fontsize=12)

    str1 = '{}nm\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.4f}\nmean: {:,.4f}\nN: {:,.0f}'\
    .format(str3,
            np.nanmin(diff),
            np.nanmax(diff),
            np.nanstd(diff),
            np.nanmedian(diff),
            np.nanmean(diff),
            len(diff))

    if vl_str == '412.5':
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
    ax1.text(xpos,bottom+0.60*(top-bottom), str1, fontsize=12)

    fig.text(0.5,0.01,'Diff. $L^{'+sensor_name+'}_{WN}$',ha='center',fontsize=12)

    ax2.hist(diff, **kwargs2)
    x0, x1 = ax2.get_xlim()
    ax2.set_xlim([x0+0.70*(x1-x0),x1])

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright='on')

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1-d,1+d), (1-d, 1+d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, d), (- d, + d), **kwargs)  # bottom-left diagonal
    ax2.plot((- d,  d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # save fig
    ofname = sensor_name+'_hist_diff_ba_zi_'+vl_str.replace(".","p")+'.pdf'
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
                mrk_style = 'xr'
            elif ins_zi_station[cnt] == 'Gloria':
                mrk_style = 'xg'
            elif ins_zi_station[cnt] == 'Galata_Platform':
                mrk_style = 'xb'
            elif ins_zi_station[cnt] == 'Helsinki_Lighthouse':
                mrk_style = 'xm'
            elif ins_zi_station[cnt] == 'Gustav_Dalen_Tower':
                mrk_style = 'xc'
                
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
                mrk_style = '+r'
            elif ins_ba_station[cnt] == 'Gloria':
                mrk_style = '+g'
            elif ins_ba_station[cnt] == 'Galata_Platform':
                mrk_style = '+b'
            elif ins_ba_station[cnt] == 'Helsinki_Lighthouse':
                mrk_style = '+m'
            elif ins_ba_station[cnt] == 'Gustav_Dalen_Tower':
                mrk_style = '+c'
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
    ofname = sensor_name+'_scatter_diff_ba_zi_'+vl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()    
#%%
#def main():
#    """business logic for when running this module as the primary one!"""
print('Main Code!')
path_out = os.path.join(path_main,'Figures')
path = os.path.join(path_main,'netcdf_file')
filename = 'Venise_20_201601001_201812031.nc'
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

day_vec =np.array([float(Time[i].replace(' ',':').split(':')[0]) for i in range(0,len(Time))])
month_vec =np.array([float(Time[i].replace(' ',':').split(':')[1]) for i in range(0,len(Time))])
year_vec =np.array([float(Time[i].replace(' ',':').split(':')[2]) for i in range(0,len(Time))])
hour_vec =np.array([float(Time[i].replace(' ',':').split(':')[3]) for i in range(0,len(Time))])
minute_vec =np.array([float(Time[i].replace(' ',':').split(':')[4]) for i in range(0,len(Time))])
second_vec =np.array([float(Time[i].replace(' ',':').split(':')[5]) for i in range(0,len(Time))])

Julian_day_vec =np.array([float(Julian_day[i]) for i in range(0,len(Time))])
date_format = "%d:%m:%Y %H:%M:%S"
ins_time = np.array([datetime.strptime(Time[i], date_format) for i in range(0,len(Time))])

doy_vec = np.array([int(float(Julian_day[i])) for i in range(0,len(Time))])

if create_list_flag:
    create_OLCI_list(path_main,Time,year_vec,month_vec,doy_vec,day_vec)
   
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
matchups_Lwn_0412p50_fq_ins_zi = []
matchups_Lwn_0442p50_fq_ins_zi = []
matchups_Lwn_0490p00_fq_ins_zi = []
matchups_Lwn_0560p00_fq_ins_zi = []
matchups_Lwn_0665p00_fq_ins_zi = []

matchups_Lwn_0412p50_fq_sat_zi = []
matchups_Lwn_0442p50_fq_sat_zi = []
matchups_Lwn_0490p00_fq_sat_zi = []
matchups_Lwn_0560p00_fq_sat_zi = []
matchups_Lwn_0665p00_fq_sat_zi = []

matchups_Lwn_0412p50_fq_ins_zi_station = []
matchups_Lwn_0442p50_fq_ins_zi_station = []
matchups_Lwn_0490p00_fq_ins_zi_station = []
matchups_Lwn_0560p00_fq_ins_zi_station = []
matchups_Lwn_0665p00_fq_ins_zi_station = []

matchups_Lwn_0412p50_fq_sat_zi_stop_time = []
matchups_Lwn_0442p50_fq_sat_zi_stop_time = []
matchups_Lwn_0490p00_fq_sat_zi_stop_time = []
matchups_Lwn_0560p00_fq_sat_zi_stop_time = []
matchups_Lwn_0665p00_fq_sat_zi_stop_time = []    

# Bailey and Werdell: initialization
matchups_Lwn_0412p50_fq_ins_ba = []
matchups_Lwn_0442p50_fq_ins_ba = []
matchups_Lwn_0490p00_fq_ins_ba = []
matchups_Lwn_0560p00_fq_ins_ba = []
matchups_Lwn_0665p00_fq_ins_ba = []

matchups_Lwn_0412p50_fq_sat_ba = []
matchups_Lwn_0442p50_fq_sat_ba = []
matchups_Lwn_0490p00_fq_sat_ba = []
matchups_Lwn_0560p00_fq_sat_ba = []
matchups_Lwn_0665p00_fq_sat_ba = []

matchups_Lwn_0412p50_fq_ins_ba_station = []
matchups_Lwn_0442p50_fq_ins_ba_station = []
matchups_Lwn_0490p00_fq_ins_ba_station = []
matchups_Lwn_0560p00_fq_ins_ba_station = []
matchups_Lwn_0665p00_fq_ins_ba_station = [] 

matchups_Lwn_0412p50_fq_sat_ba_stop_time = []
matchups_Lwn_0442p50_fq_sat_ba_stop_time = []
matchups_Lwn_0490p00_fq_sat_ba_stop_time = []
matchups_Lwn_0560p00_fq_sat_ba_stop_time = []
matchups_Lwn_0665p00_fq_sat_ba_stop_time = []         

# station_list = ['Venise','Galata_Platform','Gloria']
station_list = ['Venise','Galata_Platform','Gloria','Helsinki_Lighthouse','Gustav_Dalen_Tower']
# station_list = ['Venise']

for station_name in station_list:  
    path_to_list = os.path.join(path_main,'data','output','extract_list_'+station_name+'.txt')
    # create list of extract by station
    cmd = 'find '+os.path.join(path_main,'data','output')+' -name "extract_'+station_name+'.nc"|sort|uniq >'+path_to_list
    (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
    with open(path_to_list,'r') as file:
        for cnt, line in enumerate(file):  
            print('----------------------------')
            print('line '+str(cnt))
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
            
            # Zibordi et al. 2018
            print('--Zibordi et al. 2018')
            delta_time = 2# float in hours       
            time_diff = ins_time - sat_stop_time
            dt_hour = [i.total_seconds()/(60*60) for i in time_diff] # time diffence between in situ measurements and sat in hours
            idx_min = np.argmin(np.abs(dt_hour))
            matchup_idx_vec = np.abs(dt_hour) <= delta_time 
    
            nday = sum(matchup_idx_vec)
            if nday >=1:
                print(str(nday)+' matchups per '+year_str+' '+doy_str)
    #            print(Lwn_fonQ[idx_min,:])
    #            print(Exact_wavelengths[idx_min,:])
                
                center_px = int(len(rhow_0412p50_fq)/2 + 0.5)
                size_box = 3
                start_idx_x = int(center_px-int(size_box/2))
                stop_idx_x = int(center_px+int(size_box/2)+1)
                start_idx_y = int(center_px-int(size_box/2))
                stop_idx_y = int(center_px+int(size_box/2)+1)
                rhow_0412p50_fq_box = rhow_0412p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0442p50_fq_box = rhow_0442p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0490p00_fq_box = rhow_0490p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0560p00_fq_box = rhow_0560p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0665p00_fq_box = rhow_0665p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
    
                AOT_0865p50_box = AOT_0865p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                
                flags_mask = OLCI_flags.create_mask(WQSF[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
                print(flags_mask)
                
                if sza<=70 and vza<=56 and not flags_mask.any(): # if any of the pixels if flagged, Fails validation criteria because all have to be valid in Zibordi 2018
                    Lwn_560 = rhow_0560p00_fq_box*F0_0560p00/np.pi
                    Lwn_560_CV = Lwn_560.std()/Lwn_560.mean()    
                    
                    AOT_0865p50_CV = AOT_0865p50_box.std()/AOT_0865p50_box.mean()
                    
                    if Lwn_560_CV <= 0.2 and AOT_0865p50_CV <= 0.2:
                        # Rrs 0412p50
                        print('412.5')
                        if rhow_0412p50_fq_box.mask.any() == True or np.isnan(rhow_0412p50_fq_box).any() == True:
                            print('At least one element in sat product is invalid!')
                        else:
                            matchups_Lwn_0412p50_fq_sat_zi.append(rhow_0412p50_fq_box.mean()*F0_0412p50/np.pi)
                            matchups_Lwn_0412p50_fq_ins_zi.append(Lwn_fonQ[idx_min,0]) # 412,
                            matchups_Lwn_0412p50_fq_ins_zi_station.append(station_name)
                            matchups_Lwn_0412p50_fq_sat_zi_stop_time.append(sat_stop_time)
                            
                        # Rrs 0442p50
                        print('442.5')
                        if rhow_0442p50_fq_box.mask.any() == True or np.isnan(rhow_0442p50_fq_box).any() == True:
                            print('At least one element in sat product is invalid!')
                        else:
                            matchups_Lwn_0442p50_fq_sat_zi.append(rhow_0442p50_fq_box.mean()*F0_0442p50/np.pi)
                            matchups_Lwn_0442p50_fq_ins_zi.append(Lwn_fonQ[idx_min,1]) # 441.8
                            matchups_Lwn_0442p50_fq_ins_zi_station.append(station_name)
                            matchups_Lwn_0442p50_fq_sat_zi_stop_time.append(sat_stop_time)
                            
                        # Rrs 0490p00
                        print('490.0')
                        if rhow_0490p00_fq_box.mask.any() == True or np.isnan(rhow_0490p00_fq_box).any() == True:
                            print('At least one element in sat product is invalid!')
                        else:
                            matchups_Lwn_0490p00_fq_sat_zi.append(rhow_0490p00_fq_box.mean()*F0_0490p00/np.pi)
                            matchups_Lwn_0490p00_fq_ins_zi.append(Lwn_fonQ[idx_min,2]) # 488.5
                            matchups_Lwn_0490p00_fq_ins_zi_station.append(station_name)
                            matchups_Lwn_0490p00_fq_sat_zi_stop_time.append(sat_stop_time)
                            
                        # Rrs 0560p00
                        print('560.0')
                        if rhow_0560p00_fq_box.mask.any() == True or np.isnan(rhow_0560p00_fq_box).any() == True:
                            print('At least one element in sat product is invalid!')
                        else:
                            matchups_Lwn_0560p00_fq_sat_zi.append(rhow_0560p00_fq_box.mean()*F0_0560p00/np.pi)
                            matchups_Lwn_0560p00_fq_ins_zi.append(Lwn_fonQ[idx_min,4]) # 551,
                            matchups_Lwn_0560p00_fq_ins_zi_station.append(station_name)
                            matchups_Lwn_0560p00_fq_sat_zi_stop_time.append(sat_stop_time)
                            
                        # Rrs 0665p00
                        print('665.0')
                        if rhow_0665p00_fq_box.mask.any() == True or np.isnan(rhow_0665p00_fq_box).any() == True:
                            print('At least one element in sat product is invalid!')
                        else:
                            matchups_Lwn_0665p00_fq_sat_zi.append(rhow_0665p00_fq_box.mean()*F0_0665p00/np.pi)
                            matchups_Lwn_0665p00_fq_ins_zi.append(Lwn_fonQ[idx_min,5]) # 667.9    
                            matchups_Lwn_0665p00_fq_ins_zi_station.append(station_name)
                            matchups_Lwn_0665p00_fq_sat_zi_stop_time.append(sat_stop_time)
                            
                    else:
                        print('CV exceeds criteria: CV[Lwn(560)]='+str(Lwn_560_CV)+'; CV[AOT(865.5)]='+str(AOT_0865p50_CV))
                else:
                    print('Angles exceeds criteria: sza='+str(sza)+'; vza='+str(vza)+'; OR some pixels are flagged!')
            else:
                print('Not matchups per '+year_str+' '+doy_str)
    
             # Bailey and Werdell 2006 
            print('--Bailey and Werdell 2006')
            delta_time = 3# float in hours       
            time_diff = ins_time - sat_stop_time
            dt_hour = [i.total_seconds()/(60*60) for i in time_diff] # time diffence between in situ measurements and sat in hours
            idx_min = np.argmin(np.abs(dt_hour))
            matchup_idx_vec = np.abs(dt_hour) <= delta_time 
    
            nday = sum(matchup_idx_vec)
            if nday >=1:
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
                rhow_0412p50_fq_box = rhow_0412p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0442p50_fq_box = rhow_0442p50_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0490p00_fq_box = rhow_0490p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0560p00_fq_box = rhow_0560p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                rhow_0665p00_fq_box = rhow_0665p00_fq[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
    
                AOT_0865p50_box = AOT_0865p50[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
                
                print(rhow_0412p50_fq_box)
                print(rhow_0412p50_fq_box.mask)
                
                flags_mask = OLCI_flags.create_mask(WQSF[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y])
                print(flags_mask)
                
                NGP = np.count_nonzero(flags_mask == 0) # Number Good Pixels, Bailey and Werdell 2006
                
                if sza<=75 and vza<=60 and NGP>NTP/2+1:
    
                    # if nan, change mask
                    rhow_0412p50_fq_box = ma.masked_invalid(rhow_0412p50_fq_box)
                    rhow_0442p50_fq_box = ma.masked_invalid(rhow_0442p50_fq_box)
                    rhow_0490p00_fq_box = ma.masked_invalid(rhow_0490p00_fq_box)
                    rhow_0560p00_fq_box = ma.masked_invalid(rhow_0560p00_fq_box)
                    rhow_0665p00_fq_box = ma.masked_invalid(rhow_0665p00_fq_box)
                    AOT_0865p50_box = ma.masked_invalid(AOT_0865p50_box)
    
                    NGP_rhow_0412p50 = np.count_nonzero(rhow_0412p50_fq_box.mask == 0)
                    NGP_rhow_0442p50 = np.count_nonzero(rhow_0442p50_fq_box.mask == 0)
                    NGP_rhow_0490p00 = np.count_nonzero(rhow_0490p00_fq_box.mask == 0)
                    NGP_rhow_0560p00 = np.count_nonzero(rhow_0560p00_fq_box.mask == 0)
                    NGP_rhow_0665p00 = np.count_nonzero(rhow_0665p00_fq_box.mask == 0)
                    NGP_AOT_0865p50 = np.count_nonzero(AOT_0865p50_box.mask == 0)
    
                    mean_unfiltered_rhow_0412p50 = rhow_0412p50_fq_box.mean()
                    mean_unfiltered_rhow_0442p50 = rhow_0442p50_fq_box.mean()
                    mean_unfiltered_rhow_0490p00 = rhow_0490p00_fq_box.mean()
                    mean_unfiltered_rhow_0560p00 = rhow_0560p00_fq_box.mean()
                    mean_unfiltered_rhow_0665p00 = rhow_0665p00_fq_box.mean()
                    mean_unfiltered_AOT_0865p50 = AOT_0865p50_box.mean()
    
                    std_unfiltered_rhow_0412p50 = rhow_0412p50_fq_box.std()
                    std_unfiltered_rhow_0442p50 = rhow_0442p50_fq_box.std()
                    std_unfiltered_rhow_0490p00 = rhow_0490p00_fq_box.std()
                    std_unfiltered_rhow_0560p00 = rhow_0560p00_fq_box.std()
                    std_unfiltered_rhow_0665p00 = rhow_0665p00_fq_box.std()
                    std_unfiltered_AOT_0865p50 = AOT_0865p50_box.std()
    
                    # mask values that are not within +/- 1.5*std of mean\
                    
                    rhow_0412p50_fq_box = ma.masked_outside(rhow_0412p50_fq_box,mean_unfiltered_rhow_0412p50\
                        -1.5*std_unfiltered_rhow_0412p50\
                        , mean_unfiltered_rhow_0412p50\
                        +1.5*std_unfiltered_rhow_0412p50)
                    rhow_0442p50_fq_box = ma.masked_outside(rhow_0442p50_fq_box,mean_unfiltered_rhow_0442p50\
                        -1.5*std_unfiltered_rhow_0442p50\
                        , mean_unfiltered_rhow_0442p50\
                        +1.5*std_unfiltered_rhow_0442p50)
                    rhow_0490p00_fq_box = ma.masked_outside(rhow_0490p00_fq_box,mean_unfiltered_rhow_0490p00\
                        -1.5*std_unfiltered_rhow_0490p00\
                        , mean_unfiltered_rhow_0490p00\
                        +1.5*std_unfiltered_rhow_0490p00)
                    rhow_0560p00_fq_box = ma.masked_outside(rhow_0560p00_fq_box,mean_unfiltered_rhow_0560p00\
                        -1.5*std_unfiltered_rhow_0560p00\
                        , mean_unfiltered_rhow_0560p00\
                        +1.5*std_unfiltered_rhow_0560p00)
                    rhow_0665p00_fq_box = ma.masked_outside(rhow_0665p00_fq_box,mean_unfiltered_rhow_0665p00\
                        -1.5*std_unfiltered_rhow_0665p00\
                        , mean_unfiltered_rhow_0665p00\
                        +1.5*std_unfiltered_rhow_0665p00)
                    AOT_0865p50_box = ma.masked_outside(AOT_0865p50_box,mean_unfiltered_AOT_0865p50\
                        -1.5*std_unfiltered_AOT_0865p50\
                        , mean_unfiltered_AOT_0865p50\
                        +1.5*std_unfiltered_AOT_0865p50)
    
                    mean_filtered_rhow_0412p50 = rhow_0412p50_fq_box.mean()
                    mean_filtered_rhow_0442p50 = rhow_0442p50_fq_box.mean()
                    mean_filtered_rhow_0490p00 = rhow_0490p00_fq_box.mean()
                    mean_filtered_rhow_0560p00 = rhow_0560p00_fq_box.mean()
                    mean_filtered_rhow_0665p00 = rhow_0665p00_fq_box.mean()
                    mean_filtered_AOT_0865p50  = AOT_0865p50_box.mean()
    
                    std_filtered_rhow_0412p50 = rhow_0412p50_fq_box.std()
                    std_filtered_rhow_0442p50 = rhow_0442p50_fq_box.std()
                    std_filtered_rhow_0490p00 = rhow_0490p00_fq_box.std()
                    std_filtered_rhow_0560p00 = rhow_0560p00_fq_box.std()
                    std_filtered_rhow_0665p00 = rhow_0665p00_fq_box.std()
                    std_filtered_AOT_0865p50  = AOT_0865p50_box.std()
    
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
                    MedianCV = np.nanmedian(CVs)
    
                    print('Median CV='+str(MedianCV))
                   
                    if MedianCV <= 0.15:
                        # Rrs 0412p50
                        print('412.5')
                        if NGP_rhow_0412p50<NTP/2+1:
                            print('Exceeded: NGP_rhow_0412p50='+str(NGP_rhow_0412p50))
                        else:
                            matchups_Lwn_0412p50_fq_sat_ba.append(mean_filtered_rhow_0412p50*F0_0412p50/np.pi)
                            matchups_Lwn_0412p50_fq_ins_ba.append(Lwn_fonQ[idx_min,0]) # 412,
                            matchups_Lwn_0412p50_fq_ins_ba_station.append(station_name)
                            matchups_Lwn_0412p50_fq_sat_ba_stop_time.append(sat_stop_time)
                            
                        # Rrs 0442p50
                        print('442.5')
                        if NGP_rhow_0442p50<NTP/2+1:
                            print('Exceeded: NGP_rhow_0442p50='+str(NGP_rhow_0442p50))
                        else:
                            matchups_Lwn_0442p50_fq_sat_ba.append(mean_filtered_rhow_0442p50*F0_0442p50/np.pi)
                            matchups_Lwn_0442p50_fq_ins_ba.append(Lwn_fonQ[idx_min,1]) # 441.8
                            matchups_Lwn_0442p50_fq_ins_ba_station.append(station_name)
                            matchups_Lwn_0442p50_fq_sat_ba_stop_time.append(sat_stop_time)
                            
                        # Rrs 0490p00
                        print('490.0')
                        if NGP_rhow_0490p00<NTP/2+1:
                            print('Exceeded: NGP_rhow_0490p00='+str(NGP_rhow_0490p00))
                        else:
                            matchups_Lwn_0490p00_fq_sat_ba.append(mean_filtered_rhow_0490p00*F0_0490p00/np.pi)
                            matchups_Lwn_0490p00_fq_ins_ba.append(Lwn_fonQ[idx_min,2]) # 488.5
                            matchups_Lwn_0490p00_fq_ins_ba_station.append(station_name)
                            matchups_Lwn_0490p00_fq_sat_ba_stop_time.append(sat_stop_time)
                            
                        # Rrs 0560p00
                        print('560.0')
                        if NGP_rhow_0560p00<NTP/2+1:
                            print('Exceeded: NGP_rhow_0560p00='+str(NGP_rhow_0560p00))
                        else:
                            matchups_Lwn_0560p00_fq_sat_ba.append(mean_filtered_rhow_0560p00*F0_0560p00/np.pi)
                            matchups_Lwn_0560p00_fq_ins_ba.append(Lwn_fonQ[idx_min,4]) # 551,
                            matchups_Lwn_0560p00_fq_ins_ba_station.append(station_name)
                            matchups_Lwn_0560p00_fq_sat_ba_stop_time.append(sat_stop_time)
                            
                        # Rrs 0665p00
                        print('665.0')
                        if NGP_rhow_0665p00<NTP/2+1:
                            print('Exceeded: NGP_rhow_0665p00='+str(NGP_rhow_0665p00))
                        else:
                            matchups_Lwn_0665p00_fq_sat_ba.append(mean_filtered_rhow_0665p00*F0_0665p00/np.pi)
                            matchups_Lwn_0665p00_fq_ins_ba.append(Lwn_fonQ[idx_min,5]) # 667.9    
                            matchups_Lwn_0665p00_fq_ins_ba_station.append(station_name)
                            matchups_Lwn_0665p00_fq_sat_ba_stop_time.append(sat_stop_time)
                            
                    else:
                        print('Median CV exceeds criteria: Median[CV]='+str(MedianCV))
                else:
                    print('Angles exceeds criteria: sza='+str(sza)+'; vza='+str(vza)+'; OR NGP='+str(NGP)+'< NTP/2+1='+str(NTP/2+1)+'!')
            else:
                print('Not matchups per '+year_str+' '+doy_str)            
    
#%% plots   
prot_name = 'zi'
sensor_name = 'OLCI'
rmse_val_0412p50_zi, mean_abs_rel_diff_0412p50_zi, mean_rel_diff_0412p50_zi, r_sqr_0412p50_zi = plot_scatter(\
    matchups_Lwn_0412p50_fq_ins_zi,matchups_Lwn_0412p50_fq_sat_zi,'412.5',path_out,prot_name,sensor_name,\
    matchups_Lwn_0412p50_fq_ins_zi_station,min_val=-1.00,max_val=4.0) 
rmse_val_0442p50_zi, mean_abs_rel_diff_0442p50_zi, mean_rel_diff_0442p50_zi, r_sqr_0442p50_zi = plot_scatter(
    matchups_Lwn_0442p50_fq_ins_zi,matchups_Lwn_0442p50_fq_sat_zi,'442.5',path_out,prot_name,sensor_name,\
    matchups_Lwn_0442p50_fq_ins_zi_station,min_val=-1.00,max_val=6.0) 
rmse_val_0490p00_zi, mean_abs_rel_diff_0490p00_zi, mean_rel_diff_0490p00_zi, r_sqr_0490p00_zi = plot_scatter(
    matchups_Lwn_0490p00_fq_ins_zi,matchups_Lwn_0490p00_fq_sat_zi,'490.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0490p00_fq_ins_zi_station,min_val=-1.00,max_val=8.0) 
rmse_val_0560p00_zi, mean_abs_rel_diff_0560p00_zi, mean_rel_diff_0560p00_zi, r_sqr_0560p00_zi = plot_scatter(
    matchups_Lwn_0560p00_fq_ins_zi,matchups_Lwn_0560p00_fq_sat_zi,'560.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0560p00_fq_ins_zi_station,min_val=-1.00,max_val=6.0) 
rmse_val_0665p00_zi, mean_abs_rel_diff_0665p00_zi, mean_rel_diff_0665p00_zi, r_sqr_0665p00_zi = plot_scatter(
    matchups_Lwn_0665p00_fq_ins_zi,matchups_Lwn_0665p00_fq_sat_zi,'665.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0665p00_fq_ins_zi_station,min_val=-1.00,max_val=2.0) 

#% plots  
prot_name = 'ba' 
sensor_name = 'OLCI'
rmse_val_0412p50_ba, mean_abs_rel_diff_0412p50_ba, mean_rel_diff_0412p50_ba, r_sqr_0412p50_ba = plot_scatter(
    matchups_Lwn_0412p50_fq_ins_ba,matchups_Lwn_0412p50_fq_sat_ba,'412.5',path_out,prot_name,sensor_name,\
    matchups_Lwn_0412p50_fq_ins_ba_station,min_val=-1.00,max_val=4.0) 
rmse_val_0442p50_ba, mean_abs_rel_diff_0442p50_ba, mean_rel_diff_0442p50_ba, r_sqr_0442p50_ba = plot_scatter(
    matchups_Lwn_0442p50_fq_ins_ba,matchups_Lwn_0442p50_fq_sat_ba,'442.5',path_out,prot_name,sensor_name,\
    matchups_Lwn_0442p50_fq_ins_ba_station,min_val=-1.00,max_val=6.0) 
rmse_val_0490p00_ba, mean_abs_rel_diff_0490p00_ba, mean_rel_diff_0490p00_ba, r_sqr_0490p00_ba = plot_scatter(
    matchups_Lwn_0490p00_fq_ins_ba,matchups_Lwn_0490p00_fq_sat_ba,'490.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0490p00_fq_ins_ba_station,min_val=-1.00,max_val=8.0) 
rmse_val_0560p00_ba, mean_abs_rel_diff_0560p00_ba, mean_rel_diff_0560p00_ba, r_sqr_0560p00_ba = plot_scatter(
    matchups_Lwn_0560p00_fq_ins_ba,matchups_Lwn_0560p00_fq_sat_ba,'560.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0560p00_fq_ins_ba_station,min_val=-1.00,max_val=6.0) 
rmse_val_0665p00_ba, mean_abs_rel_diff_0665p00_ba, mean_rel_diff_0665p00_ba, r_sqr_0665p00_ba = plot_scatter(
    matchups_Lwn_0665p00_fq_ins_ba,matchups_Lwn_0665p00_fq_sat_ba,'665.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0665p00_fq_ins_ba_station,min_val=-1.00,max_val=2.0) 

#%%
# rmse
rmse_zi = [rmse_val_0412p50_zi,rmse_val_0442p50_zi,rmse_val_0490p00_zi,rmse_val_0560p00_zi,rmse_val_0665p00_zi] 
rmse_ba = [rmse_val_0412p50_ba,rmse_val_0442p50_ba,rmse_val_0490p00_ba,rmse_val_0560p00_ba,rmse_val_0665p00_ba]
wv = [412.5,442.5,490.0,560.0,665.0]
plt.figure()
plt.plot(wv,rmse_zi,'-o')
plt.plot(wv,rmse_ba,'-o')
plt.xlabel('Wavelength [nm]',fontsize=12)
plt.ylabel('$rmse$',fontsize=12)
# plt.legend(['Zibordi, Mèlin and Berthon (2018)','Bailey and Werdell (2006)'])
plt.show()

ofname = 'OLCI_rmse.pdf'
ofname = os.path.join(path_out,'source',ofname)   
plt.savefig(ofname, dpi=300)

# mean_abs_rel_diff
mean_abs_rel_diff_zi = [mean_abs_rel_diff_0412p50_zi,mean_abs_rel_diff_0442p50_zi,mean_abs_rel_diff_0490p00_zi,\
    mean_abs_rel_diff_0560p00_zi,mean_abs_rel_diff_0665p00_zi]
mean_abs_rel_diff_ba = [mean_abs_rel_diff_0412p50_ba,mean_abs_rel_diff_0442p50_ba,mean_abs_rel_diff_0490p00_ba,\
    mean_abs_rel_diff_0560p00_ba,mean_abs_rel_diff_0665p00_ba]
wv = [412.5,442.5,490.0,560.0,665.0]
plt.figure()
plt.plot(wv,mean_abs_rel_diff_zi,'-o')
plt.plot(wv,mean_abs_rel_diff_ba,'-o')
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
wv = [412.5,442.5,490.0,560.0,665.0]
plt.figure()
plt.plot(wv,mean_rel_diff_zi,'-o')
plt.plot(wv,mean_rel_diff_ba,'-o')
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
wv = [412.5,442.5,490.0,560.0,665.0]
plt.figure()
plt.plot(wv,r_sqr_zi,'-o')
plt.plot(wv,r_sqr_ba,'-o')
plt.xlabel('Wavelength [nm]',fontsize=12)
plt.ylabel('$r^2$',fontsize=12)
plt.legend(['ZMB18','BW06'],fontsize=12)
plt.show()    

ofname = 'OLCI_r_sqr.pdf'
ofname = os.path.join(path_out,'source',ofname)   
plt.savefig(ofname, dpi=300)   

#%% 
notation_flag = 0 # to display percentage difference in the plot
plot_both_methods('412.5',notation_flag,path_out,min_val=-2.00,max_val=4.0)
plot_both_methods('442.5',notation_flag,path_out,min_val=-2.00,max_val=6.0)
plot_both_methods('490.0',notation_flag,path_out,min_val=-1.00,max_val=8.0)
plot_both_methods('560.0',notation_flag,path_out,min_val=-1.00,max_val=6.0)
plot_both_methods('665.0',notation_flag,path_out,min_val=-0.60,max_val=2.0) 
#       
#%
# if __name__ == '__main__':
#     main()   
