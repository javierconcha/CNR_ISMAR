#!/usr/bin/env python3
# coding: utf-8
"""
Created on Mon Jul  8 16:41:12 2019

Script to find matchups for high res satellites. 
Example of aeronet_daily.log at the end of the script.

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
'''
Location=Galata_Platform,Latitude=43.044624,Longitude=28.193190,Elevation[m]=31.000000
Location=Gustav_Dalen_Tower,Latitude=58.594170,Longitude=17.466830,Elevation[m]=25.000000
Location=Helsinki_Lighthouse,Latitude=59.948970,Longitude=24.926360,Elevation[m]=20.000000
Location=Lake_Erie,Latitude=41.825600,Longitude=-83.193600,Elevation[m]=173.500000
Location=LISCO,Latitude=40.954517,Longitude=-73.341767,Elevation[m]=12.000000
Location=Palgrunden,Latitude=58.755333,Longitude=13.151500,Elevation[m]=49.000000
Location=Thornton_C-power,Latitude=51.532500,Longitude=2.955278,Elevation[m]=30.000000
Location=USC_SEAPRISM,Latitude=33.563710,Longitude=-118.117820,Elevation[m]=31.000000
Location=USC_SEAPRISM_2,Latitude=33.563710,Longitude=-118.117820,Elevation[m]=31.000000
Location=Venise,Latitude=45.313900,Longitude=12.508300,Elevation[m]=10.000000
Location=WaveCIS_Site_CSI_6,Latitude=28.866667,Longitude=-90.483333,Elevation[m]=32.660000
Location=Gloria,Latitude=44.599970,Longitude=29.359670,Elevation[m]=30.000000
'''
import sys
import os
import glob
from datetime import datetime
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
from scipy import stats
import common_functions

sys.path.insert(0,'/Users/javier.concha/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts')

def get_lat_lon_ins(station_name):
    if station_name == 'Galata_Platform': # Black Sea
        Latitude=43.044624
        Longitude=28.193190
    elif station_name == 'Gustav_Dalen_Tower': # Baltic Sea
        Latitude=58.594170
        Longitude=17.466830
    elif station_name == 'Helsinki_Lighthouse': # Baltic Sea
        Latitude=59.948970
        Longitude=24.926360
    elif station_name == 'Lake_Erie':
        Latitude=41.825600
        Longitude=-83.193600
    elif station_name == 'LISCO':
        Latitude=40.954517
        Longitude=-73.341767
    elif station_name == 'Palgrunden':
        Latitude=58.755333
        Longitude=13.151500
    elif station_name == 'Thornton_C-power':
        Latitude=51.532500
        Longitude=2.955278
    elif station_name == 'USC_SEAPRISM':
        Latitude=33.563710
        Longitude=-118.117820
    elif station_name == 'USC_SEAPRISM_2':
        Latitude=33.563710
        Longitude=-118.117820
    elif station_name == 'Venise': # Adriatic Sea
        Latitude=45.313900
        Longitude=12.508300
    elif station_name == 'WaveCIS_Site_CSI_6':
        Latitude=28.866667
        Longitude=-90.483333
    elif station_name == 'Gloria': # Black Sea
        Latitude=44.599970
        Longitude=29.359670
    else:
        print('ERROR: station not found: '+station_name)
    return Latitude, Longitude

#%%
def plot_scatter(x,y,str1,path_out,prot_name,sensor_name,station_vec,min_val,max_val):           

    # replace nan in y (sat data)
    x = np.array(x)
    y = np.array(y)
    station_vec = np.array(station_vec)

    x = x[~np.isnan(y)] # it is assumed that only sat data could be nan
    station_vec = station_vec[~np.isnan(y)]
    y = y[~np.isnan(y)]


    count_Venise = 0
    count_Gloria = 0
    count_Galata_Platform = 0
    count_Helsinki_Lighthouse = 0
    count_Gustav_Dalen_Tower = 0
    count_Lake_Erie = 0
    count_LISCO = 0
    count_Palgrunden = 0
    count_Thornton_C_power = 0
    count_USC_SEAPRISM = 0
    count_USC_SEAPRISM_2 = 0
    count_WaveCIS_Site_CSI_6 = 0

    plt.figure()
    #plt.errorbar(x, y, xerr=e_x, yerr=e_y, fmt='or')
    for cnt, line in enumerate(y):
        if station_vec[cnt] == 'Venise': # Adriatic Sea
            mrk_style = 'o'
            mrk_color = 'r' 
            count_Venise = count_Venise+1
        elif station_vec[cnt] == 'Gloria': # Black Sea    
            mrk_style = 'o'
            mrk_color = 'g'       
            count_Gloria = count_Gloria+1
        elif station_vec[cnt] == 'Galata_Platform': # Black Sea
            mrk_style = 'o'
            mrk_color = 'b'
            count_Galata_Platform = count_Galata_Platform+1
        elif station_vec[cnt] == 'Helsinki_Lighthouse': # Baltic Sea
            mrk_style = 'o'
            mrk_color = 'm'            
            count_Helsinki_Lighthouse = count_Helsinki_Lighthouse+1
        elif station_vec[cnt] == 'Gustav_Dalen_Tower': # Baltic Sea
            mrk_style = 'o'
            mrk_color = 'c'
            count_Gustav_Dalen_Tower = count_Gustav_Dalen_Tower+1
        elif station_vec[cnt] == 'Lake_Erie':
            mrk_style = 'o'
            mrk_color = 'brown'
            count_Lake_Erie = count_Lake_Erie+1
        elif station_vec[cnt] == 'LISCO':
            mrk_style = 'o'
            mrk_color = 'lime'
            count_LISCO = count_LISCO+1
        elif station_vec[cnt] == 'Palgrunden':
            mrk_style = 'o'
            mrk_color = 'dodgerblue'
            count_Palgrunden = count_Palgrunden+1
        elif station_vec[cnt] == 'Thornton_C-power':
            mrk_style = 'o'
            mrk_color = 'crimson'
            count_Thornton_C_power = count_Thornton_C_power+1
        elif station_vec[cnt] == 'USC_SEAPRISM':
            mrk_style = 'o'
            mrk_color = 'chocolate'
            count_USC_SEAPRISM = count_USC_SEAPRISM+1
        elif station_vec[cnt] == 'USC_SEAPRISM_2':
            mrk_style = 'o'
            mrk_color = 'darkviolet'
            count_USC_SEAPRISM_2 = count_USC_SEAPRISM_2+1
        elif station_vec[cnt] == 'WaveCIS_Site_CSI_6':
            mrk_style = 'o'
            mrk_color = 'turquoise'  
            count_WaveCIS_Site_CSI_6 = count_WaveCIS_Site_CSI_6+1

        plt.plot(x[cnt], y[cnt],marker=mrk_style,color=mrk_color)
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
    rmse_val = common_functions.rmse(y,x)
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
    
    # plt.show()   

        # latex table
    if str1 == '444.0':
        print('proto & nm & N & rmse & MAPD & MPD & $r^2$')
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
    print('count_Lake_Erie: '+str(count_Lake_Erie))
    print('count_LISCO: '+str(count_LISCO))
    print('count_Palgrunden: '+str(count_Palgrunden))
    print('count_Thornton_C_power: '+str(count_Thornton_C_power))
    print('count_USC_SEAPRISM: '+str(count_USC_SEAPRISM))
    print('count_USC_SEAPRISM_2: '+str(count_USC_SEAPRISM_2))
    print('count_WaveCIS_Site_CSI_6: '+str(count_WaveCIS_Site_CSI_6))

    return rmse_val, mean_abs_rel_diff, mean_rel_diff, r_value**2
#%%
def plot_all_methods(vl_str,notation_flag,path_out,min_val,max_val):
    print('=====================================')
    print(vl_str)
    if vl_str == '444.0':
        str0 = '0444p00'
        str3 = '444'
    elif vl_str == '497.0':
        str0 = '0497p00'
        str3 = '497'
    elif vl_str == '560.0':
        str0 = '0560p00'
        str3 = '560'
    elif vl_str == '664.0':
        str0 = '0664p00'
        str3 = '664'
    
    ins_ba_station = globals()['matchups_Lwn_'+str0+'_fq_ins_ba_station']
    ins_pa_station = globals()['matchups_Lwn_'+str0+'_fq_ins_pa_station']
    ins_va_station = globals()['matchups_Lwn_'+str0+'_fq_ins_va_station']

    sat_ba_time = globals()['matchups_Lwn_'+str0+'_fq_sat_ba_time']
    sat_pa_time = globals()['matchups_Lwn_'+str0+'_fq_sat_pa_time']
    sat_va_time = globals()['matchups_Lwn_'+str0+'_fq_sat_va_time']


    sat_ba = globals()['matchups_Lwn_'+str0+'_fq_sat_ba']
    sat_pa = globals()['matchups_Lwn_'+str0+'_fq_sat_pa']
    sat_va = globals()['matchups_Lwn_'+str0+'_fq_sat_va']

    ins_ba = globals()['matchups_Lwn_'+str0+'_fq_ins_ba']
    ins_pa = globals()['matchups_Lwn_'+str0+'_fq_ins_pa']
    ins_va = globals()['matchups_Lwn_'+str0+'_fq_ins_va']

    count_ba_pa = 0
    count_ba_va = 0
    count_pa_va = 0
    count_all = 0
    count_ba = len(ins_ba_station)
    count_pa = len(ins_pa_station)
    count_va = len(ins_va_station)

    diff_ba_pa = []
    diff_ba_va = []
    diff_pa_va = []

    count_Venise = 0
    count_Gloria = 0
    count_Galata_Platform = 0
    count_Helsinki_Lighthouse = 0
    count_Gustav_Dalen_Tower = 0
    count_Lake_Erie = 0
    count_LISCO = 0
    count_Palgrunden = 0
    count_Thornton_C_power = 0
    count_USC_SEAPRISM = 0
    count_USC_SEAPRISM_2 = 0
    count_WaveCIS_Site_CSI_6 = 0
    
    sensor_name = 'MSI'

#%   scatter plot with all methods
    fig = plt.figure()
    ax = fig.add_subplot(111) # to put two legends in the same plot
    for cnt, line in enumerate(ins_ba_station):
        if ins_ba_station[cnt] == 'Venise': # Adriatic Sea
            mrk_style = 'x'
            mrk_color = 'r' 
            count_Venise = count_Venise+1
        elif ins_ba_station[cnt] == 'Gloria': # Black Sea    
            mrk_style = 'x'
            mrk_color = 'g'       
            count_Gloria = count_Gloria+1
        elif ins_ba_station[cnt] == 'Galata_Platform': # Black Sea
            mrk_style = 'x'
            mrk_color = 'b'
            count_Galata_Platform = count_Galata_Platform+1
        elif ins_ba_station[cnt] == 'Helsinki_Lighthouse': # Baltic Sea
            mrk_style = 'x'
            mrk_color = 'm'            
            count_Helsinki_Lighthouse = count_Helsinki_Lighthouse+1
        elif ins_ba_station[cnt] == 'Gustav_Dalen_Tower': # Baltic Sea
            mrk_style = 'x'
            mrk_color = 'c'
            count_Gustav_Dalen_Tower = count_Gustav_Dalen_Tower+1
        elif ins_ba_station[cnt] == 'Lake_Erie':
            mrk_style = 'x'
            mrk_color = 'brown'
            count_Lake_Erie = count_Lake_Erie+1
        elif ins_ba_station[cnt] == 'LISCO':
            mrk_style = 'x'
            mrk_color = 'lime'
            count_LISCO = count_LISCO+1
        elif ins_ba_station[cnt] == 'Palgrunden':
            mrk_style = 'x'
            mrk_color = 'dodgerblue'
            count_Palgrunden = count_Palgrunden+1
        elif ins_ba_station[cnt] == 'Thornton_C-power':
            mrk_style = 'x'
            mrk_color = 'crimson'
            count_Thornton_C_power = count_Thornton_C_power+1
        elif ins_ba_station[cnt] == 'USC_SEAPRISM':
            mrk_style = 'x'
            mrk_color = 'chocolate'
            count_USC_SEAPRISM = count_USC_SEAPRISM+1
        elif ins_ba_station[cnt] == 'USC_SEAPRISM_2':
            mrk_style = 'x'
            mrk_color = 'darkviolet'
            count_USC_SEAPRISM_2 = count_USC_SEAPRISM_2+1
        elif ins_ba_station[cnt] == 'WaveCIS_Site_CSI_6':
            mrk_style = 'x'
            mrk_color = 'turquoise'  
            count_WaveCIS_Site_CSI_6 = count_WaveCIS_Site_CSI_6+1
        ax.plot(ins_ba[cnt], sat_ba[cnt],marker=mrk_style,color=mrk_color)

        cond1 = sat_ba_time[cnt]==np.array(sat_pa_time) # same S2A/MSI file
        cond2 = ins_ba_station[cnt]==np.array(ins_pa_station) # same stations for both methods
        cond_ba_pa = cond1&cond2

        cond1 = sat_ba_time[cnt]==np.array(sat_va_time) # same S2A/MSI file
        cond2 = ins_ba_station[cnt]==np.array(ins_va_station) # same stations for both methods
        cond_ba_va = cond1&cond2

        # coincident mea ba and pa
        idx=np.where(cond_ba_pa)
        # print('=====================================')
        # print(cnt)
        if len(idx[0]) == 1:
            count_ba_pa = count_ba_pa+1
            ax.plot([ins_ba[cnt],ins_pa[idx[0][0]]],\
                [sat_ba[cnt],sat_pa[idx[0][0]]],color=mrk_color)
            diff_ba_pa.append(sat_ba[cnt]-sat_pa[idx[0][0]])

        # coincident mea ba and va
        idx2=np.where(cond_ba_va)
        if len(idx2[0]) == 1:
            count_ba_va = count_ba_va+1
            ax.plot([ins_ba[cnt],ins_va[idx2[0][0]]],\
                [sat_ba[cnt],sat_va[idx2[0][0]]],color=mrk_color)
            diff_ba_va.append(sat_ba[cnt]-sat_va[idx2[0][0]]) 

        if (len(idx[0]) == 1)&(len(idx2[0]) == 1): # matchup coincident to all methods
            count_all = count_all+1
                
    for cnt, line in enumerate(ins_pa_station):
        if ins_pa_station[cnt] == 'Venise': # Adriatic Sea
            mrk_style = '+'
            mrk_color = 'r' 
            count_Venise = count_Venise+1
        elif ins_pa_station[cnt] == 'Gloria': # Black Sea    
            mrk_style = '+'
            mrk_color = 'g'       
            count_Gloria = count_Gloria+1
        elif ins_pa_station[cnt] == 'Galata_Platform': # Black Sea
            mrk_style = '+'
            mrk_color = 'b'
            count_Galata_Platform = count_Galata_Platform+1
        elif ins_pa_station[cnt] == 'Helsinki_Lighthouse': # Baltic Sea
            mrk_style = '+'
            mrk_color = 'm'            
            count_Helsinki_Lighthouse = count_Helsinki_Lighthouse+1
        elif ins_pa_station[cnt] == 'Gustav_Dalen_Tower': # Baltic Sea
            mrk_style = '+'
            mrk_color = 'c'
            count_Gustav_Dalen_Tower = count_Gustav_Dalen_Tower+1
        elif ins_pa_station[cnt] == 'Lake_Erie':
            mrk_style = '+'
            mrk_color = 'brown'
            count_Lake_Erie = count_Lake_Erie+1
        elif ins_pa_station[cnt] == 'LISCO':
            mrk_style = '+'
            mrk_color = 'lime'
            count_LISCO = count_LISCO+1
        elif ins_pa_station[cnt] == 'Palgrunden':
            mrk_style = '+'
            mrk_color = 'dodgerblue'
            count_Palgrunden = count_Palgrunden+1
        elif ins_pa_station[cnt] == 'Thornton_C-power':
            mrk_style = '+'
            mrk_color = 'crimson'
            count_Thornton_C_power = count_Thornton_C_power+1
        elif ins_pa_station[cnt] == 'USC_SEAPRISM':
            mrk_style = '+'
            mrk_color = 'chocolate'
            count_USC_SEAPRISM = count_USC_SEAPRISM+1
        elif ins_pa_station[cnt] == 'USC_SEAPRISM_2':
            mrk_style = '+'
            mrk_color = 'darkviolet'
            count_USC_SEAPRISM_2 = count_USC_SEAPRISM_2+1
        elif ins_pa_station[cnt] == 'WaveCIS_Site_CSI_6':
            mrk_style = '+'
            mrk_color = 'turquoise'  
            count_WaveCIS_Site_CSI_6 = count_WaveCIS_Site_CSI_6+1
        ax.plot(ins_pa[cnt], sat_pa[cnt],marker=mrk_style,color=mrk_color)    

        cond1 = sat_pa_time[cnt]==np.array(sat_va_time) # same S2A/MSI file
        cond2 = ins_pa_station[cnt]==np.array(ins_va_station) # same stations for both methods
        cond_pa_va = cond1&cond2

        # coincident mea pa and va
        idx3=np.where(cond_pa_va)
        if len(idx3[0]) == 1:
            count_pa_va = count_pa_va+1
            ax.plot([ins_pa[cnt],ins_va[idx3[0][0]]],\
                [sat_pa[cnt],sat_va[idx3[0][0]]],color=mrk_color)
            diff_pa_va.append(sat_pa[cnt]-sat_va[idx3[0][0]]) 
            
    for cnt, line in enumerate(ins_va_station):
        if ins_va_station[cnt] == 'Venise': # Adriatic Sea
            mrk_style = 'o'
            mrk_color = 'r' 
            count_Venise = count_Venise+1
        elif ins_va_station[cnt] == 'Gloria': # Black Sea    
            mrk_style = 'o'
            mrk_color = 'g'       
            count_Gloria = count_Gloria+1
        elif ins_va_station[cnt] == 'Galata_Platform': # Black Sea
            mrk_style = 'o'
            mrk_color = 'b'
            count_Galata_Platform = count_Galata_Platform+1
        elif ins_va_station[cnt] == 'Helsinki_Lighthouse': # Baltic Sea
            mrk_style = 'o'
            mrk_color = 'm'            
            count_Helsinki_Lighthouse = count_Helsinki_Lighthouse+1
        elif ins_va_station[cnt] == 'Gustav_Dalen_Tower': # Baltic Sea
            mrk_style = 'o'
            mrk_color = 'c'
            count_Gustav_Dalen_Tower = count_Gustav_Dalen_Tower+1
        elif ins_va_station[cnt] == 'Lake_Erie':
            mrk_style = 'o'
            mrk_color = 'brown'
            count_Lake_Erie = count_Lake_Erie+1
        elif ins_va_station[cnt] == 'LISCO':
            mrk_style = 'o'
            mrk_color = 'lime'
            count_LISCO = count_LISCO+1
        elif ins_va_station[cnt] == 'Palgrunden':
            mrk_style = 'o'
            mrk_color = 'dodgerblue'
            count_Palgrunden = count_Palgrunden+1
        elif ins_va_station[cnt] == 'Thornton_C-power':
            mrk_style = 'o'
            mrk_color = 'crimson'
            count_Thornton_C_power = count_Thornton_C_power+1
        elif ins_va_station[cnt] == 'USC_SEAPRISM':
            mrk_style = 'o'
            mrk_color = 'chocolate'
            count_USC_SEAPRISM = count_USC_SEAPRISM+1
        elif ins_va_station[cnt] == 'USC_SEAPRISM_2':
            mrk_style = 'o'
            mrk_color = 'darkviolet'
            count_USC_SEAPRISM_2 = count_USC_SEAPRISM_2+1
        elif ins_va_station[cnt] == 'WaveCIS_Site_CSI_6':
            mrk_style = 'o'
            mrk_color = 'turquoise'  
            count_WaveCIS_Site_CSI_6 = count_WaveCIS_Site_CSI_6+1
        ax.plot(ins_va[cnt], sat_va[cnt],marker=mrk_style,color=mrk_color,markerfacecolor='none')         

    plt.axis([min_val, max_val, min_val, max_val])
    plt.gca().set_aspect('equal', adjustable='box')
    # plot 1:1 line
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    ax.plot([xmin,xmax],[ymin, ymax],'--k')        
    plt.xlabel('$L^{PRS}_{WN}$')
    plt.ylabel('$L^{'+sensor_name+'}_{WN}$')
    if (xmin<0 or ymin<0):
        ax.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)  
    plt.text(0.05, 0.95, str3+'nm',horizontalalignment='left', fontsize=12,transform=plt.gca().transAxes)

    # save fig
    ofname = sensor_name+'_scatter_diff_ba_pa_va_'+vl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()    

    # # time series with two methods
    # plt.figure(figsize=(16,4))

    # for cnt, line in enumerate(ins_ba_station):
    #     if ins_ba_station[cnt] == 'Venise': # Adriatic Sea
    #         mrk_style = 'x'
    #         mrk_color = 'r' 
    #     elif ins_ba_station[cnt] == 'Gloria': # Black Sea    
    #         mrk_style = 'x'
    #         mrk_color = 'g'       
    #     elif ins_ba_station[cnt] == 'Galata_Platform': # Black Sea
    #         mrk_style = 'x'
    #         mrk_color = 'b'
    #     elif ins_ba_station[cnt] == 'Helsinki_Lighthouse': # Baltic Sea
    #         mrk_style = 'x'
    #         mrk_color = 'm'            
    #     elif ins_ba_station[cnt] == 'Gustav_Dalen_Tower': # Baltic Sea
    #         mrk_style = 'x'
    #         mrk_color = 'c'
    #     elif ins_ba_station[cnt] == 'Lake_Erie':
    #         mrk_style = 'x'
    #         mrk_color = 'brown'
    #     elif ins_ba_station[cnt] == 'LISCO':
    #         mrk_style = 'x'
    #         mrk_color = 'lime'
    #     elif ins_ba_station[cnt] == 'Palgrunden':
    #         mrk_style = 'x'
    #         mrk_color = 'dodgerblue'
    #     elif ins_ba_station[cnt] == 'Thornton_C-power':
    #         mrk_style = 'x'
    #         mrk_color = 'crimson'
    #     elif ins_ba_station[cnt] == 'USC_SEAPRISM':
    #         mrk_style = 'x'
    #         mrk_color = 'chocolate'
    #     elif ins_ba_station[cnt] == 'USC_SEAPRISM_2':
    #         mrk_style = 'x'
    #         mrk_color = 'darkviolet'
    #     elif ins_ba_station[cnt] == 'WaveCIS_Site_CSI_6':
    #         mrk_style = 'x'
    #         mrk_color = 'turquoise'            
                
    #     cond1 = sat_ba_time[cnt]==np.array(sat_pa_time) # same S2A/MSI file
    #     cond2 = ins_ba_station[cnt]==np.array(ins_pa_station) # same stations for both methods
    #     cond_ba_pa = cond1&cond2

    #     cond1 = sat_ba_time[cnt]==np.array(sat_va_time) # same S2A/MSI file
    #     cond2 = ins_ba_station[cnt]==np.array(ins_va_station) # same stations for both methods
    #     cond_ba_va = cond1&cond2

    #     # coincident mea ba and pa
    #     idx=np.where(cond_ba_pa)
    #     # print('=====================================')
    #     # print(cnt)
    #     if len(idx[0]) == 1:
    #         count_ba_pa = count_ba_pa+1
    #         # print('There is coincident matchups in BW.')
    #         # print(sat_zi_time[cnt])
    #         # print(sat_ba_time[idx[0][0]])
    #         # print(ins_zi_station[cnt])
    #         # print(ins_ba_station[idx[0][0]])
    #         plt.plot(sat_ba_time[cnt], sat_ba[cnt],marker=mrk_style,color=mrk_color)
    #         plt.plot([sat_ba_time[cnt],sat_pa_time[idx[0][0]]],\
    #             [sat_ba[cnt],sat_pa[idx[0][0]]],color=mrk_color)
            
    #     else:
    #         # print(idx[0])
    #         # print(sat_ba_time[cnt])
    #         plt.plot(sat_ba_time[cnt], sat_ba[cnt],mrk_style)           
                
    # for cnt, line in enumerate(ins_pa_station):
    #     if ins_pa_station[cnt] == 'Venise': # Adriatic Sea
    #         mrk_style = '+'
    #         mrk_color = 'r' 
    #     elif ins_pa_station[cnt] == 'Gloria': # Black Sea    
    #         mrk_style = '+'
    #         mrk_color = 'g'       
    #     elif ins_pa_station[cnt] == 'Galata_Platform': # Black Sea
    #         mrk_style = '+'
    #         mrk_color = 'b'
    #     elif ins_pa_station[cnt] == 'Helsinki_Lighthouse': # Baltic Sea
    #         mrk_style = '+'
    #         mrk_color = 'm'            
    #     elif ins_pa_station[cnt] == 'Gustav_Dalen_Tower': # Baltic Sea
    #         mrk_style = '+'
    #         mrk_color = 'c'
    #     elif ins_pa_station[cnt] == 'Lake_Erie':
    #         mrk_style = '+'
    #         mrk_color = 'brown'
    #     elif ins_pa_station[cnt] == 'LISCO':
    #         mrk_style = '+'
    #         mrk_color = 'lime'
    #     elif ins_pa_station[cnt] == 'Palgrunden':
    #         mrk_style = '+'
    #         mrk_color = 'dodgerblue'
    #     elif ins_pa_station[cnt] == 'Thornton_C-power':
    #         mrk_style = '+'
    #         mrk_color = 'crimson'
    #     elif ins_pa_station[cnt] == 'USC_SEAPRISM':
    #         mrk_style = '+'
    #         mrk_color = 'chocolate'
    #     elif ins_pa_station[cnt] == 'USC_SEAPRISM_2':
    #         mrk_style = '+'
    #         mrk_color = 'darkviolet'
    #     elif ins_pa_station[cnt] == 'WaveCIS_Site_CSI_6':
    #         mrk_style = '+'
    #         mrk_color = 'turquoise'          
    #     plt.plot(sat_pa_time[cnt], sat_pa[cnt],marker=mrk_style,color=mrk_color)

    # for cnt, line in enumerate(ins_va_station):
    #     if ins_va_station[cnt] == 'Venise': # Adriatic Sea
    #         mrk_style = 'o'
    #         mrk_color = 'r' 
    #     elif ins_va_station[cnt] == 'Gloria': # Black Sea    
    #         mrk_style = 'o'
    #         mrk_color = 'g'       
    #     elif ins_va_station[cnt] == 'Galata_Platform': # Black Sea
    #         mrk_style = 'o'
    #         mrk_color = 'b'
    #     elif ins_va_station[cnt] == 'Helsinki_Lighthouse': # Baltic Sea
    #         mrk_style = 'o'
    #         mrk_color = 'm'            
    #     elif ins_va_station[cnt] == 'Gustav_Dalen_Tower': # Baltic Sea
    #         mrk_style = 'o'
    #         mrk_color = 'c'
    #     elif ins_va_station[cnt] == 'Lake_Erie':
    #         mrk_style = 'o'
    #         mrk_color = 'brown'
    #     elif ins_va_station[cnt] == 'LISCO':
    #         mrk_style = 'o'
    #         mrk_color = 'lime'
    #     elif ins_va_station[cnt] == 'Palgrunden':
    #         mrk_style = 'o'
    #         mrk_color = 'dodgerblue'
    #     elif ins_va_station[cnt] == 'Thornton_C-power':
    #         mrk_style = 'o'
    #         mrk_color = 'crimson'
    #     elif ins_va_station[cnt] == 'USC_SEAPRISM':
    #         mrk_style = 'o'
    #         mrk_color = 'chocolate'
    #     elif ins_va_station[cnt] == 'USC_SEAPRISM_2':
    #         mrk_style = 'o'
    #         mrk_color = 'darkviolet'
    #     elif ins_va_station[cnt] == 'WaveCIS_Site_CSI_6':
    #         mrk_style = 'o'
    #         mrk_color = 'turquoise'  
    #     plt.plot(sat_va_time[cnt], sat_va[cnt],marker=mrk_style,color=mrk_color,markerfacecolor='none') 


    # plt.xlabel('Time')
    # sensor_name = 'OLCI'
    # plt.ylabel('$L^{'+sensor_name+'}_{WN}$')
    # # zero line
    # xmin, xmax = plt.gca().get_xlim()
    # plt.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)  
    # plt.text(0.05, 0.95, str3+'nm',horizontalalignment='left', fontsize=12,transform=plt.gca().transAxes)

    # # save fig
    # ofname = sensor_name+'_timeseries_diff_ba_pa_va_'+vl_str.replace(".","p")+'.pdf'
    # ofname = os.path.join(path_out,'source',ofname)
    # plt.savefig(ofname, dpi=300)

    # plt.show()

    # histograms of all datasets: ba, pa and va
    kwargs2 = dict(bins='auto', histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w')
    ax1.hist(sat_ba,color='red', **kwargs2)
    ax1.hist(sat_pa,color='black', **kwargs2)
    ax1.hist(sat_va,color='blue', **kwargs2)
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim([x0,x0+1*(x1-x0)])

    ax1.set_ylabel('Frequency')

    str1 = 'BW06\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.2f}\nmean: {:,.2f}\nN: {:,.0f}'\
    .format(np.nanmin(sat_ba),
            np.nanmax(sat_ba),
            np.nanstd(sat_ba),
            np.nanmedian(sat_ba),
            np.nanmean(sat_ba),
            len(sat_ba))

    str2 = 'IPK19\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.2f}\nmean: {:,.2f}\nN: {:,.0f}'\
    .format(np.nanmin(sat_pa),
            np.nanmax(sat_pa),
            np.nanstd(sat_pa),
            np.nanmedian(sat_pa),
            np.nanmean(sat_pa),
            len(sat_pa))

    str4 = 'V19\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.2f}\nmean: {:,.2f}\nN: {:,.0f}'\
    .format(np.nanmin(sat_va),
            np.nanmax(sat_va),
            np.nanstd(sat_va),
            np.nanmedian(sat_va),
            np.nanmean(sat_va),
            len(sat_va))

    bottom, top = ax1.get_ylim()
    left, right = ax1.get_xlim()
    xpos = left+0.70*(right-left)
    ax1.text(left+0.01*(right-left),bottom+0.95*(top-bottom), '{}nm'.format(str3), fontsize=12,color='black')
    ax1.text(xpos,bottom+0.70*(top-bottom), str1, fontsize=10,color='red')
    ax1.text(xpos,bottom+0.40*(top-bottom), str2, fontsize=10,color='black')
    ax1.text(xpos,bottom+0.10*(top-bottom), str4, fontsize=10,color='blue')

    fig.text(0.5,0.01,'$L^{'+sensor_name+'}_{WN}$',ha='center',fontsize=12)

    # save fig
    ofname = sensor_name+'_hist_ba_pa_va_'+vl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)

    plt.show()

    #latex table
    if vl_str == '444.0':
        print('proto & nm & min & max & std & median & mean & N\\\\')
    str_table = 'BW06 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(vl_str,\
            np.nanmin(sat_ba),
            np.nanmax(sat_ba),
            np.nanstd(sat_ba),
            np.nanmedian(sat_ba),
            np.nanmean(sat_ba),
            len(sat_ba))
    print(str_table)
    str_table = 'IPK19 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(vl_str,\
            np.nanmin(sat_pa),
            np.nanmax(sat_pa),
            np.nanstd(sat_pa),
            np.nanmedian(sat_pa),
            np.nanmean(sat_pa),
            len(sat_pa))
    print(str_table)  
    str_table = 'V19 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.0f}\\\\'\
    .format(vl_str,\
            np.nanmin(sat_va),
            np.nanmax(sat_va),
            np.nanstd(sat_va),
            np.nanmedian(sat_va),
            np.nanmean(sat_va),
            len(sat_va))
    print(str_table)  
    
    # histogram of the difference
    kwargs2 = dict(bins='auto', histtype='step')
    fig, ax1=plt.subplots(1,1,sharey=True, facecolor='w')
    ax1.hist(diff_ba_pa,color='#1f77b4', **kwargs2)
    ax1.hist(diff_ba_va,color='#ff7f0e', **kwargs2)
    ax1.hist(diff_pa_va,color='#2ca02c', **kwargs2)

    ax1.set_ylabel('Frequency',fontsize=12)

    str1 = 'BW06-IPK19\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.5f}\nmean: {:,.3f}\nN: {:,.0f}'\
    .format(np.nanmin(diff_ba_pa),
            np.nanmax(diff_ba_pa),
            np.nanstd(diff_ba_pa),
            np.nanmedian(diff_ba_pa),
            np.nanmean(diff_ba_pa),
            len(diff_ba_pa))

    # latex table
    if vl_str == '444.0':
        print('BW06-IPK19 & nm & min & max & std & median & mean & N\\\\')
    str_table = 'BW06-IPK19 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.4f} & {:,.4f} & {:,.0f}\\\\'\
    .format(str3,
            np.nanmin(diff_ba_pa),
            np.nanmax(diff_ba_pa),
            np.nanstd(diff_ba_pa),
            np.nanmedian(diff_ba_pa),
            np.nanmean(diff_ba_pa),
            len(diff_ba_pa))
    print(str_table)

    str2 = 'BW06-V19\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.5f}\nmean: {:,.3f}\nN: {:,.0f}'\
    .format(np.nanmin(diff_ba_va),
            np.nanmax(diff_ba_va),
            np.nanstd(diff_ba_va),
            np.nanmedian(diff_ba_va),
            np.nanmean(diff_ba_va),
            len(diff_ba_va))

    # latex table
    if vl_str == '444.0':
        print('BW06-V19 & nm & min & max & std & median & mean & N\\\\')
    str_table = 'BW06-V19 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.4f} & {:,.4f} & {:,.0f}\\\\'\
    .format(str3,
            np.nanmin(diff_ba_va),
            np.nanmax(diff_ba_va),
            np.nanstd(diff_ba_va),
            np.nanmedian(diff_ba_va),
            np.nanmean(diff_ba_va),
            len(diff_ba_va))
    print(str_table)

    str4 = 'IPK19-V19\nmin: {:,.2f}\nmax: {:,.2f}\nstd: {:,.2f}\nmedian: {:,.5f}\nmean: {:,.3f}\nN: {:,.0f}'\
    .format(np.nanmin(diff_pa_va),
            np.nanmax(diff_pa_va),
            np.nanstd(diff_pa_va),
            np.nanmedian(diff_pa_va),
            np.nanmean(diff_pa_va),
            len(diff_pa_va))

    # latex table
    if vl_str == '444.0':
        print('IPK19-V19 & nm & min & max & std & median & mean & N\\\\')
    str_table = 'IPK19-V19 & {} & {:,.2f} & {:,.2f} & {:,.2f} & {:,.4f} & {:,.4f} & {:,.0f}\\\\'\
    .format(str3,
            np.nanmin(diff_pa_va),
            np.nanmax(diff_pa_va),
            np.nanstd(diff_pa_va),
            np.nanmedian(diff_pa_va),
            np.nanmean(diff_pa_va),
            len(diff_pa_va))
    print(str_table)

    bottom, top = ax1.get_ylim()
    left, right = ax1.get_xlim()
    ax1.text(left+0.700*(right-left),bottom+0.70*(top-bottom), str1, fontsize=10,color='#1f77b4')
    ax1.text(left+0.700*(right-left),bottom+0.40*(top-bottom), str2, fontsize=10,color='#ff7f0e')
    ax1.text(left+0.700*(right-left),bottom+0.10*(top-bottom), str4, fontsize=10,color='#2ca02c')

    ax1.text(left+0.05*(right-left),bottom+0.95*(top-bottom), str3+'nm', fontsize=10,color='black')

    fig.text(0.5,0.01,'Diff. $L^{'+sensor_name+'}_{WN}$',ha='center',fontsize=12)

    # save fig
    ofname = sensor_name+'_hist_diff_ba_pa_va_'+vl_str.replace(".","p")+'.pdf'
    ofname = os.path.join(path_out,'source',ofname)
    plt.savefig(ofname, dpi=300)
    plt.show()

    print('Number of Matchups for BW06: '+str(count_ba))
    print('Number of Matchups for IPK19: '+str(count_pa))
    print('Number of Matchups for V19: '+str(count_va))
    print('Number of Matchups for BW06 and IPK19: '+str(count_ba_pa))
    print('Number of Matchups for BW06 and V19: '+str(count_ba_va))
    print('Number of Matchups for IPK19 and V19: '+str(count_pa_va))
    print('Number of Matchups for BW06, IPK19, V19: '+str(count_all))

    # print('count_Venise: '+str(count_Venise))
    # print('count_Gloria: '+str(count_Gloria))
    # print('count_Galata_Platform: '+str(count_Galata_Platform))
    # print('count_Helsinki_Lighthouse: '+str(count_Helsinki_Lighthouse))
    # print('count_Gustav_Dalen_Tower: '+str(count_Gustav_Dalen_Tower))
    # print('count_Lake_Erie: '+str(count_Lake_Erie))
    # print('count_LISCO: '+str(count_LISCO))
    # print('count_Palgrunden: '+str(count_Palgrunden))
    # print('count_Thornton_C_power: '+str(count_Thornton_C_power))
    # print('count_USC_SEAPRISM: '+str(count_USC_SEAPRISM))
    # print('count_USC_SEAPRISM_2: '+str(count_USC_SEAPRISM_2))
    # print('count_WaveCIS_Site_CSI_6: '+str(count_WaveCIS_Site_CSI_6))

#%%
#def main():
path_main = '/Users/javier.concha/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'
path_out = os.path.join(path_main,'Figures')
path_data = '/Users/javier.concha/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/Quinten_data/'
sensor_name = 'S2A'
path_to_list = os.path.join(path_data,sensor_name,'aeronet_list.txt') 

# Solar spectral irradiance F0 in uW/cm^2/nm 
F0_0444p00 = common_functions.get_F0(444.0,path_main)
F0_0497p00 = common_functions.get_F0(497.0,path_main)
F0_0560p00 = common_functions.get_F0(560.0,path_main)
F0_0664p00 = common_functions.get_F0(664.0,path_main) 

# Bailey and Werdell: initialization
matchups_Lwn_0444p00_fq_ins_ba = []
matchups_Lwn_0497p00_fq_ins_ba = []
matchups_Lwn_0560p00_fq_ins_ba = []
matchups_Lwn_0664p00_fq_ins_ba = []

matchups_Lwn_0444p00_fq_sat_ba = []
matchups_Lwn_0497p00_fq_sat_ba = []
matchups_Lwn_0560p00_fq_sat_ba = []
matchups_Lwn_0664p00_fq_sat_ba = []

matchups_Lwn_0444p00_fq_ins_ba_station = []
matchups_Lwn_0497p00_fq_ins_ba_station = []
matchups_Lwn_0560p00_fq_ins_ba_station = []
matchups_Lwn_0664p00_fq_ins_ba_station = []

matchups_Lwn_0444p00_fq_sat_ba_time = []
matchups_Lwn_0497p00_fq_sat_ba_time = []
matchups_Lwn_0560p00_fq_sat_ba_time = []
matchups_Lwn_0664p00_fq_sat_ba_time = []

# Pahlevan: initialization
matchups_Lwn_0444p00_fq_ins_pa = []
matchups_Lwn_0497p00_fq_ins_pa = []
matchups_Lwn_0560p00_fq_ins_pa = []
matchups_Lwn_0664p00_fq_ins_pa = []

matchups_Lwn_0444p00_fq_sat_pa = []
matchups_Lwn_0497p00_fq_sat_pa = []
matchups_Lwn_0560p00_fq_sat_pa = []
matchups_Lwn_0664p00_fq_sat_pa = []

matchups_Lwn_0444p00_fq_ins_pa_station = []
matchups_Lwn_0497p00_fq_ins_pa_station = []
matchups_Lwn_0560p00_fq_ins_pa_station = []
matchups_Lwn_0664p00_fq_ins_pa_station = []

matchups_Lwn_0444p00_fq_sat_pa_time = []
matchups_Lwn_0497p00_fq_sat_pa_time = []
matchups_Lwn_0560p00_fq_sat_pa_time = []
matchups_Lwn_0664p00_fq_sat_pa_time = []

# Vanhellemont: initialization
matchups_Lwn_0444p00_fq_ins_va = []
matchups_Lwn_0497p00_fq_ins_va = []
matchups_Lwn_0560p00_fq_ins_va = []
matchups_Lwn_0664p00_fq_ins_va = []

matchups_Lwn_0444p00_fq_sat_va = []
matchups_Lwn_0497p00_fq_sat_va = []
matchups_Lwn_0560p00_fq_sat_va = []
matchups_Lwn_0664p00_fq_sat_va = []

matchups_Lwn_0444p00_fq_ins_va_station = []
matchups_Lwn_0497p00_fq_ins_va_station = []
matchups_Lwn_0560p00_fq_ins_va_station = []
matchups_Lwn_0664p00_fq_ins_va_station = []

matchups_Lwn_0444p00_fq_sat_va_time = []
matchups_Lwn_0497p00_fq_sat_va_time = []
matchups_Lwn_0560p00_fq_sat_va_time = []
matchups_Lwn_0664p00_fq_sat_va_time = []

donut_mask = np.array([[False,False,False,False,False,False,False],\
              [False,False,False,False,False,False,False],\
              [False,False,True,True,True,False,False],\
              [False,False,True,True,True,False,False],\
              [False,False,True,True,True,False,False],\
              [False,False,False,False,False,False,False],\
              [False,False,False,False,False,False,False]])
   
with open(path_to_list,'r') as file_list:
    for cnt, file_name in enumerate(file_list):
        folder_name = file_name.split('/')[1]
        station_name = folder_name[:-15]
        year_str = folder_name.split('_')[-3]
        month_str = folder_name.split('_')[-2]
        day_str = folder_name.split('_')[-1]
        
        # sat data
        path_to_nc = glob.glob(os.path.join(path_data,sensor_name,folder_name)+'/*.nc')[0]
         
        nc_f0 = Dataset(path_to_nc,'r')
        
        date_format = "%Y-%m-%dT%H:%M:%S.%fZ" 
        sat_time = datetime.strptime(nc_f0.isodate, date_format)
        
        lat = nc_f0.variables['lat'][:]
        lon = nc_f0.variables['lon'][:]
        
        in_situ_lat, in_situ_lon = get_lat_lon_ins(station_name)
        
        r, c = common_functions.find_row_column_from_lat_lon(lat,lon,in_situ_lat,in_situ_lon)

        rhos_0444p00 =  nc_f0.variables['rhos_444'][:]
        rhos_0497p00 =  nc_f0.variables['rhos_497'][:]
        rhos_0560p00 =  nc_f0.variables['rhos_560'][:]
        rhos_0664p00 =  nc_f0.variables['rhos_664'][:]
        rhos_0704p00 =  nc_f0.variables['rhos_704'][:]
        rhos_0740p00 =  nc_f0.variables['rhos_740'][:]
        rhos_0782p00 =  nc_f0.variables['rhos_782'][:]
        rhos_0835p00 =  nc_f0.variables['rhos_835'][:]
        rhos_0865p00 =  nc_f0.variables['rhos_865'][:]
        rhos_1614p00 =  nc_f0.variables['rhos_1614'][:]
        rhos_2202p00 =  nc_f0.variables['rhos_2202'][:]
        
        rhot_1614p00 =  nc_f0.variables['rhot_1614'][:]
        
        sza = nc_f0.THS
        vza = nc_f0.THV
                
        nc_f0.close()

        # in situ data        
        with open(os.path.join(path_data,sensor_name,file_name[2:-1]), 'r') as file:
            ins_time = []
            # Date
            line_num_date = 3-1
            line_num_time = 4-1
            ins_all_lines = file.readlines()
            line_date = ins_all_lines[line_num_date]   
            line_time = ins_all_lines[line_num_time] 
            
            str_list_date = line_date[:-1].replace('=',',').split(',')
            str_list_time = line_time[:-1].replace('=',',').split(',')
            
            n_ins = len(str_list_date)-1
            for i in range(n_ins):
                date_and_time_str = str_list_date[1:][i]+' '+str_list_time[1:][i]
                date_format = "%d/%m/%Y %H:%M:%S"
                ins_time.append(datetime.strptime(date_and_time_str, date_format))

        # Bailey and Werdell 2006 
        print('--Bailey and Werdell 2006')
        delta_time = 3# float in hours   
        ins_time = np.array(ins_time)
        time_diff = ins_time - sat_time
        dt_hour = [i.total_seconds()/(60*60) for i in time_diff] # time diffence between in situ measurements and sat in hours
        idx_min = np.argmin(np.abs(dt_hour))
        matchup_idx_vec = np.abs(dt_hour) <= delta_time 
        
#           104	Lwn(442)=0.285881,0.295781,0.288577,0.269275,0.233640,0.196600
#   105	Lwn(491)=0.532647,0.578646,0.523528,0.501363,0.397166,0.397819
#   106	Lwn(530)=0.759344,0.803289,0.810068,0.715048,0.648792,0.641661
#   107	Lwn(551)=0.855308,0.897261,0.892521,0.827199,0.776998,0.742645
#   108	Lwn(668)=0.246346,0.269323,0.275534,0.258861,0.244163,0.249685
        
        Lwn_fq_0442p00 = float(ins_all_lines[104-1][:-1].replace('=',',').split(',')[idx_min+1]) 
        Lwn_fq_0491p00 = float(ins_all_lines[105-1][:-1].replace('=',',').split(',')[idx_min+1]) 
        Lwn_fq_00551p0 = float(ins_all_lines[107-1][:-1].replace('=',',').split(',')[idx_min+1]) 
        Lwn_fq_0668p00 = float(ins_all_lines[108-1][:-1].replace('=',',').split(',')[idx_min+1]) 

        nday = sum(matchup_idx_vec)
        if nday >=1:
            print(str(nday)+' matchups for '+folder_name)
            
            size_box = 5
            NTP = size_box*size_box # Number Total Pixels, excluding land pixels, Bailey and Werdell 2006
            start_idx_x = (r-int(size_box/2))
            stop_idx_x = (r+int(size_box/2)+1)
            start_idx_y = (c-int(size_box/2))
            stop_idx_y = (c+int(size_box/2)+1)

            rhos_0444p00_box = rhos_0444p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0497p00_box = rhos_0497p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0560p00_box = rhos_0560p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0664p00_box = rhos_0664p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0704p00_box = rhos_0704p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0740p00_box = rhos_0740p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0782p00_box = rhos_0782p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0835p00_box = rhos_0835p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0865p00_box = rhos_0865p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_1614p00_box = rhos_1614p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_2202p00_box = rhos_2202p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]  
            
            # for cloud screening
            rhot_1614p00_box = rhot_1614p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y] 
            
            flags_mask = rhot_1614p00_box>0.0215
            
            rhos_0444p00_box.mask =  rhos_0444p00_box.mask or flags_mask 
            rhos_0497p00_box.mask =  rhos_0497p00_box.mask or flags_mask
            rhos_0560p00_box.mask =  rhos_0560p00_box.mask or flags_mask
            rhos_0664p00_box.mask =  rhos_0664p00_box.mask or flags_mask
            
            NGP = np.count_nonzero(flags_mask == 0) # Number Good Pixels, Bailey and Werdell 2006
                
            if sza<=75 and vza<=60 and NGP>NTP/2+1 and rhot_1614p00_box.mean()<=0.01:
                # if nan, change mask            
                rhos_0444p00_box = ma.masked_invalid(rhos_0444p00_box)
                rhos_0497p00_box = ma.masked_invalid(rhos_0497p00_box)
                rhos_0560p00_box = ma.masked_invalid(rhos_0560p00_box)
                rhos_0664p00_box = ma.masked_invalid(rhos_0664p00_box)

                NGP_rhos_0444p00 = np.count_nonzero(rhos_0444p00_box.mask == 0)
                NGP_rhos_0497p00 = np.count_nonzero(rhos_0497p00_box.mask == 0)
                NGP_rhos_0560p00 = np.count_nonzero(rhos_0560p00_box.mask == 0)
                NGP_rhos_0664p00 = np.count_nonzero(rhos_0664p00_box.mask == 0)

                mean_unfiltered_rhos_0444p00 = rhos_0444p00_box.mean()
                mean_unfiltered_rhos_0497p00 = rhos_0497p00_box.mean()
                mean_unfiltered_rhos_0560p00 = rhos_0560p00_box.mean()
                mean_unfiltered_rhos_0664p00 = rhos_0664p00_box.mean()
 
                std_unfiltered_rhos_0444p00 = rhos_0444p00_box.std()
                std_unfiltered_rhos_0497p00 = rhos_0497p00_box.std()
                std_unfiltered_rhos_0560p00 = rhos_0560p00_box.std()
                std_unfiltered_rhos_0664p00 = rhos_0664p00_box.std()

                # mask values that are not within +/- 1.5*std of mean\               
                rhos_0444p00_box = ma.masked_outside(rhos_0444p00_box,mean_unfiltered_rhos_0444p00\
                    -1.5*std_unfiltered_rhos_0444p00\
                    , mean_unfiltered_rhos_0444p00\
                    +1.5*std_unfiltered_rhos_0444p00)
                rhos_0497p00_box = ma.masked_outside(rhos_0497p00_box,mean_unfiltered_rhos_0497p00\
                    -1.5*std_unfiltered_rhos_0497p00\
                    , mean_unfiltered_rhos_0497p00\
                    +1.5*std_unfiltered_rhos_0497p00)
                rhos_0560p00_box = ma.masked_outside(rhos_0560p00_box,mean_unfiltered_rhos_0560p00\
                    -1.5*std_unfiltered_rhos_0560p00\
                    , mean_unfiltered_rhos_0560p00\
                    +1.5*std_unfiltered_rhos_0560p00)
                rhos_0664p00_box = ma.masked_outside(rhos_0664p00_box,mean_unfiltered_rhos_0664p00\
                    -1.5*std_unfiltered_rhos_0664p00\
                    , mean_unfiltered_rhos_0664p00\
                    +1.5*std_unfiltered_rhos_0664p00)

                mean_filtered_rhos_0444p00 = rhos_0444p00_box.mean()
                mean_filtered_rhos_0497p00 = rhos_0497p00_box.mean()
                mean_filtered_rhos_0560p00 = rhos_0560p00_box.mean()
                mean_filtered_rhos_0664p00 = rhos_0664p00_box.mean()

                std_filtered_rhos_0444p00 = rhos_0444p00_box.std()
                std_filtered_rhos_0497p00 = rhos_0497p00_box.std()
                std_filtered_rhos_0560p00 = rhos_0560p00_box.std()
                std_filtered_rhos_0664p00 = rhos_0664p00_box.std()

                CV_filtered_rhos_0444p00 = std_filtered_rhos_0444p00/mean_filtered_rhos_0444p00
                CV_filtered_rhos_0497p00 = std_filtered_rhos_0497p00/mean_filtered_rhos_0497p00
                CV_filtered_rhos_0560p00 = std_filtered_rhos_0560p00/mean_filtered_rhos_0560p00
                CV_filtered_rhos_0664p00 = std_filtered_rhos_0664p00/mean_filtered_rhos_0664p00
                
                CVs = [CV_filtered_rhos_0444p00,CV_filtered_rhos_0497p00, CV_filtered_rhos_0560p00]
                print(CVs)
                MedianCV = np.nanmedian(np.abs(CVs))

                print('Median CV={:.4f}'.format(MedianCV))
               
                if MedianCV <= 0.15:
                    # Rrs 0444p00
                    print('444.0')
                    if NGP_rhos_0444p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0444p00={:.0f}'.format(NGP_rhos_0444p00))
                    else:
                        matchups_Lwn_0444p00_fq_sat_ba.append(mean_filtered_rhos_0444p00*F0_0444p00/np.pi)
                        matchups_Lwn_0444p00_fq_ins_ba.append(Lwn_fq_0442p00)
                        matchups_Lwn_0444p00_fq_ins_ba_station.append(station_name)
                        matchups_Lwn_0444p00_fq_sat_ba_time.append(sat_time)
                    # Rrs 0497p00
                    print('497.0')
                    if NGP_rhos_0497p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0497p00={:.0f}'.format(NGP_rhos_0497p00))
                    else:
                        matchups_Lwn_0497p00_fq_sat_ba.append(mean_filtered_rhos_0497p00*F0_0497p00/np.pi)
                        matchups_Lwn_0497p00_fq_ins_ba.append(Lwn_fq_0491p00)
                        matchups_Lwn_0497p00_fq_ins_ba_station.append(station_name)
                        matchups_Lwn_0497p00_fq_sat_ba_time.append(sat_time)
                    # Rrs 0560p00
                    print('560.0')
                    if NGP_rhos_0560p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0560p00={:.0f}'.format(NGP_rhos_0560p00))
                    else:
                        matchups_Lwn_0560p00_fq_sat_ba.append(mean_filtered_rhos_0560p00*F0_0560p00/np.pi)
                        matchups_Lwn_0560p00_fq_ins_ba.append(Lwn_fq_00551p0)
                        matchups_Lwn_0560p00_fq_ins_ba_station.append(station_name)
                        matchups_Lwn_0560p00_fq_sat_ba_time.append(sat_time)
                    # Rrs 0664p00
                    print('664.0')
                    if NGP_rhos_0664p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0664p00={:.0f}'.format(NGP_rhos_0664p00))
                    else:
                        matchups_Lwn_0664p00_fq_sat_ba.append(mean_filtered_rhos_0664p00*F0_0664p00/np.pi)
                        matchups_Lwn_0664p00_fq_ins_ba.append(Lwn_fq_0668p00)  
                        matchups_Lwn_0664p00_fq_ins_ba_station.append(station_name)
                        matchups_Lwn_0664p00_fq_sat_ba_time.append(sat_time)  
                else:
                    print('Median CV exceeds criteria: Median[CV]={:.4f}'.format(MedianCV))
            else:
                print('Angles exceeds criteria: sza={:.2f}'.format(sza)+'; vza={:.2f}'.format(vza)+\
                    '; OR mean of rhot_1614={:.4f}'.format(rhot_1614p00_box.mean())+'>0.01!\n'+\
                    '; OR NGP={:.0f}'.format(NGP)+'< NTP/2+1={:.0f}'.format(NTP/2+1)+'!')
#                print('Angles exceeds criteria: sza='+str(sza)+'; vza='+str(vza)+'; OR NGP='+str(NGP)+'< NTP/2+1='+str(NTP/2+1)+'!')
        else:
            print('Not matchups for '+folder_name) 

        # Pahlevan based on Bailey and Werdell 2006 
        print('--Pahlevan based on Bailey and Werdell 2006')
        delta_time = 0.5# float in hours   
        ins_time = np.array(ins_time)
        time_diff = ins_time - sat_time
        dt_hour = [i.total_seconds()/(60*60) for i in time_diff] # time diffence between in situ measurements and sat in hours
        idx_min = np.argmin(np.abs(dt_hour))
        matchup_idx_vec = np.abs(dt_hour) <= delta_time 

        Lwn_fq_0442p00 = float(ins_all_lines[104-1][:-1].replace('=',',').split(',')[idx_min+1]) 
        Lwn_fq_0491p00 = float(ins_all_lines[105-1][:-1].replace('=',',').split(',')[idx_min+1]) 
        Lwn_fq_00551p0 = float(ins_all_lines[107-1][:-1].replace('=',',').split(',')[idx_min+1]) 
        Lwn_fq_0668p00 = float(ins_all_lines[108-1][:-1].replace('=',',').split(',')[idx_min+1])        

        nday = sum(matchup_idx_vec)
        if nday >=1:
            print(str(nday)+' matchups for '+folder_name)
            
            size_box = 7
            NTP = size_box*size_box # Number Total Pixels, excluding land pixels, Bailey and Werdell 2006
            start_idx_x = (r-int(size_box/2))
            stop_idx_x = (r+int(size_box/2)+1)
            start_idx_y = (c-int(size_box/2))
            stop_idx_y = (c+int(size_box/2)+1)

            rhos_0444p00_box = rhos_0444p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0497p00_box = rhos_0497p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0560p00_box = rhos_0560p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0664p00_box = rhos_0664p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0704p00_box = rhos_0704p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0740p00_box = rhos_0740p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0782p00_box = rhos_0782p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0835p00_box = rhos_0835p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_0865p00_box = rhos_0865p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_1614p00_box = rhos_1614p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]
            rhos_2202p00_box = rhos_2202p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y]  
            
            # for cloud screening
            rhot_1614p00_box = rhot_1614p00[start_idx_x:stop_idx_x,start_idx_y:stop_idx_y] 
            
            flags_mask = rhot_1614p00_box>0.0215
            
            rhos_0444p00_box.mask =  rhos_0444p00_box.mask | flags_mask | donut_mask 
            rhos_0497p00_box.mask =  rhos_0497p00_box.mask | flags_mask | donut_mask
            rhos_0560p00_box.mask =  rhos_0560p00_box.mask | flags_mask | donut_mask
            rhos_0664p00_box.mask =  rhos_0664p00_box.mask | flags_mask | donut_mask
            
            NGP = np.count_nonzero(flags_mask == 0) # Number Good Pixels, Bailey and Werdell 2006
                
            if sza<=75 and vza<=60 and NGP>NTP/2+1 and rhot_1614p00_box.mean()<=0.01:
                # if nan, change mask            
                rhos_0444p00_box = ma.masked_invalid(rhos_0444p00_box)
                rhos_0497p00_box = ma.masked_invalid(rhos_0497p00_box)
                rhos_0560p00_box = ma.masked_invalid(rhos_0560p00_box)
                rhos_0664p00_box = ma.masked_invalid(rhos_0664p00_box)

                NGP_rhos_0444p00 = np.count_nonzero(rhos_0444p00_box.mask == 0)
                NGP_rhos_0497p00 = np.count_nonzero(rhos_0497p00_box.mask == 0)
                NGP_rhos_0560p00 = np.count_nonzero(rhos_0560p00_box.mask == 0)
                NGP_rhos_0664p00 = np.count_nonzero(rhos_0664p00_box.mask == 0)

                mean_unfiltered_rhos_0444p00 = rhos_0444p00_box.mean()
                mean_unfiltered_rhos_0497p00 = rhos_0497p00_box.mean()
                mean_unfiltered_rhos_0560p00 = rhos_0560p00_box.mean()
                mean_unfiltered_rhos_0664p00 = rhos_0664p00_box.mean()
 
                std_unfiltered_rhos_0444p00 = rhos_0444p00_box.std()
                std_unfiltered_rhos_0497p00 = rhos_0497p00_box.std()
                std_unfiltered_rhos_0560p00 = rhos_0560p00_box.std()
                std_unfiltered_rhos_0664p00 = rhos_0664p00_box.std()

                # mask values that are not within +/- 1.5*std of mean\               
                rhos_0444p00_box = ma.masked_outside(rhos_0444p00_box,mean_unfiltered_rhos_0444p00\
                    -1.5*std_unfiltered_rhos_0444p00\
                    , mean_unfiltered_rhos_0444p00\
                    +1.5*std_unfiltered_rhos_0444p00)
                rhos_0497p00_box = ma.masked_outside(rhos_0497p00_box,mean_unfiltered_rhos_0497p00\
                    -1.5*std_unfiltered_rhos_0497p00\
                    , mean_unfiltered_rhos_0497p00\
                    +1.5*std_unfiltered_rhos_0497p00)
                rhos_0560p00_box = ma.masked_outside(rhos_0560p00_box,mean_unfiltered_rhos_0560p00\
                    -1.5*std_unfiltered_rhos_0560p00\
                    , mean_unfiltered_rhos_0560p00\
                    +1.5*std_unfiltered_rhos_0560p00)
                rhos_0664p00_box = ma.masked_outside(rhos_0664p00_box,mean_unfiltered_rhos_0664p00\
                    -1.5*std_unfiltered_rhos_0664p00\
                    , mean_unfiltered_rhos_0664p00\
                    +1.5*std_unfiltered_rhos_0664p00)

                mean_filtered_rhos_0444p00 = rhos_0444p00_box.mean()
                mean_filtered_rhos_0497p00 = rhos_0497p00_box.mean()
                mean_filtered_rhos_0560p00 = rhos_0560p00_box.mean()
                mean_filtered_rhos_0664p00 = rhos_0664p00_box.mean()

                std_filtered_rhos_0444p00 = rhos_0444p00_box.std()
                std_filtered_rhos_0497p00 = rhos_0497p00_box.std()
                std_filtered_rhos_0560p00 = rhos_0560p00_box.std()
                std_filtered_rhos_0664p00 = rhos_0664p00_box.std()

                CV_filtered_rhos_0444p00 = std_filtered_rhos_0444p00/mean_filtered_rhos_0444p00
                CV_filtered_rhos_0497p00 = std_filtered_rhos_0497p00/mean_filtered_rhos_0497p00
                CV_filtered_rhos_0560p00 = std_filtered_rhos_0560p00/mean_filtered_rhos_0560p00
                CV_filtered_rhos_0664p00 = std_filtered_rhos_0664p00/mean_filtered_rhos_0664p00
                
                CVs = [CV_filtered_rhos_0444p00,CV_filtered_rhos_0497p00, CV_filtered_rhos_0560p00]
                print(CVs)
                MedianCV = np.nanmedian(np.abs(CVs))

                print('Median CV={:.4f}'.format(MedianCV))
               
                if MedianCV <= 0.15:
                    # Rrs 0444p00
                    print('444.0')
                    if NGP_rhos_0444p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0444p00={:.0f}'.format(NGP_rhos_0444p00))
                    else:
                        matchups_Lwn_0444p00_fq_sat_pa.append(mean_filtered_rhos_0444p00*F0_0444p00/np.pi)
                        matchups_Lwn_0444p00_fq_ins_pa.append(Lwn_fq_0442p00)
                        matchups_Lwn_0444p00_fq_ins_pa_station.append(station_name)
                        matchups_Lwn_0444p00_fq_sat_pa_time.append(sat_time)
                    # Rrs 0497p00
                    print('497.0')
                    if NGP_rhos_0497p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0497p00={:.0f}'.format(NGP_rhos_0497p00))
                    else:
                        matchups_Lwn_0497p00_fq_sat_pa.append(mean_filtered_rhos_0497p00*F0_0497p00/np.pi)
                        matchups_Lwn_0497p00_fq_ins_pa.append(Lwn_fq_0491p00)
                        matchups_Lwn_0497p00_fq_ins_pa_station.append(station_name)
                        matchups_Lwn_0497p00_fq_sat_pa_time.append(sat_time)
                    # Rrs 0560p00
                    print('560.0')
                    if NGP_rhos_0560p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0560p00={:.0f}'.format(NGP_rhos_0560p00))
                    else:
                        matchups_Lwn_0560p00_fq_sat_pa.append(mean_filtered_rhos_0560p00*F0_0560p00/np.pi)
                        matchups_Lwn_0560p00_fq_ins_pa.append(Lwn_fq_00551p0)
                        matchups_Lwn_0560p00_fq_ins_pa_station.append(station_name)
                        matchups_Lwn_0560p00_fq_sat_pa_time.append(sat_time)
                    # Rrs 0664p00
                    print('664.0')
                    if NGP_rhos_0664p00<NTP/2+1:
                        print('Exceeded: NGP_rhos_0664p00={:.0f}'.format(NGP_rhos_0664p00))
                    else:
                        matchups_Lwn_0664p00_fq_sat_pa.append(mean_filtered_rhos_0664p00*F0_0664p00/np.pi)
                        matchups_Lwn_0664p00_fq_ins_pa.append(Lwn_fq_0668p00)
                        matchups_Lwn_0664p00_fq_ins_pa_station.append(station_name)
                        matchups_Lwn_0664p00_fq_sat_pa_time.append(sat_time)
                else:
                    print('Median CV exceeds criteria: Median[CV]={:.4f}'.format(MedianCV))
            else:
                print('Angles exceeds criteria: sza={:.2f}'.format(sza)+'; vza={:.2f}'.format(vza)+\
                    '; OR mean of rhot_1614={:.4f}'.format(rhot_1614p00_box.mean())+'>0.01!\n'+\
                    '; OR NGP={:.0f}'.format(NGP)+'< NTP/2+1={:.0f}'.format(NTP/2+1)+'!')
#                print('Angles exceeds criteria: sza='+str(sza)+'; vza='+str(vza)+'; OR NGP='+str(NGP)+'< NTP/2+1='+str(NTP/2+1)+'!')
        else:
            print('Not matchups for '+folder_name) 

        # Vanhellemot
        print('--Vanhellemot')

        # in situ from aeronet_i.log
        # in_situ_lat, in_situ_lon = get_lat_lon_ins(station_name)
        
        # r, c = create_extract.find_row_column_from_lat_lon(lat,lon,in_situ_lat,in_situ_lon)

        # in situ data        
        # with open(os.path.join(path_data,sensor_name,file_name[2:-1]), 'r') as file:
        #     ins_time = []
        #     # Date
        #     line_num_date = 3-1
        #     line_num_time = 4-1
        #     ins_all_lines = file.readlines()
        #     line_date = ins_all_lines[line_num_date]   
        #     line_time = ins_all_lines[line_num_time] 
            
        #     str_list_date = line_date[:-1].replace('=',',').split(',')
        #     str_list_time = line_time[:-1].replace('=',',').split(',')
            
        #     n_ins = len(str_list_date)-1
        #     for i in range(n_ins):
        #         date_and_time_str = str_list_date[1:][i]+' '+str_list_time[1:][i]
        #         date_format = "%d/%m/%Y %H:%M:%S"
        #         ins_time.append(datetime.strptime(date_and_time_str, date_format))

        # delta_time = 2.0# float in hours   
        # ins_time = np.array(ins_time)
        # time_diff = ins_time - sat_time
        # dt_hour = [i.total_seconds()/(60*60) for i in time_diff] # time diffence between in situ measurements and sat in hours
        # idx_min = np.argmin(np.abs(dt_hour))
        # matchup_idx_vec = np.abs(dt_hour) <= delta_time 

   #      # from aeronet_i.log
   #         101  Lwn(412)=0.5702557894736842
   # 102  Lwn(441)=0.7611690526315789
   # 103  Lwn(489)=0.9938783157894736
   # 104  Lwn(530)=0.7793456842105263
   # 105  Lwn(551)=0.5839507894736842
   # 106  Lwn(668)=0.06526310526315791
   # 107  Lwn(869)=-0.004366210526315784
   # 108  Lwn(1018)=-0.0016966842105263127
        # 109  Lwn_f/Q(412)=0.5564285789473684
        # 110  Lwn_f/Q(441)=0.7373514736842104
        # 111  Lwn_f/Q(489)=0.9496013157894736
        # 112  Lwn_f/Q(530)=0.7502892631578948
        # 113  Lwn_f/Q(551)=0.5632072105263157
        # 114  Lwn_f/Q(668)=0.06371473684210528
        # 115  Lwn_f/Q(869)=-0.004366210526315784
        # 116  Lwn_f/Q(1018)=-0.0016966842105263127

        ins_file_path = os.path.join(path_data,sensor_name,folder_name)+'/aeronet_i.log'

        with open(ins_file_path, 'r') as file:
            ins_all_lines = file.readlines()
            Lwn_fq_0442p00 = float(ins_all_lines[102-1][:-1].split('=')[1]) 
            Lwn_fq_0491p00 = float(ins_all_lines[103-1][:-1].split('=')[1]) 
            Lwn_fq_00551p0 = float(ins_all_lines[105-1][:-1].split('=')[1]) 
            Lwn_fq_0668p00 = float(ins_all_lines[106-1][:-1].split('=')[1])   

        # from l2_dsf_new.log
        # 90  kernel_rho_t=0.136428374656
        # 96    kernel_wave_s=444,497,560,664,704,740,782,835,865,1614,2202
        #.      0             1   2   3   4   5   6   7   8   9   10   11  
        # 97  kernel_rho_s=0.0171095194906,...
        # 98  kernel_sdev_s=0.000584191352004,...

        sat_file_path = os.path.join(path_data,sensor_name,folder_name)+'/l2_dsf_new.log'  

        with open(sat_file_path, 'r') as file:
            ins_all_lines = file.readlines()
            mean_filtered_rhot_1614p00 = float(ins_all_lines[90-1][:-1].replace('=',',').split(',')[10]) 
            mean_filtered_rhos_0444p00 = float(ins_all_lines[97-1][:-1].replace('=',',').split(',')[1]) 
            mean_filtered_rhos_0497p00 = float(ins_all_lines[97-1][:-1].replace('=',',').split(',')[2])
            mean_filtered_rhos_0560p00 = float(ins_all_lines[97-1][:-1].replace('=',',').split(',')[3])
            mean_filtered_rhos_0664p00 = float(ins_all_lines[97-1][:-1].replace('=',',').split(',')[4])

        if sza<=75 and vza<=60 and mean_filtered_rhot_1614p00<=0.01:
            matchups_Lwn_0444p00_fq_sat_va.append(mean_filtered_rhos_0444p00*F0_0444p00/np.pi)
            matchups_Lwn_0444p00_fq_ins_va.append(Lwn_fq_0442p00)
            matchups_Lwn_0444p00_fq_ins_va_station.append(station_name)
            matchups_Lwn_0444p00_fq_sat_va_time.append(sat_time)

            matchups_Lwn_0497p00_fq_sat_va.append(mean_filtered_rhos_0497p00*F0_0497p00/np.pi)
            matchups_Lwn_0497p00_fq_ins_va.append(Lwn_fq_0491p00)
            matchups_Lwn_0497p00_fq_ins_va_station.append(station_name)
            matchups_Lwn_0497p00_fq_sat_va_time.append(sat_time)

            matchups_Lwn_0560p00_fq_sat_va.append(mean_filtered_rhos_0560p00*F0_0560p00/np.pi)
            matchups_Lwn_0560p00_fq_ins_va.append(Lwn_fq_00551p0)
            matchups_Lwn_0560p00_fq_ins_va_station.append(station_name)
            matchups_Lwn_0560p00_fq_sat_va_time.append(sat_time)

            matchups_Lwn_0664p00_fq_sat_va.append(mean_filtered_rhos_0664p00*F0_0664p00/np.pi)
            matchups_Lwn_0664p00_fq_ins_va.append(Lwn_fq_0668p00)
            matchups_Lwn_0664p00_fq_ins_va_station.append(station_name)
            matchups_Lwn_0664p00_fq_sat_va_time.append(sat_time)
                  

#%% plots  
prot_name = 'ba' 
sensor_name = 'MSI'
rmse_val_0444p00_ba, mean_abs_rel_diff_0444p00_ba, mean_rel_diff_0444p00_ba, r_sqr_0444p00_ba = plot_scatter(
    matchups_Lwn_0444p00_fq_ins_ba,matchups_Lwn_0444p00_fq_sat_ba,'444.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0444p00_fq_ins_ba_station,min_val=-0.50,max_val=3.50) 
rmse_val_0497p00_ba, mean_abs_rel_diff_0497p00_ba, mean_rel_diff_0497p00_ba, r_sqr_0497p00_ba = plot_scatter(
    matchups_Lwn_0497p00_fq_ins_ba,matchups_Lwn_0497p00_fq_sat_ba,'497.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0497p00_fq_ins_ba_station,min_val= 0.00,max_val=4.00) 
rmse_val_0560p00_ba, mean_abs_rel_diff_0560p00_ba, mean_rel_diff_0560p00_ba, r_sqr_0560p00_ba = plot_scatter(
    matchups_Lwn_0560p00_fq_ins_ba,matchups_Lwn_0560p00_fq_sat_ba,'560.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0560p00_fq_ins_ba_station,min_val= 0.00,max_val=4.00) 
rmse_val_0664p00_ba, mean_abs_rel_diff_0664p00_ba, mean_rel_diff_0664p00_ba, r_sqr_0664p00_ba = plot_scatter(
    matchups_Lwn_0664p00_fq_ins_ba,matchups_Lwn_0664p00_fq_sat_ba,'664.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0664p00_fq_ins_ba_station,min_val=-0.20,max_val=1.50) 

#% plots  
prot_name = 'pa' 
sensor_name = 'MSI'
rmse_val_0444p00_pa, mean_abs_rel_diff_0444p00_pa, mean_rel_diff_0444p00_pa, r_sqr_0444p00_pa = plot_scatter(
    matchups_Lwn_0444p00_fq_ins_pa,matchups_Lwn_0444p00_fq_sat_pa,'444.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0444p00_fq_ins_pa_station,min_val=-0.50,max_val=3.50) 
rmse_val_0497p00_pa, mean_abs_rel_diff_0497p00_pa, mean_rel_diff_0497p00_pa, r_sqr_0497p00_pa = plot_scatter(
    matchups_Lwn_0497p00_fq_ins_pa,matchups_Lwn_0497p00_fq_sat_pa,'497.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0497p00_fq_ins_pa_station,min_val= 0.00,max_val=4.00) 
rmse_val_0560p00_pa, mean_abs_rel_diff_0560p00_pa, mean_rel_diff_0560p00_pa, r_sqr_0560p00_pa = plot_scatter(
    matchups_Lwn_0560p00_fq_ins_pa,matchups_Lwn_0560p00_fq_sat_pa,'560.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0560p00_fq_ins_pa_station,min_val= 0.00,max_val=4.00) 
rmse_val_0664p00_pa, mean_abs_rel_diff_0664p00_pa, mean_rel_diff_0664p00_pa, r_sqr_0664p00_pa = plot_scatter(
    matchups_Lwn_0664p00_fq_ins_pa,matchups_Lwn_0664p00_fq_sat_pa,'664.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0664p00_fq_ins_pa_station,min_val=-0.20,max_val=1.50) 

#% plots  
prot_name = 'va' 
sensor_name = 'MSI'
rmse_val_0444p00_va, mean_abs_rel_diff_0444p00_va, mean_rel_diff_0444p00_va, r_sqr_0444p00_va = plot_scatter(
    matchups_Lwn_0444p00_fq_ins_va,matchups_Lwn_0444p00_fq_sat_va,'444.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0444p00_fq_ins_va_station,min_val=-0.50,max_val=3.50) 
rmse_val_0497p00_va, mean_abs_rel_diff_0497p00_va, mean_rel_diff_0497p00_va, r_sqr_0497p00_va = plot_scatter(
    matchups_Lwn_0497p00_fq_ins_va,matchups_Lwn_0497p00_fq_sat_va,'497.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0497p00_fq_ins_va_station,min_val= 0.00,max_val=4.00) 
rmse_val_0560p00_va, mean_abs_rel_diff_0560p00_va, mean_rel_diff_0560p00_va, r_sqr_0560p00_va = plot_scatter(
    matchups_Lwn_0560p00_fq_ins_va,matchups_Lwn_0560p00_fq_sat_va,'560.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0560p00_fq_ins_va_station,min_val= 0.00,max_val=4.00) 
rmse_val_0664p00_va, mean_abs_rel_diff_0664p00_va, mean_rel_diff_0664p00_va, r_sqr_0664p00_va = plot_scatter(
    matchups_Lwn_0664p00_fq_ins_va,matchups_Lwn_0664p00_fq_sat_va,'664.0',path_out,prot_name,sensor_name,\
    matchups_Lwn_0664p00_fq_ins_va_station,min_val=-0.20,max_val=1.50) 


#%%
# rmse
rmse_ba = [rmse_val_0444p00_ba,rmse_val_0497p00_ba,rmse_val_0560p00_ba,rmse_val_0664p00_ba]
rmse_pa = [rmse_val_0444p00_pa,rmse_val_0497p00_pa,rmse_val_0560p00_pa,rmse_val_0664p00_pa]
rmse_va = [rmse_val_0444p00_va,rmse_val_0497p00_va,rmse_val_0560p00_va,rmse_val_0664p00_va]
wv = [444.0,497.0,560.0,664.0]
plt.figure()
plt.plot(wv,rmse_ba,'-o')
plt.plot(wv,rmse_pa,'-o')
plt.plot(wv,rmse_va,'-o')
plt.xlabel('Wavelength [nm]',fontsize=12)
plt.ylabel('rmse',fontsize=12)
plt.legend(['BW06','IPK19','V19'])
plt.show()

ofname = 'S2A_rmse.pdf'
ofname = os.path.join(path_out,'source',ofname)   
plt.savefig(ofname, dpi=300)

# mean_abs_rel_diff
mean_abs_rel_diff_ba = [mean_abs_rel_diff_0444p00_ba,mean_abs_rel_diff_0497p00_ba,\
    mean_abs_rel_diff_0560p00_ba,mean_abs_rel_diff_0664p00_ba]
mean_abs_rel_diff_pa = [mean_abs_rel_diff_0444p00_pa,mean_abs_rel_diff_0497p00_pa,\
    mean_abs_rel_diff_0560p00_pa,mean_abs_rel_diff_0664p00_pa]
mean_abs_rel_diff_va = [mean_abs_rel_diff_0444p00_va,mean_abs_rel_diff_0497p00_va,\
    mean_abs_rel_diff_0560p00_va,mean_abs_rel_diff_0664p00_va]    
wv = [444.0,497.0,560.0,664.0]
plt.figure()
plt.plot(wv,mean_abs_rel_diff_ba,'-o')
plt.plot(wv,mean_abs_rel_diff_pa,'-o')
plt.plot(wv,mean_abs_rel_diff_va,'-o')
plt.xlabel('Wavelength [nm]',fontsize=12)
plt.ylabel('MAPD [%]',fontsize=12)
# plt.legend(['Bailey and Werdell','Pahlevan','Vanhellemont'])
plt.show()

ofname = 'S2A_mean_abs_rel_diff.pdf'
ofname = os.path.join(path_out,'source',ofname)   
plt.savefig(ofname, dpi=300)

# mean_rel_diff
mean_rel_diff_ba = [mean_rel_diff_0444p00_ba,mean_rel_diff_0497p00_ba,\
    mean_rel_diff_0560p00_ba,mean_rel_diff_0664p00_ba]
mean_rel_diff_pa = [mean_rel_diff_0444p00_pa,mean_rel_diff_0497p00_pa,\
    mean_rel_diff_0560p00_pa,mean_rel_diff_0664p00_pa]
mean_rel_diff_va = [mean_rel_diff_0444p00_va,mean_rel_diff_0497p00_va,\
    mean_rel_diff_0560p00_va,mean_rel_diff_0664p00_va]    
wv = [444.0,497.0,560.0,664.0]
plt.figure()
plt.plot(wv,mean_rel_diff_ba,'-o')
plt.plot(wv,mean_rel_diff_pa,'-o')
plt.plot(wv,mean_rel_diff_va,'-o')
plt.xlabel('Wavelength [nm]',fontsize=12)
plt.ylabel('MPD [%]',fontsize=12)
# plt.legend(['Bailey and Werdell','Pahlevan','Vanhellemont'])
plt.show()    

ofname = 'S2A_mean_rel_diff.pdf'
ofname = os.path.join(path_out,'source',ofname)   
plt.savefig(ofname, dpi=300)

# r_sqr
r_sqr_ba = [r_sqr_0444p00_ba,r_sqr_0497p00_ba,\
    r_sqr_0560p00_ba,r_sqr_0664p00_ba]
r_sqr_pa = [r_sqr_0444p00_pa,r_sqr_0497p00_pa,\
    r_sqr_0560p00_pa,r_sqr_0664p00_pa]
r_sqr_va = [r_sqr_0444p00_va,r_sqr_0497p00_va,\
    r_sqr_0560p00_va,r_sqr_0664p00_va]    
wv = [444.0,497.0,560.0,664.0]
plt.figure()
plt.plot(wv,r_sqr_ba,'-o')
plt.plot(wv,r_sqr_pa,'-o')
plt.plot(wv,r_sqr_va,'-o')
plt.xlabel('Wavelength [nm]',fontsize=12)
plt.ylabel('$r^2$',fontsize=12)
# plt.legend(['Bailey and Werdell','Pahlevan','Vanhellemont'])
plt.show()    

ofname = 'S2A_r_sqr.pdf'
ofname = os.path.join(path_out,'source',ofname)   
plt.savefig(ofname, dpi=300)  

#            print(ins_time)
#%% all approaches comparison
notation_flag = 0 # to display percentage difference in the plot
plot_all_methods('444.0',notation_flag,path_out,min_val=-2.00,max_val=4.0)
plot_all_methods('497.0',notation_flag,path_out,min_val=-1.00,max_val=8.0)
plot_all_methods('560.0',notation_flag,path_out,min_val=-1.00,max_val=6.0)
plot_all_methods('664.0',notation_flag,path_out,min_val=-0.60,max_val=2.0)                 
##%%
#if __name__ == '__main__':
#    main()  
#%% aeronet_daily.log example
'''
     1	## Log generated at 2018-03-28 15:27:15
     2	##
     3	Date(dd:mm:yyyy)=01/09/2015,01/09/2015,01/09/2015,01/09/2015,01/09/2015,01/09/2015
     4	Time(hh:mm:ss)=15:06:52,15:27:44,16:00:27,16:27:44,18:00:19,18:27:50
     5	Julian_Day=244.629769,244.644259,244.666979,244.685926,244.750220,244.769329
     6	Instrument_Number=601,601,601,601,601,601
     7	Solar_Zenith(413)=40.163668,37.676782,34.660258,33.163003,36.083570,39.043058
     8	Solar_Zenith(442)=40.234223,37.737249,34.700623,33.182947,36.032393,38.976847
     9	Solar_Zenith(491)=40.127415,37.645747,34.639613,33.153490,36.110066,39.077268
    10	Solar_Zenith(530)=40.197846,37.706063,34.681005,33.173223,36.058715,39.010923
    11	Solar_Zenith(551)=40.093353,37.616609,34.620275,33.143496,36.136651,39.109531
    12	Solar_Zenith(668)=40.268517,37.766672,34.721566,33.193392,36.007701,38.944841
    13	Solar_Zenith(870)=40.305017,37.798008,34.741368,33.203329,35.983089,38.910902
    14	Solar_Zenith(1018)=40.339426,37.827572,34.762506,33.213999,35.957027,38.879024
    15	Solar_Azimuth(413)=137.188165,144.394907,157.156374,168.970892,209.559119,219.631430
    16	Solar_Azimuth(442)=137.007913,144.195609,156.927990,168.724882,209.344893,219.442552
    17	Solar_Azimuth(491)=137.281212,144.497775,157.274198,169.090259,209.669284,219.728534
    18	Solar_Azimuth(530)=137.100709,144.298214,157.038666,168.844129,209.455314,219.539915
    19	Solar_Azimuth(551)=137.368905,144.594716,157.385197,169.217149,209.779317,219.819805
    20	Solar_Azimuth(668)=136.920694,144.099163,156.810510,168.598248,209.240848,219.350793
    21	Solar_Azimuth(870)=136.828149,143.996820,156.700048,168.479124,209.136686,219.253171
    22	Solar_Azimuth(1018)=136.741166,143.900622,156.582797,168.352622,209.025887,219.161168
    23	Lt_Mean(413)=0.465098,0.482106,0.488623,0.511513,0.459694,0.433784
    24	Lt_Mean(442)=0.494768,0.525039,0.530420,0.530645,0.480305,0.443531
    25	Lt_Mean(491)=0.642081,0.682583,0.669030,0.671645,0.584301,0.538172
    26	Lt_Mean(530)=0.745702,0.791460,0.819316,0.779654,0.715300,0.665067
    27	Lt_Mean(551)=0.793005,0.842509,0.861372,0.833765,0.770530,0.712932
    28	Lt_Mean(668)=0.310587,0.325341,0.342669,0.336321,0.328487,0.307327
    29	Lt_Mean(870)=0.050783,0.054434,0.054807,0.054890,0.055056,0.051281
    30	Lt_Mean(1018)=0.022270,0.021838,0.023671,0.022270,0.023456,0.021838
    31	Lt_Stddev(413)=0.020373,0.025189,0.032405,0.015095,0.015643,0.021645
    32	Lt_Stddev(442)=0.022015,0.025665,0.027148,0.018171,0.014282,0.023081
    33	Lt_Stddev(491)=0.037523,0.026627,0.021393,0.015279,0.022538,0.012786
    34	Lt_Stddev(530)=0.031832,0.020943,0.020058,0.014950,0.018372,0.017785
    35	Lt_Stddev(551)=0.018922,0.026039,0.014683,0.008699,0.012859,0.018426
    36	Lt_Stddev(668)=0.012913,0.009368,0.010960,0.005154,0.011668,0.008496
    37	Lt_Stddev(870)=0.004529,0.006424,0.002685,0.001411,0.005900,0.001752
    38	Lt_Stddev(1018)=0.006010,0.001551,0.006532,0.001798,0.001633,0.002468
    39	Lt_Min_rel(413)=0.442686,0.456197,0.445865,0.491961,0.441096,0.410100
    40	Lt_Min_rel(442)=0.473129,0.487704,0.498355,0.507885,0.465842,0.415950
    41	Lt_Min_rel(491)=0.608713,0.652306,0.639228,0.648343,0.552439,0.520339
    42	Lt_Min_rel(530)=0.715300,0.763913,0.798636,0.758511,0.685207,0.642767
    43	Lt_Min_rel(551)=0.762797,0.812662,0.842654,0.821696,0.755209,0.691251
    44	Lt_Min_rel(668)=0.293087,0.313389,0.328544,0.329688,0.311673,0.297376
    45	Lt_Min_rel(870)=0.044601,0.047298,0.051447,0.053106,0.046675,0.049165
    46	Lt_Min_rel(1018)=0.018064,0.019951,0.019681,0.019951,0.021569,0.019142
    47	Li_Mean_val(413)=12.032784,12.042321,12.564748,13.282159,12.592830,11.845748
    48	Li_Mean_val(442)=11.550181,11.537101,11.990797,12.745337,12.124588,11.338282
    49	Li_Mean_val(491)=10.457236,10.347065,10.809676,11.546261,10.939926,10.170580
    50	Li_Mean_val(530)=8.867562,8.717351,9.088505,9.773711,9.328481,8.656650
    51	Li_Mean_val(551)=8.192871,8.034361,8.425576,9.017457,8.583604,7.930053
    52	Li_Mean_val(668)=4.704839,4.578263,4.746586,5.148235,5.079037,4.675864
    53	Li_Mean_val(870)=1.627065,1.563449,1.605491,1.740469,1.801873,1.647118
    54	Li_Mean_val(1018)=0.779703,0.756517,0.778445,0.846206,0.892398,0.815830
    55	Li_Stddev(413)=0.008411,0.005506,0.006618,0.006929,0.005582,0.006018
    56	Li_Stddev(442)=0.005826,0.023312,0.002966,0.004887,0.006175,0.004887
    57	Li_Stddev(491)=0.003203,0.011385,0.003746,0.000458,0.012133,0.002858
    58	Li_Stddev(530)=0.005401,0.003119,0.003564,0.003647,0.004250,0.002480
    59	Li_Stddev(551)=0.002921,0.002736,0.007961,0.003009,0.003980,0.002606
    60	Li_Stddev(668)=0.001981,0.001838,0.001513,0.002641,0.002641,0.007616
    61	Li_Stddev(870)=0.001571,0.001571,0.001334,0.001660,0.002156,0.001808
    62	Li_Stddev(1018)=0.002350,0.001868,0.002179,0.002548,0.002350,0.002350
    63	AOT(413)=0.462956,0.410602,0.409889,0.437301,0.429415,0.435288
    64	AOT(442)=0.419358,0.371529,0.370393,0.394778,0.388857,0.395089
    65	AOT(491)=0.353782,0.314324,0.313,0.333277,0.328143,0.334256
    66	AOT(530)=0.309901,0.274592,0.273279,0.291137,0.288414,0.294774
    67	AOT(551)=0.290502,0.257771,0.25758,0.27393,0.271369,0.277882
    68	AOT(668)=0.196823,0.174629,0.173951,0.185062,0.183793,0.189186
    69	AOT(870)=0.116534,0.105566,0.1061,0.112092,0.111693,0.115177
    70	AOT(1018)=0.07277,0.063678,0.063307,0.066494,0.065544,0.068675
    71	OOT(413)=0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
    72	OOT(442)=0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
    73	OOT(491)=0.007003,0.007005,0.007007,0.007008,0.007005,0.007003
    74	OOT(530)=0.022595,0.022602,0.022608,0.022610,0.022601,0.022593
    75	OOT(551)=0.028713,0.028721,0.028729,0.028732,0.028720,0.028710
    76	OOT(668)=0.016055,0.016059,0.016064,0.016065,0.016059,0.016053
    77	OOT(870)=0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
    78	OOT(1018)=0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
    79	ROT(413)=0.315517,0.315519,0.315522,0.315525,0.315533,0.315519
    80	ROT(442)=0.238473,0.238474,0.238477,0.238479,0.238485,0.238474
    81	ROT(491)=0.154185,0.154186,0.154188,0.154189,0.154193,0.154186
    82	ROT(530)=0.112253,0.112254,0.112255,0.112256,0.112259,0.112254
    83	ROT(551)=0.096378,0.096378,0.096379,0.096380,0.096383,0.096378
    84	ROT(668)=0.044032,0.044032,0.044032,0.044033,0.044034,0.044032
    85	ROT(870)=0.015115,0.015115,0.015115,0.015115,0.015116,0.015115
    86	ROT(1018)=0.008046,0.008046,0.008046,0.008046,0.008046,0.008046
    87	Lw(413)=0.123655,0.137301,0.119050,0.147138,0.110366,0.098806
    88	Lw(442)=0.166894,0.182187,0.186469,0.176998,0.147410,0.117991
    89	Lw(491)=0.331455,0.378303,0.358063,0.348586,0.265120,0.253066
    90	Lw(530)=0.480190,0.533067,0.562240,0.504772,0.440210,0.415279
    91	Lw(551)=0.545575,0.599903,0.623501,0.587590,0.529775,0.482857
    92	Lw(668)=0.168346,0.192151,0.205083,0.196032,0.178281,0.174499
    93	Lw(870)=0.001462,0.005896,0.009687,0.007921,-0.000648,0.005880
    94	Lw(1018)=-0.002609,-0.000083,-0.000566,-0.002018,-0.001869,-0.002297
    95	Lw_Q(413)=0.113825,0.126998,0.109736,0.136188,0.100809,0.089950
    96	Lw_Q(442)=0.152354,0.167110,0.170530,0.162522,0.133656,0.106613
    97	Lw_Q(491)=0.296426,0.340004,0.320868,0.313676,0.235470,0.223957
    98	Lw_Q(530)=0.424681,0.473946,0.498217,0.449288,0.386473,0.363220
    99	Lw_Q(551)=0.480492,0.531178,0.550094,0.520741,0.462928,0.420319
   100	Lw_Q(668)=0.154055,0.176843,0.187891,0.180488,0.161601,0.157581
   101	Lw_Q(870)=0.001462,0.005896,0.009687,0.007921,-0.000648,0.005880
   102	Lw_Q(1018)=-0.002609,-0.000083,-0.000566,-0.002018,-0.001869,-0.002297
   103	Lwn(413)=0.224969,0.236104,0.194812,0.236671,0.185448,0.174993
   104	Lwn(442)=0.285881,0.295781,0.288577,0.269275,0.233640,0.196600
   105	Lwn(491)=0.532647,0.578646,0.523528,0.501363,0.397166,0.397819
   106	Lwn(530)=0.759344,0.803289,0.810068,0.715048,0.648792,0.641661
   107	Lwn(551)=0.855308,0.897261,0.892521,0.827199,0.776998,0.742645
   108	Lwn(668)=0.246346,0.269323,0.275534,0.258861,0.244163,0.249685
   109	Lwn(870)=0.002020,0.007838,0.012370,0.009940,-0.000842,0.007958
   110	Lwn(1018)=-0.003558,-0.000108,-0.000714,-0.002499,-0.002393,-0.003063
   111	Lwn_f/Q(413)=0.199846,0.211956,0.174424,0.213668,0.163291,0.152634
   112	Lwn_f/Q(442)=0.249731,0.261344,0.254737,0.239803,0.202877,0.168915
   113	Lwn_f/Q(491)=0.447051,0.492275,0.445930,0.431374,0.332136,0.328300
   114	Lwn_f/Q(530)=0.623785,0.669912,0.676658,0.604037,0.531507,0.518073
   115	Lwn_f/Q(551)=0.696629,0.742246,0.739626,0.693380,0.630837,0.593902
   116	Lwn_f/Q(668)=0.213507,0.236848,0.241637,0.229630,0.209553,0.211489
   117	Lwn_f/Q(870)=0.002020,0.007838,0.012370,0.009940,-0.000842,0.007958
   118	Lwn_f/Q(1018)=-0.003558,-0.000108,-0.000714,-0.002499,-0.002393,-0.003063
   119	Pressure=1015.398385,1015.404937,1015.415209,1015.423776,1015.452206,1015.405284
   120	Wind_Speed=2.470539,2.406795,2.306852,2.223507,1.942060,1.977730
   121	Chlorophyll-a=6.732552,6.154193,7.999881,7.332512,11.758411,10.220790
   122	Sea_Surface_Reflectance=0.026513,0.026481,0.026010,0.025961,0.026263,0.026279
   123	Ozone=303.586432,303.578366,303.565719,303.555173,303.519384,303.508747
   124	ExactWavelength-413=413.200000,413.200000,413.200000,413.200000,413.200000,413.200000
   125	ExactWavelength-442=442.100000,442.100000,442.100000,442.100000,442.100000,442.100000
   126	ExactWavelength-491=491.500000,491.500000,491.500000,491.500000,491.500000,491.500000
   127	ExactWavelength-530=531.100000,531.100000,531.100000,531.100000,531.100000,531.100000
   128	ExactWavelength-551=551.300000,551.300000,551.300000,551.300000,551.300000,551.300000
   129	ExactWavelength-668=668.400000,668.400000,668.400000,668.400000,668.400000,668.400000
   130	ExactWavelength-870=870.800000,870.800000,870.800000,870.800000,870.800000,870.800000
   131	ExactWavelength-1018=1018.400000,1018.400000,1018.400000,1018.400000,1018.400000,1018.400000
   132	Last_Processing_Date(dd:mm:yyyy)=10/03/2017,10/03/2017,10/03/2017,10/03/2017,10/03/2017,10/03/2017
   133	Day=01,01,01,01,01,01
   134	Month=09,09,09,09,09,09
   135	Year=2015,2015,2015,2015,2015,2015
   136	isodate=2015-09-01,2015-09-01,2015-09-01,2015-09-01,2015-09-01,2015-09-01
   137	Hour=15,15,16,16,18,18
   138	Minute=06,27,00,27,00,27
   139	Second=52,44,27,44,19,50
   140	Time_Float=15.114444444444445,15.462222222222222,16.0075,16.46222222222222,18.005277777777778,18.46388888888889
   141	isotime=15:06:52,15:27:44,16:00:27,16:27:44,18:00:19,18:27:50
   142	##
   143	## End log file
'''