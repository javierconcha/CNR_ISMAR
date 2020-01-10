#!/usr/bin/env python3
# coding: utf-8
"""
Created on Fri Jan 10 12:40:47 2020

@author: javier.concha
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
#!/usr/bin/env python
# coding: utf-8
"""
Created on Wed Sep  4 18:38:15 2019
Create list of OLCI WFR files from DataArchive in the virtual machine.
@author: javier.concha
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
import datetime
import subprocess
import sys
import zipfile
import argparse

import common_functions

def contain_location(path_source,in_situ_lat,in_situ_lon):
    ## open netcdf file  
    coordinates_filename = 'geo_coordinates.nc'
    
    filepah = os.path.join(path_source,coordinates_filename)
    nc_f0 = Dataset(filepah,'r')
    
    lat = nc_f0.variables['latitude'][:,:]
    lon = nc_f0.variables['longitude'][:,:]
    
    nc_f0.close()

    
    if in_situ_lat >= lat.min()  and in_situ_lat <= lat.max() and in_situ_lon >= lon.min() and in_situ_lon <= lon.max():
        contain_flag = 1
    else:
        contain_flag = 0
        
    return contain_flag
        
#%%
def main():
    """business logic for when running this module as the primary one!"""
    print('Main Code!')
    
    if sys.platform == 'linux': 
        path_main = '/home/Javier.Concha/Val_Prot/codes/'
        path_source = '/DataArchive/OC/OLCI/sources/'     
    elif sys.platform == 'darwin':
        path_main = '/Users/javier.concha/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'
        path_source = os.path.join(path_main,'data/source/')
    else:
        print('Error: host flag is not either mac or vm')
   
    station_list = ['Venise','Galata_Platform','Gloria','Helsinki_Lighthouse','Gustav_Dalen_Tower']
    parser = argparse.ArgumentParser(description="Create list of OLCI WFR files from DataArchive in the virtual machine.")
    parser.add_argument("-s", "--station" , help="The Aeronet OC station", type=str,choices=station_list)
    args = parser.parse_args()
    
    if args.station:
        station_name  = args.station
    else:
        station_name = 'Venise'
    print('The station name is: '+station_name)    
    
    # open in situ data for specific AOC site
    path = os.path.join(path_main,'netcdf_file')
    
    if station_name == 'Venise':
        filename = 'Venise_20V3_20160101_20190918.nc'
    elif station_name == 'Galata_Platform':
        filename = 'Galata_Platform_20V3_20160101_20190918.nc'
    elif station_name == 'Gloria':
        filename = 'Gloria_20V3_20160101_20190918.nc'
    elif station_name == 'Helsinki_Lighthouse':
        filename = 'Helsinki_Lighthouse_20V3_20160101_20190918.nc'
    elif station_name == 'Gustav_Dalen_Tower':
        filename = 'Gustav_Dalen_Tower_20V3_20160101_20190918.nc'
    
    filename_insitu = os.path.join(path,filename)
    if not os.path.exists(filename_insitu):
        print('File does not exist')
        
    nc_f0 = Dataset(filename_insitu,'r')
    
    Time = nc_f0.variables['Time'][:]
    Julian_day = nc_f0.variables['Julian_day'][:]
    
    nc_f0.close()
    
    day_vec =np.array([float(Time[i].replace(' ',':').split(':')[0]) for i in range(0,len(Time))])
    month_vec =np.array([float(Time[i].replace(' ',':').split(':')[1]) for i in range(0,len(Time))])
    year_vec =np.array([float(Time[i].replace(' ',':').split(':')[2]) for i in range(0,len(Time))])
    
    doy_vec = np.array([int(float(Julian_day[i])) for i in range(0,len(Time))])
    
    lat_ins, lon_ins = common_functions.get_lat_lon_ins(station_name)
    
    f = open(path_main+'OLCI_list_'+filename.split('.')[0]+'.txt','a+')
    
    cmd = 'mkdir ./temp_'+station_name # create temp folder
    (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
    
    last_day = datetime.datetime(1990,1,1)
    
    # open year/month folder
    for i in range(len(Time)):
        
        date1 = datetime.datetime(int(year_vec[i]),int(month_vec[i]),int(day_vec[i]))
        
        if date1 != last_day:
            print('--------------------------------------------------------')
            last_day = date1
            
            print(date1)
            
            # create list in txt file with file starting with "S3A_OL_2_WFR____"
            cmd = 'ls -1 '+\
                path_source+str(int(year_vec[i]))+'/'+str(int(doy_vec[i]))+'/S3A_OL_2_WFR____*.zip > ./temp_'+station_name+'/temp_list.txt'
    #        print(cmd)
            # New process, connected to the Python interpreter through pipes:
            prog = subprocess.Popen(cmd, shell=True,stderr=subprocess.PIPE)
            out, err = prog.communicate()
            if not err:
                # iterate list
                with open('./temp_'+station_name+'/temp_list.txt','r') as file:
                    for line in file:                    
                        # unzip and adding exception handling
                        try:
                            zip = zipfile.ZipFile(line[:-1])
                            zip.extractall('./temp_'+station_name)
                            zip.close()
                        except IOError as e:
                            print("Unable to copy file. %s" % e)
                        except:
                            print("Unexpected error:", sys.exc_info())
                            
                        if line[:-1].split('.')[-2] == 'SEN3': # if ends in SEN3.zip
                            prod_name = line[:-1].split('.')[-3].split('/')[-1]
                        else: # if ends in .zip
                            prod_name = line[:-1].split('.')[-2].split('/')[-1]    
                        
                        path_source2 = './temp_'+station_name+'/'+prod_name+'.SEN3/'
                        print(path_source2)
                        
                        # check if file include lat lon
                        contain_flag = contain_location(path_source2,lat_ins,lon_ins)    
                        if contain_flag:
                            print('Product contains location!')
                            f.write(prod_name+'\n')
                        else:
                            print('Product DOES NOT contains location!')
                    
                        # delete files
                        cmd = 'rm -r ./temp_'+station_name+'/*.SEN3' # remove .SEN* folder
                        (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
                        cmd = 'rm ./temp_'+station_name+'/temp_list.txt'
                        (ls_status, ls_output) = subprocess.getstatusoutput(cmd)

    cmd = 'rm -r ./temp_'+station_name
    (ls_status, ls_output) = subprocess.getstatusoutput(cmd)

    f.close()  
    cmd = 'cat '+path_main+'OLCI_list_'+filename.split('.')[0]+'.txt|sort|uniq > '+path_main+'OLCI_list_'+filename.split('.')[0]+'_uniq.txt'
    (ls_status, ls_output) = subprocess.getstatusoutput(cmd)
#%%
if __name__ == '__main__':
    main()        
