#!/usr/bin/env python
# coding: utf-8
"""
Created by Marco Bracaglia
Modified by Javier on Augutst 02 2019
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
# from shusget.sh
# -c <lon1,lat1:lon2,lat2>     : Search for products intersecting a rectangular Area of Interst (or Bounding Box)"
#    -C <CSVfile>            : write the list of product results in a specified CSV file. Default file is './products-list.csv'
#  -L <lock folder>        : by default only one instance of dhusget can be executed at a time. This is ensured by the creation
# of a temporary lock folder /Users/javier.concha/dhusget_tmp/lock which is removed a the end of each run.
# For running more than one dhusget instance at a time is sufficient to assign different lock folders
# using the -L option (e.g. '-L foldername') to each dhusget instance;
#%%
import os, sys
import csv
import datetime

main_path = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/'
#%%
def olci_get(date,lat1,lat2,lon1,lon2):
    
    
    if (date-datetime.datetime(2017,11,29)).total_seconds()<0:
        site="codarep.eumetsat.int"
    else:
        site="coda.eumetsat.int"
    year=str(date.year)
    month='0'*(2-len(str(date.month)))+str(date.month)    
    day='0'*(2-len(str(date.day)))+str(date.day)    
    datestart = year+'-'+month+'-'+day
    dateend = datestart
    
    lat1_str = '{:,.2f}'.format(lat1)
    lat2_str = '{:,.2f}'.format(lat2)
    lon1_str = '{:,.2f}'.format(lon1)
    lon2_str = '{:,.2f}'.format(lon2)
    location_str = lon1_str+','+lat1_str+':'+lon2_str+','+lat2_str #"12.0,44.0:14.0,46.00"
    print(location_str)
    
    csv_f = main_path+'csv_file/'+datestart.replace('-','')+"_"+dateend.replace('-','')+'.csv'
    
    print('csv_f:')
    print(csv_f)
    
    os.system('rm -r /Users/javier.concha/dhusget_tmp/')
    
    cmd = './dhusget.sh -u jaconcha -p a1b2c3d4 '+\
    '-d '+site+' -m Sentinel-3 -i OLCI -T OL_2_WFR___ '+\
    '-S '+datestart+'T00:00:00Z -E '+dateend+'T23:59:59Z '+\
    '-c '+location_str+' '\
    "-F "+"'timeliness:Non Time Critical' "\
    '-C '+csv_f
#    '-L /Users/javier.concha/dhusget_tmp/'
    print(cmd)
    
    os.system(cmd)

    file_list = []
#    i = 0
    with open(csv_f) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0][0:3] == 'S3A':
                file_list.append(row[0])
                
    
#            if i==0 or i%2==0:
#                file_olci=row[0]
#                file_list.append(file_olci)
#                i+=1
#    print (file_list)
#    os.system('rm '+csv_f)
    os.system('rm -r /Users/javier.concha/dhusget_tmp/')
    
    

    return file_list    


def olci_get_L1(date):
    if (date-datetime.datetime(2017,11,29)).total_seconds()<0:
            site="codarep.eumetsat.int"
    else:
            site="coda.eumetsat.int"
    year=str(date.year)
    month='0'*(2-len(str(date.month)))+str(date.month)
    day='0'*(2-len(str(date.day)))+str(date.day)
    datestart=year+'-'+month+'-'+day
    dateend=datestart
    #os.system("./dhusget.sh -u javier.concha -p a1b2c3d4 -d coda.eumetsat.int -m Sentinel-3 -i OLCI -T OL_2_WFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00 -F"+' timeliness:"Non Time Critical"'\
    #+" -q /home/Marco.Bracaglia/progetti/VGOCS/OLCI/xml_file/"+datestart.replace(':','')+"_"+dateend.replace(':','')+".xml"+\
    #" -C /home/Marco.Bracaglia/progetti/VGOCS/OLCI/xls_file/"+datestart.replace(':','')+"_"+dateend.replace(':','')+".xls")
    print ("./dhusget.sh -u jaconcha -p a1b2c3d4 -d "+site+" -m Sentinel-3 -i OLCI -T OL_1_EFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00"+" -F'timeliness:"+'"Non Time Critical"'+"'"+" -o product -O . -C pippo.csv")
    os.system("./dhusget.sh -u jaconcha -p a1b2c3d4 -d "+site+" -m Sentinel-3 -i OLCI -T OL_1_EFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00"+" -F'timeliness:"+'"Non Time Critical"'+"'"+" -o product -O /home/Marco.Bracaglia/progetti/VGOCS/OLCI/L1_temp/ -C /home/Marco.Bracaglia/progetti/VGOCS/OLCI/csv_file/"+datestart.replace('-','')+"_"+dateend.replace('-','')+"_L1.csv")

    """os.system("./dhusget.sh -u javier.concha -p a1b2c3d4 -d "+site+" -m Sentinel-3 -i OLCI -T OL_1_EFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00 -F"+"'timeliness:"+\
    '"Non Time Critical"'+"'"+" -C /Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/csv_file/"+datestart.replace('-','')+"_"+dateend.replace('-','')+".csv -o 'all' -O home/Marco.Bracaglia/progetti/VGOCS/OLCI/L1_temp/")"""
    #os.system('rm '+'/home/Marco.Bracaglia/progetti/VGOCS/OLCI/xml_file/'+datestart.replace(':','')+"_"+dateend.replace(':','')+'.xml')
    csv_f='/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts/csv_file/'+datestart.replace('-','')+"_"+dateend.replace('-','')+'_L1.csv'
    file_list=[]
    i=0
    with open(csv_f) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                    if i==0 or i%2==0:
                            file_olci=row[0]
                            file_list.append(file_olci)
                            i+=1
    print (file_list)
    os.system('rm '+csv_f)
    return file_list

