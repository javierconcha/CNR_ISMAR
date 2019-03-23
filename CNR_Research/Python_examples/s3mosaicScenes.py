#!/usr/bin/env python
# -*- coding: ascii -*-

"""
@author: mario.benincasa@artov.isac.cnr.it
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

"""
s3mosaicScenes.py
 - Mosaic/assemble reprojected scenes on the common map
release 1.0 20170426
release 1.0a 20170523
  warnings suppression
release 1.0b 20170801
  remove source_file attribute
release 1.0c 20170901
  netcdf attributes granules changed in source_files
release 1.0d 20170926
  maintain timeliness of products 
release 1.0e 20181116
  patch for the "coverage" band
  no average but sum... and saturate to 1
"""

"""
usage
$ s3mosaicScenes.py $inputScenes -of $dailyFile
"""
"""
test command line:
/store2/OC/O/daily_granules/2017/111/O2017111085933--bal-FR.nc /store2/OC/O/daily_granules/2017/111/O2017111090233--bal-FR.nc /store2/OC/O/daily_granules/2017/111/O2017111104032--bal-FR.nc /store2/OC/O/daily_granules/2017/111/O2017111104332--bal-FR.nc -of pippo.nc -v
"""

release='1.0e'

import warnings
warnings.filterwarnings('ignore')

import argparse, sys, shutil, os, numpy as np
from netCDF4 import Dataset

# Command line parsing
parser= argparse.ArgumentParser(description='Mosaic/assemble reprojected scenes on the common map, rel. %s'%release)
parser.add_argument('iscenes', nargs='+', action='store', help='input scenes')
parser.add_argument("-of", "--ofname",action="store", help="Output file pathname, e.g. Oyyyyjjj--area-res.nc")
parser.add_argument("-v", "--verbose",action="store_true", help="Verbose output")
args=parser.parse_args()

#open all input files for reading, create a list of filenames, associate number to scene
fmblist=dict()
fnameList=list()
nscenes=len(args.iscenes)
num_scene=dict()
for scene in args.iscenes:
    fmblist[scene]=Dataset(scene, 'r')
    fnameList.append(os.path.basename(scene))
for i,scene in enumerate(args.iscenes):
    num_scene[scene]=i
#check if input files do overlap
s0=args.iscenes[0]
latsize=len(fmblist[s0].dimensions['lat'])
lonsize=len(fmblist[s0].dimensions['lon'])
minlat=fmblist[s0].variables['lat'].valid_min
maxlat=fmblist[s0].variables['lat'].valid_max
minlon=fmblist[s0].variables['lon'].valid_min
maxlon=fmblist[s0].variables['lon'].valid_max
varlist=list()
for vname in fmblist[s0].variables.iterkeys():
    varlist.append(vname)
#check if all files are on the same grid
files_on_same_grid=True
for scene in args.iscenes:
    if (latsize != len(fmblist[scene].dimensions['lat']) or \
    lonsize != len(fmblist[scene].dimensions['lon']) or \
    minlat != fmblist[scene].variables['lat'].valid_min or \
    maxlat != fmblist[scene].variables['lat'].valid_max or \
    minlon != fmblist[scene].variables['lon'].valid_min or \
    maxlon != fmblist[scene].variables['lon'].valid_max ):
        files_on_same_grid=False
#check if all files have same variables
files_on_same_vars=True
for scene in args.iscenes:
    svarlist=list()
    for vname in fmblist[scene].variables.iterkeys():
        svarlist.append(vname)
    if cmp(varlist,svarlist)!=0:
        files_on_same_vars=False
#if not same grid or not same variables, terminate
if (not files_on_same_grid):
    if args.verbose:
            print 'Input files are not on same grid'
    sys.exit()
if (not files_on_same_vars):
    if args.verbose:
            print 'Input files do not have the same vars'
    sys.exit()
#timeliness of the mosaic: NT(i.e. DT) if all files are NT, NR (i.e. NRT) if at least one file is NR
timeliness="NT"
for scene in args.iscenes:
    if fmblist[scene].timeliness != "NT":
        timeliness=fmblist[scene].timeliness

###################
#create destination file as a copy of the first file --- then modifiy later
shutil.copyfile(s0, args.ofname)
fmerge=Dataset(args.ofname, 'r+')
#remove unneeded attributes and add new ones:
fmerge.delncattr('start_time')
fmerge.delncattr('stop_time')
fmerge.delncattr('source_file')
fmerge.timeliness=timeliness
if timeliness=="NT":
    fmerge.product_version='v02'
else:
    fmerge.product_version='v02QL'
fmerge.mosaic=sys.argv[0]
fmerge.source_files=', '.join(fnameList)
# 3d matrix to contain any field from all scenes
field3d=np.ma.zeros((latsize,lonsize,nscenes), dtype=np.float)
#cycle on all fields
for var in varlist:
    if var=='lat' or var=='lon':
        continue
    # load the field from all scenes in memory
    for scene in args.iscenes:
        field3d[:,:,num_scene[scene]]=fmblist[scene].variables[var][:,:]
    if var=='coverage':
        #take the *sum* along the scene axis
        appo_var=field3d.sum(axis=2)
        appo_var[appo_var>0]=1
        fmerge.variables[var][:,:]=appo_var
    else:
        #take the *mean* along the scenes axis
        fmerge.variables[var][:,:]=field3d.mean(axis=2)
fmerge.close()
for scene in args.iscenes:
    fmblist[scene].close()
