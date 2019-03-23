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
s3productInfo.py
Reports information on Sentinel3-OLCI L2 granules
rel 1.0 20170321
  final cosmetic modifications
rel 1.0a 20180212 (as per s3reprojectsBands.py rel 2.1c )
  zip files from CODA do not end with .SEN3, while unpacked directory names do
"""
release='1.0a'

import os,sys,argparse,tarfile,zipfile
import snappy,pyproj
import numpy as np
from shutil import rmtree
from pyresample import geometry,utils

# Command line parsing
parser= argparse.ArgumentParser(description='Reports information on Sentinel3-OLCI L2 granules, rel. %s'%release)
parser.add_argument('iprod', action='store', help='input Sentinel3 L2 product (directory or tarfile or zipfile)')
parser.add_argument("-b",  "--bands", action="store_true",help="list bands ")
parser.add_argument("-s",  "--sizes", action="store_true",help="size in pixels")
parser.add_argument("-c",  "--coord", action="store_true",help="coordinates of the box")
parser.add_argument("-a", "--area",   action="store", help="check if granule overlaps with area: MED, BS, BAL, EUR")
parser.add_argument("-t", "--tardir", action="store", help="temp directory where to un-tar or unzip PRODUCT.tar/.zip, default is current dir")
parser.add_argument("-del", "--delete", action="store_true", help="Delete un-tar-ed or unzipped files, default keeps them")
args=parser.parse_args()

#no need to wait all the stuff ... 
if args.area:
    if args.area.upper() not in ('MED', 'BS', 'BAL', 'EUR'):
        print "Area not defined!"
        sys.exit()

cwd=os.getcwd()
istar=iszip=False
    
if args.tardir and os.path.isdir(args.tardir):
    tardir=args.tardir
else:
    tardir=cwd

if args.iprod.endswith('/'):
    args.iprod=args.iprod[:-1]
l2prodname=os.path.basename(args.iprod)
if l2prodname.endswith('.tar') or l2prodname.endswith('.zip'):
    l2prodname=l2prodname[:-4]
#rel 1.0a, zip files from CODA
if not l2prodname.endswith('.SEN3'):   
    l2prodname+='.SEN3'
if l2prodname.startswith('S3A_OL_2_WFR') or l2prodname.startswith('S3A_OL_2_WRR'):
    if os.path.isdir(args.iprod):
        product=args.iprod
    elif tarfile.is_tarfile(args.iprod):
        istar=True
        tar = tarfile.open(args.iprod, mode='r')
        tar.extractall(path=tardir)
        tar.close()
        product=os.path.join(tardir,l2prodname)
    elif zipfile.is_zipfile(args.iprod):
        iszip=True
        zipf=zipfile.ZipFile(args.iprod, mode='r')
        zipf.extractall(path=tardir)
        zipf.close()
        product=os.path.join(tardir,l2prodname)
    else:
        sys.exit(args.iprod+' not a valid product')
else:
    sys.exit(args.iprod+" it is not an OLCI level 2 Water Full/Reduced Resolution product")

p=snappy.ProductIO.readProduct(product)

if args.bands:
    bandnames=p.getBandNames()
    for band in bandnames:
        print band

if args.sizes or args.coord or args.area:    
    w=p.getSceneRasterWidth()
    h=p.getSceneRasterHeight()

if args.sizes:
    print 'Height: %s, Width: %s'%(h,w)

if args.coord or args.area:
    readlats=p.getBand('latitude')
    readlons=p.getBand('longitude')
    lons=np.zeros(w*h, np.float32)
    lats=np.zeros(w*h, np.float32)
    readlons.readPixels(0,0,w,h,lons)
    readlats.readPixels(0,0,w,h,lats)
    lons.shape=h,w
    lats.shape=h,w
    latmin=lats.min()
    latmax=lats.max()
    lonmin=lons.min()
    lonmax=lons.max()
if args.coord:
    print 'Min Lat: %s, Max Lat: %s'%(latmin, latmax)
    print 'Min Lon: %s, Max Lon: %s'%(lonmin, lonmax)
if args.area:
    #definition of the swath object
    swath = geometry.SwathDefinition(lons=lons, lats=lats)
    #definition of the area  
    if args.area.upper()=='MED':
        w_bound_deg=-6.0
        e_bound_deg=36.5
        s_bound_deg=30
        n_bound_deg=46
    elif args.area.upper()=='BS':
        w_bound_deg=26.5
        e_bound_deg=42
        s_bound_deg=40
        n_bound_deg=48
    elif args.area.upper()=='BAL':
        w_bound_deg=9.25
        e_bound_deg=30.25
        s_bound_deg=53.25
        n_bound_deg=65.85
    elif args.area.upper()=='EUR':
        w_bound_deg=-30.0
        e_bound_deg=42.0
        s_bound_deg=20.0
        n_bound_deg=65.85
    else:
        if args.verbose:
            print "Area not defined! REALLY!"
        sys.exit()    
    area_id=args.area.lower()
    area_name=args.area.lower()
    proj_id=args.area.lower()
    proj4_args=('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +units=m')
    prj = pyproj.Proj(proj4_args)
    # X_bound_deg is bound in degree, X_bound_m is meter on map
    [w_bound_m,e_bound_m],[s_bound_m,n_bound_m]=prj([w_bound_deg,e_bound_deg],[s_bound_deg,n_bound_deg])
    area_extent=(w_bound_m, s_bound_m, e_bound_m, n_bound_m)
    xsize=ysize=1000  # whatever numbers
    area = utils.get_area_def(area_id, area_name, proj_id, proj4_args, xsize, ysize, area_extent)    
    #test if swath and area overlap
    if swath.overlaps(area):
        overlap_fraction=swath.overlap_rate(area)
        print 'Granule overlaps with', args.area, 'area'
        print '### overlap fraction is', overlap_fraction
    else:
        print 'Granule DOES NOT overlap with', args.area, 'area'

p.dispose()

if (istar or iszip) and args.delete:
    rmtree(product)
