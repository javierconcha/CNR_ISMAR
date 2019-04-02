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
s3reprojectBands.py
 - reproject OLCI bands on equirectangular map
release 1.0 201611xx
release 1.1 20170208
  some rewriting
release 1.1a 20170321
  added new flags CLOUD_AMBIGUOUS and CLOUD_MARGIN, commented out by now
  added -od --outdir switch to force an output directory other than cwd
release 1.2 20170522
  CHL_MEDOC production
  coeff=[0.11384242,-4.2663530,6.8988439,-6.8106774,2.0103385]
  r=log10(max([rrs442_5,rrs490,rrs510])/rrs560)
  chlcase1=10^[coeff(0)+coeff(1)r+coeff(2)r^2+coeff(3)r^3+coeff(4)r^4]
  where rrs442_5=Oa03, rrs_490=Oa04, rrs510=Oa05, rrs560=Oa06
release 2.0 20170731
  aimed at OLCI production under CMEMS  
  major rewrite ...  
  use of s3olciprocConf.yaml
  allow mgmt of different flags for different products 
    (see Product Notice EUM/OPS-SEN3/DOC/17/928907 S3A.PN.OLCI-L2M.01 
     issue date 20170705)
  removal of multiple output file production (use s3splitBands.py instead)
  removal of graphics production (use ncplotmb.py instead)
release 2.1 20170901
  netcdf attributes fix
  *** regional algorithm for BS!!!
release 2.1a 20170914
  basin mask has to be flipped upside down (np.flipud(land_mask))
  because point 0,0 is lower,left
release 2.1b 20170926
  timeliness of products:
  product_version (v02 and v02QL) and timeliness (NR and NT) attributes 
release 2.1c 20180113
  zip files from CODA do not end with .SEN3, while unpacked directory names do
release 2.1c 20180316 -> no change in versioning!!
  netcdf attribute netcdf_versione changed to netcdf_version  (trailing e)
release 2.1d 20181015
  line 208: flagset['proc_openwater'] changed to flagset['proc_complexwater']
  in flags compilation for bands 'CHL_NN', 'TSM_NN', 'ADG443_NN'
release 2.2 20181116
  check filename against substring 'S3?_OL_2_W?R*', import fnmatch
  coverage field: 0 no coverage, 1 coverage, but no land mask
release 2.2a 20181219
  coverage field: -999.0 no coverage, 1.0 coverage, but no land mask
  20181221: dtype of coverage field float32
release 2.3 20190121
  brdf, bidirectional reflectance distribution function
  Morel - f/Q, foq ... brdf factor as foq0/foq (~ 1, i.e.  0.6 < foq0/foq < 1.2 )
  Wang2006 
  fresnel_sensor*fresnel_solar*foq0/foq
"""

"""
command line
  -b typical
  -fl jul17
  -a med
  -res med
  -od /satdata/mario
  /satdata/olci/S3A_OL_2_WFR____20170722T092331_20170722T092631_20170723T165647_0179_020_150_2340_MAR_O_NT_002.SEN3

if file is zip or tar ...
  -t /tmp 
  -del

other valid filenames: ...
/satdata/olci/S3A_OL_2_WFR____20170722T092331_20170722T092631_20170723T165647_0179_020_150_2340_MAR_O_NT_002.SEN3
/satdata/olci/S3A_OL_2_WFR____20170719T071705_20170719T072005_20170719T092521_0179_020_106_2160_MAR_O_NR_002.SEN3
/satdata/olci/S3A_OL_2_WFR____20170206T110511_20170206T110811_20170206T130618_0179_014_094_2160_MAR_O_NR_002.SEN3
/satdata/olci/S3A_OL_2_WRR____20170206T110121_20170206T114508_20170206T133646_2626_014_094______MAR_O_NR_002.SEN3
/DataArchive/OC/OLCI/2017/140/sources/ODA/MED/S3A_OL_2_WFR____20170520T095410_20170520T095710_20170520T120108_0179_018_022_2159_MAR_O_NR_002.SEN3.tar 
"""
"""
other command lines
-b typical -fl jul17 -a bs -res bs -od /satdata/mario /satdata/olci/S3A_OL_2_WFR____20170822T073547_20170822T073847_20170823T133439_0179_021_206_2160_MAR_O_NT_002.SEN3
-b typical -fl jul17 -a med -res med -od /satdata/mario /satdata/olci/S3A_OL_2_WFR____20170822T073547_20170822T073847_20170823T133439_0179_021_206_2160_MAR_O_NT_002.SEN3
-b typical -fl jul17 -a med -res med -od /satdata/mario /satdata/olci/S3A_OL_2_WFR____20170822T073847_20170822T074147_20170823T133558_0179_021_206_2340_MAR_O_NT_002.SEN3

"""


####################
## initialization ##
####################

release='2.3'

import warnings
warnings.filterwarnings('ignore')

import os, yaml, sys, tarfile, zipfile, argparse, datetime, multiprocessing
import fnmatch
from shutil import rmtree
import numpy as np
from netCDF4 import Dataset
from pyresample import geometry, utils, kd_tree
import pyproj
import snappy
from scipy import misc

#2.3
sys.path.insert(0,os.path.join(os.environ['HOME'],'bin/BRDF_mario'))
from brdf_mario import brdf, foq_nbands, foq_bands
# foq_nbands=7, foq_bands=np.array([412.5, 442.5, 490. , 510. , 560. , 620. , 660. ])

n_cores=multiprocessing.cpu_count()
cwd=os.getcwd()
runtime=datetime.datetime.utcnow()

#read configuration file in same directory where we reside
sdir=os.path.dirname(os.path.realpath(__file__)) #where we reside!
configfile=os.path.join(sdir,'s3olciprocConf.yaml')
if os.path.isfile(configfile):
    with open(configfile) as fp:
        strdata=fp.read()
    strdata=strdata.replace('\t', '  ')  #remove tabs from yaml file
    conf=yaml.load(strdata)
else:
    print "## missing configuration file:"
    print "   "+configfile
    sys.exit()

# Command line parsing
parser=argparse.ArgumentParser(description='Reproject Sentinel3-OLCI bands on equirectangular map, rel. %s'%release)
parser.add_argument('iprod', action='store', help='input Sentinel3 L2 product ((directory or tarfile or zipfile)')
parser.add_argument("-t", "--tardir", action="store", help="temp directory where to un-tar or unzip PRODUCT.tar/.zip, default is current dir")
parser.add_argument("-del", "--delete", action="store_true", help="Delete un-tar-ed or unzipped files, default keeps them")
parser.add_argument("-c", "--extraconfig", action="store", help="specify an extra configuration file, besides "+configfile)
parser.add_argument("-b", "--bandlist", action="store",help="List of bands to be reprojected, as defined in config file(s)")
parser.add_argument("-fl", "--flaglist", help="Set of flags to be applied, , as defined in config file(s)")
parser.add_argument("-a", "--area",  action="store", help="Area to project to, as defined in config file(s)")
parser.add_argument("-res", "--resolution", action="store", help="resolution name, as defined in config file(s)")
parser.add_argument("-o", "--ofname",action="store", help="Output filename, default Oyyyyjjjhhmmss--area-hr.nc")
parser.add_argument("-od", "--outdir", action="store", help="Output directory name, default current directory")
parser.add_argument("-v", "--verbose",action="store_true", help="Verbose output")
args=parser.parse_args()

#merge config files
if args.extraconfig:
    if os.path.isfile(args.extraconfig):
        with open(args.extraconfig) as fp:
            strdata=fp.read()
        strdata=strdata.replace('\t', '  ')  #remove tabs from yaml file
        conf2=yaml.load(strdata)
    else:
        if args.verbose:
            print "## extra config file not found, ignoring ..."
    #update configuration settings
    for key in conf2.keys():
        if type(conf[key]) is dict:
            conf[key].update(conf2[key])
        else:
            conf[key]=conf2[key]

if args.bandlist in conf['bandlists'].keys():
    bandlist=conf['bandlists'][args.bandlist]
else:
    if args.verbose:
        print "## bandlist not in config file(s)"
    sys.exit()

if args.area in conf['areas'].keys():
    w_bound_deg,s_bound_deg,e_bound_deg,n_bound_deg=conf['areas'][args.area]
else:
    if args.verbose:
        print "## area not in config file(s)"
    sys.exit()

## new flags reading procedure: flags per band
flags=dict()
for band in bandlist:
    flags[band]=''
flagset=dict()

if args.flaglist in conf['perbandflaglists'].keys():
    for flagsetname in ('common_oc_positive', 'common_oc', 'proc_openwater', 'proc_complexwater', 'proc_watervapour' ):
        if flagsetname in conf['perbandflaglists'][args.flaglist].keys() and conf['perbandflaglists'][args.flaglist][flagsetname]:            
            flagset[flagsetname] = '(' + \
                                   ' | '.join(conf['perbandflaglists'][args.flaglist][flagsetname])  + \
                                   ')'
            # flag | flag | flag .... 
            """
            flagset[flagsetname] += conf['perbandflaglists'][args.flaglist][flagsetname][0]
            if len(conf['perbandflaglists'][args.flaglist][flagsetname]) > 1:
                for flag in conf['perbandflaglists']['jul17'][flagsetname][1:]:
                    flagset[flagsetname] += ' | ' + flag
            flagset[flagsetname] += ')'
            """
        else:
            flagset[flagsetname] = ''
    for band in bandlist:
        if 'iwv' in band.lower():
            flags[band]= 'not ' + flagset['proc_watervapour']
        if 'reflectance' in band.lower() or band in ('CHL_OC4ME', 'KD490_M07', 'PAR', 'T865', 'A865', 'CHL'):
            flags[band] =  flagset['common_oc_positive']
            if flagset['common_oc']:
                flags[band] += ' and not ' + flagset['common_oc']
            if flagset['proc_openwater']:
                flags[band] += ' and not ' + flagset['proc_openwater']
        if band in ('CHL_NN', 'TSM_NN', 'ADG443_NN'):
            flags[band] =  flagset['common_oc_positive']
            if flagset['common_oc']:
                flags[band] += ' and not ' + flagset['common_oc']
            if flagset['proc_complexwater']:
                flags[band] += ' and not ' + flagset['proc_complexwater']
        if band in conf['perbandflaglists'][args.flaglist].keys():
            flags[band] += ' and not (' + \
                           ' | '.join(conf['perbandflaglists'][args.flaglist][band]) + \
                           ')'

elif args.flaglist in conf['flaglists'].keys():
    #backward compatibilty: same flags for all bands
    flaglist=conf['flaglists'][args.flaglist]
    FLAGS  = 'not (' + flaglist[0]
    for flag in flaglist[1:]:
        FLAGS+= '| ' + flag
    FLAGS +=')'
    for band in bandlist:
        flags[band]=FLAGS
else:
    if args.verbose:
        print "## flaglist not provided or not in config file(s): no flags will be applied"
    # flags[band] already set to empty strings!

if args.resolution in conf['resolutions'].keys():
    xsize,ysize=conf['resolutions'][args.resolution]
else:
    if args.verbose:
        print "## resolution name not found in config file(s)"
    sys.exit()

if args.outdir and os.path.isdir(args.outdir):
    outdir=args.outdir
else:
    outdir=cwd        
    
#new names of bands
bandnames=conf['bandnames']

#chl_MEDOC algorithm parameters:
medoc_coeff=conf['algorithm_parameters']['chl_medoc']['coefficients']
medoc_MAXCHL=conf['algorithm_parameters']['chl_medoc']['max_chl']

#reflectrance type
reflectance_type=conf['reflectance_type']

#definition of the area and of the projection (prj4)
area_id=args.area
area_name=args.area
proj_id=args.area
proj4_args=('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +units=m')
prj = pyproj.Proj(proj4_args)
# X_bound_deg is bound in degree, X_bound_m is meter on map
[w_bound_m,e_bound_m],[s_bound_m,n_bound_m]=prj([w_bound_deg,e_bound_deg],[s_bound_deg,n_bound_deg])
area_extent=(w_bound_m, s_bound_m, e_bound_m, n_bound_m)
area = utils.get_area_def(area_id, area_name, proj_id, proj4_args, xsize, ysize, area_extent)
xxx,yyy=area.get_lonlats()

# read radius parameter, (default is 5000 meters), and fillvalue for netcdf
radius=int(conf['radius'])
fillvalue=float(conf['fillvalue'])

#read mask file names 
bs_land=os.path.join(sdir, conf['masks']['bs_land'])
med_land=os.path.join(sdir, conf['masks']['med_land'])
bs_case=os.path.join(sdir, conf['masks']['bs_case'])

#####################
## /initialization ##
#####################


# opening input product args.iprod
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
#rel 2.1c, zip files from CODA
if not l2prodname.endswith('.SEN3'):   
    l2prodname+='.SEN3'
# Check if l2prodname indecate and S3[A|B|C|D] OLCI product, either WFR of WRR
if fnmatch.fnmatch(l2prodname,'S3?_OL_2_W[FR]R*'):
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
        if args.verbose:
            print args.iprod+' not a valid product'
        sys.exit()
else:
    if args.verbose:
        print args.iprod+" is not an OLCI level 2 Water Full/Reduced Resolution product"
    sys.exit()
    
#detect the timeliness of OLCI L2 product, NR for Near Real Time, NT for 
#Non Time critical, i.e. DT, ... chars 88 and 89 of OLCI productname
timeliness=l2prodname[88:90]

if args.verbose:
        print "opening product ", product
p=snappy.ProductIO.readProduct(product)

w=p.getSceneRasterWidth()   # was readvar.getRasterWidth()  in the following stanza
h=p.getSceneRasterHeight()  # was readvar.getRasterHeight() in the following stanza    
readvar=p.getBand('latitude')
lats=np.zeros(w*h, np.float32)
readvar.readPixels(0,0,w,h,lats)
latmin=lats.min()
latmax=lats.max()
lats.shape=h,w    
readvar=p.getBand('longitude')
lons=np.zeros(w*h, np.float32)
readvar.readPixels(0,0,w,h,lons)
lonmin=lons.min()
lonmax=lons.max()
lons.shape=h,w

#definition of the swath object
swath = geometry.SwathDefinition(lons=lons, lats=lats)    

start_time_str=str(p.getStartTime())
start_time=datetime.datetime.strptime(start_time_str, '%d-%b-%Y %H:%M:%S.%f')
stop_time_str=str(p.getEndTime())
stop_time=datetime.datetime.strptime(stop_time_str, '%d-%b-%Y %H:%M:%S.%f')

#test if swath and area overlap. If not, exit
if not(area.overlaps(swath)):
    if args.verbose:
        print "product outside of area"
    p.dispose()
    if (istar or iszip) and args.delete:
        rmtree(product)
    sys.exit()

#create output file
if args.ofname:
    ofname=args.ofname
else:
    #ofname='O'+start_time.strftime('%Y%j%H%M%S')+'--'+area_name+'-hr.nc'
    # temporary only add _brdf to filename
    ofname='O'+start_time.strftime('%Y%j%H%M%S')+'--'+area_name+'-hr_brdf_w_ANNOT_DROUT.nc'
ofname=os.path.join(outdir,ofname)
fmb=Dataset(ofname, 'w') #create output file
fmb.createDimension('lat', ysize)
fmb.createDimension('lon', xsize)
lat = fmb.createVariable('lat',  'single', ('lat',))    
#lat[:]=np.linspace(s_bound_deg, n_bound_deg, ysize, endpoint=True)
lat[:]=yyy[:,0]
setattr(fmb.variables['lat'], 'units', 'degrees_north')
setattr(fmb.variables['lat'], 'long_name', 'latitude')
setattr(fmb.variables['lat'], 'standard_name', 'latitude')
setattr(fmb.variables['lat'], 'valid_min', s_bound_deg)
setattr(fmb.variables['lat'], 'valid_max', n_bound_deg)
lon = fmb.createVariable('lon',  'single', ('lon',))
#lon[:]=np.linspace(w_bound_deg, e_bound_deg, xsize, endpoint=True)
lon[:]=xxx[0,:]
setattr(fmb.variables['lon'], 'units', 'degrees_east')
setattr(fmb.variables['lon'], 'long_name', 'longitude')
setattr(fmb.variables['lon'], 'standard_name', 'longitude')
setattr(fmb.variables['lon'], 'valid_min', w_bound_deg)
setattr(fmb.variables['lon'], 'valid_max', e_bound_deg)
#global attributes of file
fmb.netcdf_version='v4'
fmb.Conventions= "CF-1.4"
fmb.platform = "Sentinel-3a"
fmb.product_level='L3'
fmb.contact= "technical@gos.artov.isac.cnr.it"
fmb.institution="CNR-GOS"
fmb.references=''
fmb.sensor='Ocean and Land Colour Instrument'
fmb.sensor_name='OLCIa'
fmb.reproject=sys.argv[0]
fmb.easternmost_longitude=e_bound_deg
fmb.westernmost_longitude=w_bound_deg
fmb.northernmost_latitude=n_bound_deg
fmb.southernmost_latitude=s_bound_deg
fmb.start_date=start_time.strftime("%Y-%m-%d")
fmb.start_time=start_time.strftime("%H:%M:%S UTC")
fmb.stop_date=stop_time.strftime("%Y-%m-%d")
fmb.stop_time=stop_time.strftime("%H:%M:%S UTC")
fmb.source_file=p.getName()
fmb.grid_resolution=" 1 km"
fmb.grid_mapping="Equirectangular"
fmb.software_name = "GOS Processing chain"
fmb.creation_date = runtime.strftime("%Y-%m-%d")
fmb.creation_time = runtime.strftime("%H:%M:%S UTC")
fmb.distribution_statement = "See CMEMS Data License"
fmb.naming_authority = "CMEMS"
fmb.cmems_production_unit = "OC-CNR-ROMA-IT"
fmb.institution = "CNR-GOS"
fmb.source='surface observation'
fmb.timeliness=timeliness
if timeliness=="NT":
    fmb.product_version='v02'
else:
    fmb.product_version='v02QL'
    
#extract the neighbour info once and reuse many, to optimize computation time
#http://pyresample.readthedocs.io/en/latest/swath.html#resampling-from-neighbour-info
# still in doubt why kd_tree.get_sample_from_neighbour_info does not accept nprocs=n_cores argument ...
valid_input_index, valid_output_index, index_array, distance_array =\
    kd_tree.get_neighbour_info(swath, area, radius, neighbours=1, nprocs=n_cores)

if args.verbose:
    print "... Computing BRDF coefficients"

# call the bdrf correction
# and create the new RRS bands

#wind speed components, ws0 and ws1 and
#viewing geometries:
#OAA,OZA,SAA,SZA
#O obs, S solar
#A Azimuth, Z Zenith
#A Angle
ws0=np.zeros(h*w, np.float32)
ws1=np.zeros(h*w, np.float32)
chl=np.zeros(h*w, np.float32)
sza=np.zeros(h*w, np.float32)
saa=np.zeros(h*w, np.float32)
vza=np.zeros(h*w, np.float32)
vaa=np.zeros(h*w, np.float32)

varhandle=p.getTiePointGrid('horizontal_wind_vector_1')
varhandle.readPixels(0,0,w,h,ws0)
varhandle=p.getTiePointGrid('horizontal_wind_vector_2')
varhandle.readPixels(0,0,w,h,ws1)
varhandle=p.getBand('CHL_OC4ME')
varhandle.readPixels(0,0,w,h,chl)
varhandle=p.getTiePointGrid('SZA')
varhandle.readPixels(0,0,w,h,sza)
varhandle=p.getTiePointGrid('SAA')
varhandle.readPixels(0,0,w,h,saa)
varhandle=p.getTiePointGrid('OZA')
varhandle.readPixels(0,0,w,h,vza)
varhandle=p.getTiePointGrid('OAA')
varhandle.readPixels(0,0,w,h,vaa)

specBRDF_mario=np.ones((w*h,foq_nbands), np.float32)
specBRDF_mario=brdf(ws0, ws1, chl, sza, saa, vza, vaa)
del ws0, ws1, chl, sza, saa, vza, vaa # free some memory

if args.verbose:
    print "... Done BRDF coefficients"

# foq_band_names = {'Oa02_reflectance': 0, 'Oa03_reflectance': 1, 'Oa04_reflectance': 2,
#                   'Oa05_reflectance': 3, 'Oa06_reflectance': 4, 'Oa07_reflectance': 5, 
#                   'Oa08_reflectance': 6}
# foq_bands = np.array([412.5, 442.5, 490. , 510. , 560. , 620. , 660. ])
# mo non tengo voglia di scrivere sta cosa meglio di cosi:
# associative array (i.e. dict) of bands for which we have a bdrf correction
foq_band_idx_by_name=dict()
for i in range(2,9):
    foq_band_idx_by_name["Oa0%i_reflectance"%i]=i-2

    
for bandname in bandlist:
    # main cicle: process all the requested bands!
    #change bandname to varname, if present in config file
    if bandname in bandnames.keys():
        varname=bandnames[bandname]
        if args.verbose:
            print "... reading band ", bandname, ' renamed to ', varname
    else:
        varname=bandname
        if args.verbose:
            print "... reading band ", bandname
    
    #compute the BAND (coverage, or chl with medoc algorith, or new ones to come)
    #OR
    #read the BAND from input file, masking with flags[bandname]
    #and put it the swath_var variable
    if bandname.lower()=='coverage':
        #"coverage" field: 1.0 or -999.0 if pixel is in swath or not - float32
        bandtype=np.dtype('float32')
        swath_var=np.ones((h,w),dtype='float32')
    elif bandname.lower()=='chl' and area_id != 'bs':
        #CHL_MEDOC alghorithm (Volpe et al. 2007)
        bandtype=np.dtype('float32')
        isFlag=False
        rrsfield=dict()
        for rrs in ('Oa03_reflectance', 'Oa04_reflectance', 'Oa05_reflectance', 'Oa06_reflectance'):
            apporrs=np.zeros(w*h,bandtype)
            read_rrs=p.getBand(rrs)
            read_rrs.readPixels(0,0,w,h,apporrs)
            if rrs in foq_band_idx_by_name.keys():
                w_idx=foq_band_idx_by_name[rrs]
                apporrs*=specBRDF_mario[:,w_idx]
            valids=np.zeros(w*h, np.uint8)
            read_rrs.setValidPixelExpression(flags[rrs])
            read_rrs.readValidMask(0,0,w,h,valids)
            invalids=np.where(valids==1,0,1)
            rrsfield[rrs]=np.ma.array(apporrs, mask=invalids, fill_value=fillvalue)            
            rrsfield[rrs]/=np.pi
        r=np.log10(np.maximum(rrsfield['Oa03_reflectance'],rrsfield['Oa04_reflectance'],rrsfield['Oa05_reflectance']) / rrsfield['Oa06_reflectance'])
        swath_var=10**(medoc_coeff[0] + medoc_coeff[1]*r + medoc_coeff[2]*r**2 + medoc_coeff[3]*r**3 + medoc_coeff[4]*r**4)
        swath_var=np.ma.masked_greater(swath_var, medoc_MAXCHL)
        swath_var.shape=h,w
    elif bandname.lower()=='chl' and area_id == 'bs':
        #BSAlg algorithm (Kopelevich et al., 2013)
        #  here only the band ratio rrs510*189.8667 / rrs560*188.2640 is computed
        #  the case1/case2 part is taken into account on the gridd_var (because of the mask!!)
        bandtype=np.dtype('float32')
        isFlag=False
        rrsfield=dict()
        for rrs in ('Oa05_reflectance', 'Oa06_reflectance'):
            apporrs=np.zeros(w*h,bandtype)
            read_rrs=p.getBand(rrs)
            read_rrs.readPixels(0,0,w,h,apporrs)
            if rrs in foq_band_idx_by_name.keys():
                w_idx=foq_band_idx_by_name[rrs]
                apporrs*=specBRDF_mario[:,w_idx]
            valids=np.zeros(w*h, np.uint8)
            read_rrs.setValidPixelExpression(flags[rrs])
            invalids=np.where(valids==1,0,1)
            rrsfield[rrs]=np.ma.array(apporrs, mask=invalids, fill_value=fillvalue)
            #rrsfield[rrs]/=np.pi  # useless, ... ratio will be taken ...
        swath_var=(rrsfield['Oa05_reflectance']*189.8667)/(rrsfield['Oa06_reflectance']*188.2640)
        swath_var.shape=h,w    
    else:
        readband=p.getBand(bandname)
        # check if variable exists!
        if readband is None:
            if args.verbose:
                print 'XXX ... band not present in source file'
            continue
        isFlag=readband.isFlagBand()
        if readband.isFloatingPointType():
            bandtype=np.dtype('float32')
        else:
            bandtype=np.dtype('int32')
        band=np.zeros(w*h,bandtype)
        readband.readPixels(0,0,w,h,band)
        if bandname in foq_band_idx_by_name.keys():
            w_idx=foq_band_idx_by_name[bandname]
            band*=specBRDF_mario[:,w_idx]
        valids=np.zeros(w*h, np.uint8)
        readband.setValidPixelExpression(flags[bandname])
        readband.readValidMask(0,0,w,h,valids)
        invalids=np.where(valids==1,0,1)
        swath_var=np.ma.array(band, mask=invalids, fill_value=fillvalue)
        swath_var.shape=h,w

    #create the netcdf variable in the output file    
    gridd_var=fmb.createVariable(varname, bandtype, ('lat', 'lon',), fill_value=fillvalue, zlib=True, complevel=6)
    
    #create a numpy *masked/unmasked* array to hold the values to be later saved as gridd_var in netcdf
    # netcdf variables cannot be addressed with e.g. gridd_var[land_mask]=fillvalue
    #   this would result in "Index cannot be multidimensional   !!!"
    # resample the swath_var from neighbour info ... was gridd_var[:]=
    gridd_array=kd_tree.get_sample_from_neighbour_info('nn', \
                                                         area.shape, \
                                                         swath_var, \
                                                         valid_input_index, \
                                                         valid_output_index, \
                                                         index_array, \
                                                         fill_value=fillvalue)

    #print bandname, type(gridd_array)
    # case1/case2 elaboration for chl in Black Sea bs:
    if bandname.lower()=='chl' and area_id == 'bs':
        mask=misc.imread(bs_case) #mask for case1/case2 !!
        mask=mask[:,:,1]          #use only GREEN byte ... 
        c1vals=(210,136,187,217,152,255)
        c2vals=(70,0,175)
        #cXpxls are bool arrays, True if pixel is case X, X=1,2
        c1pxls=np.in1d(mask, c1vals)   #numpy.isin available only in Numpy 1.13
        c1pxls=c1pxls.reshape(mask.shape)
        c2pxls=np.in1d(mask, c2vals)
        c2pxls=c2pxls.reshape(mask.shape)
        # apply caseX algorithm to caseX pixels, X=1,2
        gridd_array[c1pxls]=1.13* gridd_array[c1pxls]**(-3.33)
        gridd_array[c2pxls]=0.88* gridd_array[c2pxls]**(-2.24)
    
    #put -999 in 'coverage' band where it is not 1
    if bandname.lower()=='coverage':
        gridd_array[gridd_array<>1]=-999
    
    # OLCI reflectances are directional reflectance 
    # need to be divided by pi in order to have Remote Sensing Reflectance
    # http://www.oceanopticsbook.info/view/overview_of_optical_oceanography/reflectances
    # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-3-olci/product-types/level-2-water
    # already multiplied by brdf factor earlier ... 
    if "reflectance" in bandname:
        if reflectance_type=='rrs':
            gridd_array[:]/=np.pi

    # Apply basin mask, for all bands except coverage
    if bandname.lower()!='coverage':
        if area_id == 'bs':
            basin_mask=Dataset(bs_land,'r')
            basin_mask_present=True
        elif area_id == 'med':
            basin_mask=Dataset(med_land,'r')
            basin_mask_present=True
        else:
            basin_mask_present=False    
        if basin_mask_present:
            land_mask=np.where(basin_mask.variables['Land_Mask'][:]==1, True, False)        
            land_mask=np.flipud(land_mask) ### 0,0 is lower left!!
            gridd_array[land_mask]=fillvalue
            if isinstance(gridd_array, np.ma.core.MaskedArray):   # paranoia paranoia
                gridd_array.mask=np.ma.mask_or(gridd_array.mask,land_mask)
            basin_mask.close()
            
    # copy gridd_array in netcdf var
    gridd_var[:]=gridd_array
    
    #attributes of var
    setattr(fmb.variables[varname], 'coordinates', 'lat lon')
    if varname != bandname:
        setattr(fmb.variables[varname], 'band_name', 'OLCI band name '+bandname)
    if "reflectance" in bandname:
        if reflectance_type=='rrs':
            setattr(fmb.variables[varname], 'long_name', 'Remote Sensing Reflectance at '+varname)
            setattr(fmb.variables[varname], 'standard_name', 'surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air')
            setattr(fmb.variables[varname], 'units', 'sr^-1')
            setattr(fmb.variables[varname], 'valid_min', 1e-06)
            setattr(fmb.variables[varname], 'valid_max', 1)
        elif reflectance_type=='rho':
            setattr(fmb.variables[varname], 'long_name', 'spectral irradiance reflectance at '+varname)
            setattr(fmb.variables[varname], 'standard_name', 'surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_irradiance')
            setattr(fmb.variables[varname], 'units', 'dimensionless')
        else:
            setattr(fmb.variables[varname], 'long_name', 'reflectance (unknown type) at '+varname)
        setattr(fmb.variables[varname], 'type', 'surface')
    setattr(fmb.variables[varname], 'source', 'OLCI - Level2')
    if bandname.lower() == 'kd490':
        setattr(fmb.variables[varname], 'long_name', 'OLCI Diffuse Attenuation Coefficient at 490nm')
        setattr(fmb.variables[varname], 'standard_name', 'volume_attenuation_coefficient_of_downwelling_radiative_flux_in_sea_water')
        setattr(fmb.variables[varname], 'type', 'surface')
        setattr(fmb.variables[varname], 'units', 'm^-1')
        setattr(fmb.variables[varname], 'missing_value', -999.0)
        setattr(fmb.variables[varname], 'valid_min', 0.0)
        setattr(fmb.variables[varname], 'valid_max', 10.0)
    if bandname.lower()=='chl':
        setattr(fmb.variables[varname], 'long_name', 'Chlorophyll a concentration')
        setattr(fmb.variables[varname], 'standard_name', 'mass_concentration_of_chlorophyll_a_in_sea_water')
        setattr(fmb.variables[varname], 'type', 'surface')
        setattr(fmb.variables[varname], 'units', 'milligram m^-3')
        setattr(fmb.variables[varname], 'missing_value', -999.0)
        setattr(fmb.variables[varname], 'valid_min', 0.01)
        setattr(fmb.variables[varname], 'valid_max', 100.0)
        setattr(fmb.variables[varname], 'comment', 'r=log10(max([rrs442_5,rrs490,rrs510])/rrs560);chlcase1=10^[coeff([0]+coeff[(1]*r+coeff[2[)*r^2+coeff[3]*r^3+coeff[4]*r^4]')
    if bandname.lower()=='coverage':
        setattr(fmb.variables[varname], 'coverage_field', '1 if pixel is in swath, -999 if it is not, regardles land mask')
    if bandname in flags.keys():
        setattr(fmb.variables[varname], 'applied_flags', flags[bandname])
        
    # /main cycle END


# closing and cleaning up
fmb.close()
p.dispose()
if (istar or iszip) and args.delete:
    rmtree(product)
