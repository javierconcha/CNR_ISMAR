#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:19:18 2019

@author: javier

from computemdistance.pro and make_chl_mario.pro by Mario
This classifier is based on D'Alimonte et al. (2003): "Use of the Novelty Detection Technique to
Identify the Range of Applicability of Empirical Ocean Color Algorithms"

Inputs:
    - CovMatAAOT_245.nc
    - CovMatMEDOC4_245.nc
"""
#%% Initializaion
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap

from numpy.linalg import inv
from numpy.linalg import det

import math

import os.path
import os
os.environ['QT_QPA_PLATFORM']='offscreen' # to avoid error "QXcbConnection: Could not connect to display"

import time

from os import access, R_OK
#%%

def computemdistance(CovMat,mean_data,x):
#;++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#;NAME
#; ComputeDistance
#;
#;DESCRIPTION
#; Function for computing Mahalanobis
#;
#;CALLING SEQUENCE
#; ComputeMDistance,matstr,x
#;
#;INPUTS
#; matstr: matrix structure
#; x: input vector
#
#;OUTPUTS
#; Mahalanobis distance
#;
#;COMMENTS
#; See D'alimonte et al., IEEE, 2003.
#;
#;MODIFICATION HISTORY
#; Written by: Melin F.
#;--------------------------------------------------------------------------------------------------    
#    dimx = len(x)
#    determinante = det(CovMatAAOT_245_MATRIX)
    um = mean_data # mean
    sigma = inv(CovMat)
    xx = x - um
    xt = np.transpose(xx)
    
    distance = xx @ sigma @ xt
    
    return distance
    
#%%
#    host = 'mac'
host = 'mac'

if host == 'mac':
    path_in = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/DataArchive/OLCI_NAS/20160426_20190228'   
    path_out = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/DataArchive/OLCI_NAS/data/'        
    path_list = path_in
elif host == 'vm':
    path_in = '/store3/OLCI_NAS/20160426_20190228/'
    path_out = '/home/Vittorio.Brando/Javier/data'
    path_list = '/home/Vittorio.Brando/Javier/codes'
else:
    print('Not sure from where this script will be run!')
    
#%% To plot density      
filename = 'VGOCS_2016117094003_O.nc' # open an original for to copy properties to output file

nc_f0=Dataset(os.path.join(path_in,filename), 'r')
    
lat0 = nc_f0.variables['lat'][:]
lon0 = nc_f0.variables['lon'][:]

rrs442_5 = nc_f0.groups['rrs_data']['Rrs_442.5'][:]
rrs490 = nc_f0.groups['rrs_data']['Rrs_490.0'][:]
rrs510 = nc_f0.groups['rrs_data']['Rrs_510.0'][:]
rrs560 = nc_f0.groups['rrs_data']['Rrs_560.'][:]

ylen = len(lat0)
xlen = len(lon0)    

path1 = '/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/OLCI_flag_comp/Case12_proc'
filename1 = 'CovMatAAOT_245.nc'
filename2 = 'CovMatMEDOC4_245.nc'
nc_f1=Dataset(os.path.join(path1,filename1), 'r')
nc_f2=Dataset(os.path.join(path1,filename2), 'r')

# Case 2
CovMatAAOT_245_MATRIX = nc_f1.variables['MATRIX'][:]
CovMatAAOT_245_INVERSE = nc_f1.variables['INVERSE'][:]
CovMatAAOT_245_DETERMINANT = nc_f1.variables['DETERMINANT'][:]
CovMatAAOT_245_MEAN = nc_f1.variables['MEAN'][:]

# Case 1
CovMatMEDOC4_245_MATRIX = nc_f2.variables['MATRIX'][:]
CovMatMEDOC4_245_INVERSE = nc_f2.variables['INVERSE'][:]
CovMatMEDOC4_245_DETERMINANT = nc_f2.variables['DETERMINANT'][:]
CovMatMEDOC4_245_MEAN = nc_f2.variables['MEAN'][:]

#;  443      490      510      555  nm
F0=np.array([188.76 , 193.38 , 192.56 , 183.76]) #SeaWiFS
#;  Mediterranean OC4'S FUNCTIONAL FORM
a1=np.array([0.066233171,-3.4442,3.8590429,-2.8148532,0.37119454]) # ; new, by /store/woc/insitu/optics/script/make_oc_algorithm.pro
#;  Adriatic OC4'S FUNCTIONAL FORM
a2=np.array([0.091,-2.620,-1.148,-4.949]) # ; Berthon & Zibordi (2003) IJRS
#;  a2=[0.236,-3.331,2.386,4.2834,-5.816] $ ; D'Alimonte & Zibordi (2003) IEEE

rrs = np.array([rrs442_5[300,300], rrs490[300,300], rrs510[300,300], rrs560[300,300]])

r1=np.log10(max([rrs[0],rrs[1],rrs[2]])/rrs[3])
case1=10**(a1[0]+a1[1]*r1+a1[2]*r1**2+a1[3]*r1**3+a1[4]*r1**4)
r2=np.log10(rrs[1]/rrs[3])                      
case2=10**(a2[0]+a2[1]*r2+a2[2]*r2**2+a2[3]*r2**3) # Berthon & Zibordi (2003) IJRS

spectrum=np.log10(rrs[[0,2,3]]*F0[[0,2,3]])
#%%
#; Case 1
d1 = computemdistance(CovMatMEDOC4_245_MATRIX,CovMatMEDOC4_245_MEAN,spectrum)
print(d1)
p1=1/((2*math.pi)**(len(spectrum)/2.)*math.sqrt(CovMatMEDOC4_245_DETERMINANT))*math.exp(-0.5*d1) # pdf for case 1
print(p1)


#; Case 2
d2 = computemdistance(CovMatAAOT_245_MATRIX,CovMatAAOT_245_MEAN,spectrum)
print(d2)
p2=1/((2*math.pi)**(len(spectrum)/2.)*math.sqrt(CovMatAAOT_245_DETERMINANT))*math.exp(-0.5*d2) # pdf for case 2
print(p2)

#;-------------------------------------
#; Merged Chl is compute with one of following case
#;-------------------------------------
#; Case 1
if d1 <= 6.25 and d2 > 6.25:
    print('Case 1')
    out = case1
    wtm = 1
#; Case 2    
elif d1 > 6.25 and d2 <= 6.25:
    print('Case 2')
    out = case2
    wtm = 0
#; Mix of the two (blending approach)
else:
    print('Mixed')
    out = (p1*case1+p2*case2)/(p1+p2)
    wtm = p1/(p1+p2)
    
print(out)
print(wtm)

if math.isnan(out):
#    localtime = time.asctime( time.localtime(time.time()) )
#    print(localtime+" make_merged_chl - INFO - Pixel ("+"column_line"+ ") was found NaN and set to missing value (-999)")
    out = -999

# From make_chl_mario.pro:
#    d1 LE 6.25 AND d2 GT 6.25: BEGIN & out(igood)=case1 & wtm(igood)=1 & END
#; Case 2
#    d1 GT 6.25 AND d2 LE 6.25: BEGIN & out(igood)=case2 & wtm(igood)=0 & END
#; Mix of the two
#    ELSE: BEGIN & out(igood)=(p1*case1+p2*case2)/(p1+p2) & wtm(igood)=p1/(p1+p2) & END
#    ENDCASE
#    IF FINITE(out(igood)) EQ 0 THEN BEGIN
#       PRINT,STRING(SYSTIME())+" make_merged_chl - INFO - Pixel ("+STRING(column_line(sea(good(igood)),dim(0),/SILENT),FORMAT='(2(i4,x))')+") was found NaN and set to missing value (-999)"
#       out(igood)=-999
#    ENDIF
#
#ENDFOR
#
#chl=FLTARR(dim)-999. & chl(sea(good))=out
#mas=FLTARR(dim)-999. & mas(sea(good))=wtm
#
#RETURN,CREATE_STRUCT('CHL',chl,'WTM',mas)

#%% Example: https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/
XYZ = np.matrix([[64, 580, 29],[66,570,33],[68,590,37],[69,660,46],[73,600,55]])
m = np.array([68,600,40])
cov_mat = np.cov(np.transpose(XYZ))
n = 5

v = np.array([66,640,44])

d3 = computemdistance(cov_mat,m,v)
MD3 = math.sqrt(d3)

print(d3)
print(MD3)