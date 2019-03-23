import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import argparse
import sys, os
parser=argparse.ArgumentParser()
parser.add_argument('-ff', '--firstf', type=str, action='store', help='The first nc file')
parser.add_argument('-sf', '--secondf', type=str, action='store', help='The second nc file')
args=parser.parse_args()
nc_f1=Dataset(args.firstf, 'r')
nc_f2=Dataset(args.secondf, 'r')
var1=nc_f1.variables
var2=nc_f2.variables
time=var1['Time'][:]
for (v1,v2) in zip(var1, var2):
	if v1=='Time' or v1=='Level':
		var1.pop(v1)
	if v2=='Time' or v2=='Level':
		var2.pop(v2)

for v in var1:
	if v not in var2:
		print ("WARNING: NO "+v+" VARIABLE IN THE SECOND FILE")
for v in var2:
	if v not in var1:
		print ("WARNING: NO "+v+" VARIABLE IN THE FIRST FILE")

for v in var1:
	if v in var2:
		arr1=var1[v][:]
		arr2=var2[v][:]
		mask1=ma.getmask(arr1)
		mask2=ma.getmask(arr2)
		diffm1=ma.where((mask1==True) & (mask2==False))
		diffm2=ma.where((mask2==1) & (mask1==0) )
		
		if len(diffm1[0])>0:
			print ("WARNING: SOME "+v+ " PIXEL ARE MASKED FOR THE FIRST FILE AND NOT FOR THE SECOND ONE")
		if len(diffm2[0])>0:
			
			print ("WARNING: SOME "+v+ " PIXEL ARE MASKED FOR THE SECOND FILE AND NOT FOR THE FIRST ONE")
		arr1=ma.array(arr1, mask=mask2)
		print (arr1.shape)
		arr2=ma.array(arr2, mask=mask1)
		diff=ma.abs((arr1[:,2]-arr2[:,2]))*100/ma.abs(arr2[:,2])
		diffm=ma.where(diff>.5)
		if len(diffm[0])>0:
			print (arr1[diffm])
			print (arr2[diffm])
			for t in time[diffm]:
				print (t)
			print ("WARNING NOT ALL VALUES ARE EQUAL FOR "+v+" VARIABLES")
