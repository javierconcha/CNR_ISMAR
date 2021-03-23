import numpy
import numpy.ma as ma
from netCDF4 import Dataset
import os, sys

def OLCI_flag(flag_file):
	nc_f=Dataset(flag_file, 'r')
	flag=nc_f.variables['WQSF'][:]
	mask=ma.zeros(flag.shape)
	"""mask[ma.where((flag & 1) |  (flag & 4) | (flag & 8) | (flag &  8388608) | (flag & 16777216) | (flag & 16) | (flag & 256) | (flag & 512 ) | (flag & 1024) | (flag & 4096) | (flag & 8192) | (flag &  131072) | \
	(flag &  549755813888) | (flag &  2199023255552) | (flag &  4398046511104) | (flag &  8796093022208) | (flag &  17592186044416) | (flag &  35184372088832) | \
	(flag & 70368744177664) | (flag &  140737488355328)| (flag & 17179869184) | (flag &  137438953472))]=1"""
	mask[ma.where((flag & 1) |  (flag & 4) | (flag & 8) | (flag &  8388608) | (flag & 16777216) | (flag & 16) | (flag & 256) | (flag & 512 ) | (flag & 1024) | (flag & 4096) | (flag & 8192) | (flag &  131072) | \
	(flag &  2199023255552) | (flag &  4398046511104) | (flag &  8796093022208) | (flag &  17592186044416) | (flag &  35184372088832) | \
	(flag & 70368744177664) | (flag &  140737488355328))]=1
	nc_f.close()
	return mask

