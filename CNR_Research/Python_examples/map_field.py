import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import sys, os
import argparse
import glob
import datetime
#%%
from color_constants import RGB
from shapely.geometry import Point, Polygon
import geopandas as gpd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
parser=argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True)
parser.add_argument('-sd', '--sdate', nargs=2, required=True)
parser.add_argument('-ed', '--edate', nargs=2, required=True)
parser.add_argument('-f', '--products',nargs='+', required=True)
parser.add_argument('-it', '--it', nargs='+',required=False)
parser.add_argument('-st', '--st', nargs='+',required=False)
parser.add_argument('-r', '--res',required=False)
args=parser.parse_args()
path=args.path
if args.res:
	res=args.res.lower()
	if res not in ['f', 'r']:
		sys.exit('map resolution (-res) can be just full (f) or reduced(r)')
else:
	res='f'
	
sjday=args.sdate[1]
sjday='0'*(3-len(str(sjday)))+sjday
ejday=args.edate[1]
ejday='0'*(3-len(str(ejday)))+ejday
syear=args.sdate[0]
eyear=args.edate[0]
prods=args.products
files=sorted(glob.glob(path+'*.nc'))
sjdate=syear+sjday
ejdate=eyear+ejday
sdate=datetime.datetime(int(syear)-1,12,31)+datetime.timedelta(days=int(sjday))
edate=datetime.datetime(int(eyear)-1,12,31)+datetime.timedelta(days=int(ejday))
sdates=str(sdate.year)+'0'*(2-len(str(sdate.month)))+str(sdate.month)+'0'*(2-len(str(sdate.day)))+str(sdate.day)
edates=str(edate.year)+'0'*(2-len(str(edate.month)))+str(edate.month)+'0'*(2-len(str(edate.day)))+str(edate.day)
files_list=[]
ind_in=[]
ind_fin=[]
map_dir='/home/Marco.Bracaglia/progetti/VGOCS/maps/'+sdates+'_'+edates+'/'
if os.path.isdir(map_dir)==False:
	os.makedirs(map_dir)
for i,f in enumerate(files):
	filename=f.split('/')[-1]
	if filename.replace('VGOCS_','')[:7]==sjdate:
		ind_in.append(i)
	if filename.replace('VGOCS_','')[:7]==ejdate:
		ind_fin.append(i)

files_list=files[ind_in[0]:ind_fin[-1]+1]
if args.it:
	it=args.it
	if len(it)!=len(prods):
		sys.exit('Products and threshold must have same size')
if args.st:
	st=args.st
	if len(st)!=len(prods):
		sys.exit('Products and threshold must have same size')
sh = gpd.read_file('/home/Marco.Bracaglia/progetti/VIIRS_SWATH/ISPRA/Bacini_idrografici_principali_0607.shp')
sh = sh.to_crs(epsg=4326)
if res=='f':  #LOAD THE SHAPE FILE
	sh1 = gpd.read_file('/home/Marco.Bracaglia/progetti/VIIRS_SWATH/GSHHS_shp/h/GSHHS_h_L1.shp')
	borders=list(sh1['geometry'])
rivers=list(sh['geometry'])[:7]

coords=[(12.,44.), (12.,46.), (14.,46.), (14.,44.)]
poly_NAD=Polygon(coords)

for nc_file in files_list:
	nc_f=Dataset(nc_file, 'r')
	for i, prod in enumerate(prods):
		if prod.upper()=='RRS':
			prod[:3]='Rrs'
			group='rrs_data'
		else:
			group='IOP_data'
		out_dir=map_dir+prod.replace('_','').lower()+'/'
		if os.path.isdir(out_dir)==False:
			os.makedirs(out_dir)
		filename=nc_file.split('/')[-1]
		mapname=filename.replace('VGOCS', prod.replace('_','').lower()).replace('.nc', '.png')
		g=nc_f.groups[group]	
		var=g[prod][:]
		if i==0:
			lat=nc_f.variables['lat'][:]
			lon=nc_f.variables['lon'][:]
		if args.it:
			inf=float(it[i])
		else:
			inf=0.
		if args.st:
			sup=float(st[i])
		else:
			sup=ma.max(var)
		
		m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
			llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='f')
		x,y=np.meshgrid(lon, lat)
		m.drawparallels(np.linspace(min(lat), max(lat), 7),labels=[0,0,0,0])
		m.drawmeridians(np.linspace(min(lon), max(lon), 7),labels=[0,0,0,0])
		if res=='f':
			m.drawlsmask(land_color=RGB.hex_format(RGB(169,169,169)),ocean_color=RGB.hex_format(RGB(169,169,169)),resolution='f',lakes=True, grid=1.25) #same colour for land and ocean
			for (r,i) in zip(borders, range(len(borders))): #plot the borders and fill the continent if the polygon is inside poly_NAD
				#if inf['NAM']!='UNK':
				#if r.intersects(poly_NAD):
				elements=list(r.exterior.coords)
				xx,yy=zip(*elements)
				xy = zip(xx,yy)
				poly = Polygon( xy )
				if poly.intersects(poly_NAD)==True:
					m.plot(xx,yy,marker=None, color='black', linewidth=0.5)
					poly = matplotlib.patches.Polygon( xy, facecolor=RGB.hex_format(RGB(238, 213, 183)) )
					plt.gca().add_patch(poly)
		else:
			m.drawlsmask(land_color=RGB.hex_format(RGB(238, 213, 183)),ocean_color=RGB.hex_format(RGB(169,169,169)),resolution='f',lakes=True, grid=1.25)	
			m.drawcoastlines()
			
		for (r,i) in zip(rivers, range(len(rivers))):
			#if inf['NAM']!='UNK':
			if i!=3 and i!=0 and i!=1:
				elements=list(r.exterior.coords)
				xx,yy=zip(*elements)
				m.plot(xx,yy,marker=None, color=RGB.hex_format(RGB(0,229,238)), linewidth=1)

		m.drawrivers(linewidth=1.0, color=RGB.hex_format(RGB(0,229,238)))
		cs=m.imshow(var,origin='upper', extent=[min(lon), max(lon), min(lat), max(lat)],cmap=plt.cm.Spectral_r ,vmin=inf, vmax=sup)
		plt.axes().set_aspect('equal')
		cbar=m.colorbar(cs,location='bottom',pad="10%")
		cbar.ax.tick_params(labelsize=7)
		cbar.set_label(prod, fontsize=12)
		plt.savefig(out_dir+mapname, dpi=200)
		plt.close()







