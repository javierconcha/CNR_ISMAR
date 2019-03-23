from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib


m = Basemap(llcrnrlat=min(lat),urcrnrlat=max(lat),\
	llcrnrlon=min(lon),urcrnrlon=max(lon), resolution='f')
x,y=np.meshgrid(lon, lat)
m.drawparallels(np.linspace(min(lat), max(lat), 7),labels=[0,0,0,0])
m.drawmeridians(np.linspace(min(lon), max(lon), 7),labels=[0,0,0,0])
m.drawlsmask(land_color='grey',ocean_color='white',resolution='f',lakes=True, grid=1.25)
m.drawcoastlines()
m.drawrivers(linewidth=1.0, color='cyan')
cs=m.imshow(var,origin='upper', extent=[min(lon), max(lon), min(lat), max(lat)])
plt.savefig(outpath, dpi=200)
plt.close()

