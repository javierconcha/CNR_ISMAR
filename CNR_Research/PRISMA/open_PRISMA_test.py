#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Feb 27 17:43:33 2020
Script to open data from the PRISMA sensor
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
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import h5py

# user defined functions
sys.path.insert(0,'/Users/javier.concha/Desktop/Javier/2019_ROMA/CNR_Research/HYPERNETS_Validation_Protocols/python_scripts')
import common_functions # 

def open_PRISMA(path_to_file):
    f = h5py.File(path_to_file, 'r')
    # reading name and value for root attributes (metadata contained in HDF5 root)
    # for attribute in f.attrs:
    #     print(attribute,f.attrs[attribute])
    # # reading names for all attributes (metadata) contained in HDF5 Groups
    # # specific method for reading the values shall be built depending by the specific metadata type (a single value, an array, a matrix, etc)
    # def printname(name):
    #     print(name)
    # f.visit(printname)

    prod_lvl = path_to_file.split('/')[-1].split('_')[1]

     # reading SWIR & VNIR datacubes
    swir = f['/HDFEOS/SWATHS/PRS_'+prod_lvl+'_HCO/Data Fields/SWIR_Cube']
    vnir = f['/HDFEOS/SWATHS/PRS_'+prod_lvl+'_HCO/Data Fields/VNIR_Cube']
    
    if prod_lvl == 'L1':
    	# get geolocation info ----
    	lat_swir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_SWIR']
    	lon_swir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR']
    	
    	lat_vnir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR']
    	lon_vnir = f['/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR']
    elif prod_lvl[1] == '2':
	    # get geolocation info ----
	    lat_swir = f['/HDFEOS/SWATHS/PRS_'+prod_lvl+'_HCO/Geolocation Fields/Latitude']
	    lon_swir = f['/HDFEOS/SWATHS/PRS_'+prod_lvl+'_HCO/Geolocation Fields/Longitude']
	    
	    lat_vnir = f['/HDFEOS/SWATHS/PRS_'+prod_lvl+'_HCO/Geolocation Fields/Latitude']
	    lon_vnir = f['/HDFEOS/SWATHS/PRS_'+prod_lvl+'_HCO/Geolocation Fields/Longitude']
    
    # :List_Cw_Vnir = 0.0f, 0.0f, 0.0f, 972.65137f, 962.2815f, 951.3803f, 939.87714f, 929.4021f, 919.1934f, 908.6615f, 898.00415f, 887.28357f, 876.6585f, 865.9668f, 855.1996f, 844.44434f, 833.7686f, 823.16296f, 812.55695f, 801.95374f, 791.3786f, 780.9292f, 770.54535f, 760.1139f, 749.7475f, 739.4342f, 729.26086f, 719.1886f, 709.01324f, 699.11145f, 689.43713f, 679.4886f, 669.83234f, 660.27905f, 650.8059f, 641.34863f, 632.146f, 623.2119f, 614.18665f, 605.40576f, 596.4894f, 587.8376f, 579.3654f, 571.01636f, 562.75f, 554.5778f, 546.493f, 538.49963f, 530.67944f, 522.92865f, 515.1879f, 507.67554f, 500.14948f, 492.71198f, 485.42203f, 478.18848f, 470.96112f, 463.7431f, 456.38937f, 449.04562f, 441.67093f, 434.32034f, 426.9796f, 419.3852f, 411.33044f, 402.45462f; // float
    # :List_Cw_Vnir_Flags = 0UB, 0UB, 0UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB; // ubyte
    # :List_Cw_Swir = 2496.877f, 2490.0305f, 2483.593f, 2476.7944f, 2469.4182f, 2462.8157f, 2456.3328f, 2448.9263f, 2442.1904f, 2435.3296f, 2428.4036f, 2420.9905f, 2414.159f, 2407.3418f, 2399.804f, 2392.8442f, 2385.8118f, 2378.5127f, 2371.339f, 2364.3738f, 2357.026f, 2349.578f, 2342.5918f, 2335.2395f, 2327.6096f, 2320.6433f, 2312.9158f, 2305.505f, 2298.3381f, 2290.5762f, 2283.2778f, 2275.7732f, 2268.0452f, 2260.6309f, 2252.8518f, 2245.2166f, 2237.651f, 2229.7563f, 2222.1958f, 2214.345f, 2206.6006f, 2198.8901f, 2190.8389f, 2183.1863f, 2175.0745f, 2167.2554f, 2159.2913f, 2151.1292f, 2143.2288f, 2135.2388f, 2127.097f, 2118.9587f, 2110.8193f, 2102.5515f, 2094.4104f, 2086.1042f, 2077.7546f, 2069.542f, 2061.1409f, 2052.7603f, 2044.4255f, 2035.9985f, 2027.479f, 2019.0529f, 2010.4133f, 2001.8461f, 1993.2917f, 1984.5515f, 1975.7843f, 1967.0625f, 1958.3737f, 1949.6423f, 1940.834f, 1931.9949f, 1923.0927f, 1914.051f, 1904.6321f, 1895.8495f, 1886.8143f, 1878.4668f, 1868.0199f, 1859.3008f, 1850.2821f, 1841.0529f, 1831.7446f, 1822.1523f, 1812.8234f, 1803.3687f, 1793.6924f, 1784.4453f, 1774.9119f, 1765.2563f, 1755.5525f, 1745.933f, 1736.2584f, 1726.4401f, 1716.6006f, 1706.811f, 1697.0128f, 1687.1749f, 1677.127f, 1666.9692f, 1656.7743f, 1646.9758f, 1636.8574f, 1626.7844f, 1616.587f, 1606.2463f, 1595.9828f, 1585.6343f, 1575.396f, 1565.1245f, 1554.5829f, 1544.0325f, 1533.567f, 1523.0099f, 1512.4164f, 1501.7822f, 1491.2006f, 1480.6229f, 1469.7032f, 1459.0706f, 1448.9243f, 1438.1736f, 1427.1089f, 1416.3131f, 1405.3914f, 1394.5273f, 1383.0116f, 1372.7213f, 1360.8453f, 1349.6013f, 1338.9486f, 1328.0923f, 1317.0206f, 1306.0494f, 1295.1996f, 1284.2821f, 1273.3088f, 1262.3367f, 1250.7979f, 1240.0587f, 1229.0012f, 1217.698f, 1207.1168f, 1196.1769f, 1185.4003f, 1174.5281f, 1163.4866f, 1152.4738f, 1141.8761f, 1131.145f, 1120.497f, 1109.7427f, 1099.1104f, 1088.5919f, 1078.0413f, 1067.6122f, 1057.3828f, 1047.4341f, 1037.7673f, 1028.8129f, 1017.9914f, 1008.17413f, 998.38153f, 988.4178f, 978.74365f, 969.39557f, 959.5276f, 951.01624f, 942.96277f, 0.0f, 0.0f; // float
    # :List_Cw_Swir_Flags = 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 1UB, 0UB, 0UB; // ubyte
    # :List_Fwhm_Vnir = 0.0f, 0.0f, 0.0f, 12.91153f, 12.983634f, 13.522025f, 13.167123f, 12.756155f, 12.91504f, 13.082645f, 13.121019f, 13.102607f, 13.073585f, 13.15429f, 13.131966f, 13.061808f, 12.990087f, 12.972249f, 12.945051f, 12.928755f, 12.8288965f, 12.756103f, 12.733164f, 12.772942f, 12.617298f, 12.538957f, 12.411063f, 12.352824f, 12.427184f, 11.981684f, 12.081189f, 11.965828f, 11.833743f, 11.718903f, 11.636767f, 11.532773f, 11.15188f, 11.233146f, 11.043023f, 10.954897f, 10.797392f, 10.548604f, 10.398883f, 10.308916f, 10.17477f, 10.057176f, 9.988766f, 9.784185f, 9.667365f, 9.637314f, 9.49066f, 9.368525f, 9.296826f, 9.203521f, 9.028441f, 8.951238f, 8.956281f, 8.959137f, 9.080744f, 9.1460905f, 9.219894f, 9.25125f, 9.402363f, 9.834814f, 10.555419f, 11.413751f; // float
    # :List_Fwhm_Swir = 9.530614f, 9.036248f, 8.984794f, 9.680815f, 9.149814f, 8.90354f, 9.453581f, 9.431822f, 8.7873745f, 9.40067f, 9.464742f, 9.691836f, 8.97784f, 9.704579f, 9.6363125f, 9.286388f, 9.706244f, 9.633341f, 9.516845f, 9.51342f, 9.954835f, 9.567097f, 9.620664f, 10.056107f, 9.693366f, 9.764342f, 10.139665f, 9.672169f, 10.0234375f, 10.004962f, 9.815104f, 10.233932f, 9.899899f, 10.246022f, 10.175902f, 10.069694f, 10.317282f, 10.290884f, 10.143012f, 10.389603f, 10.067344f, 10.552496f, 10.316623f, 10.305227f, 10.659013f, 10.189168f, 10.820628f, 10.553013f, 10.528742f, 10.707762f, 10.627058f, 10.801973f, 10.80722f, 10.83661f, 10.779761f, 11.042035f, 10.867285f, 11.108671f, 10.901592f, 11.005702f, 10.930906f, 11.262461f, 11.046657f, 11.28012f, 11.19699f, 11.3805485f, 11.212891f, 11.588905f, 11.374359f, 11.525665f, 11.419775f, 11.483898f, 11.789079f, 11.631749f, 11.955762f, 11.574869f, 12.524459f, 11.542392f, 11.567157f, 12.545796f, 11.145657f, 12.099602f, 12.396975f, 12.4054985f, 12.242381f, 12.863776f, 12.232095f, 12.296617f, 12.525058f, 12.494041f, 12.283826f, 12.422597f, 12.856553f, 12.918615f, 12.6999445f, 12.575837f, 12.71506f, 13.016736f, 13.12668f, 12.998398f, 12.912523f, 13.3708515f, 12.585966f, 13.177498f, 13.253216f, 13.306338f, 13.734705f, 13.48711f, 13.622092f, 13.63427f, 13.571609f, 13.747836f, 13.862521f, 13.741892f, 13.91596f, 13.901828f, 14.005367f, 14.092263f, 13.980284f, 14.168185f, 14.241516f, 13.700014f, 13.85956f, 14.6774235f, 14.082534f, 14.2978f, 14.388855f, 14.213633f, 14.839593f, 14.532085f, 15.091254f, 14.479001f, 14.438772f, 14.679442f, 14.564553f, 14.493105f, 14.449337f, 14.524426f, 14.553549f, 14.9414215f, 14.7266865f, 14.364552f, 14.746681f, 14.385526f, 14.561595f, 14.346457f, 14.386706f, 14.303056f, 14.665467f, 14.322483f, 14.102333f, 14.347128f, 14.031915f, 14.408157f, 14.059064f, 14.077611f, 13.991558f, 13.864974f, 13.65352f, 13.500643f, 13.309181f, 13.041258f, 12.960461f, 12.43994f, 12.6957f, 12.338166f, 12.208103f, 12.108707f, 12.050459f, 10.962302f, 10.920637f, 0.0f, 0.0f; // float
    
    # :ScaleFactor_Vnir = 100.0f; // float
    # :Offset_Vnir = 0.0f; // float
    # :ScaleFactor_Swir = 100.0f; // float
    # :Offset_Swir = 0.0f; // float
    
    ScaleFactor_vnir = 100
    Offset_vnir = 0
    wl_vnir = [0.0,0.0,0.0,972.65137,962.2815,951.3803,939.87714,929.4021,919.1934,908.6615,898.00415,887.28357,876.6585,865.9668,855.1996,844.44434,833.7686,823.16296,812.55695,801.95374,791.3786,780.9292,770.54535,760.1139,749.7475,739.4342,729.26086,719.1886,709.01324,699.11145,689.43713,679.4886,669.83234,660.27905,650.8059,641.34863,632.146,623.2119,614.18665,605.40576,596.4894,587.8376,579.3654,571.01636,562.75,554.5778,546.493,538.49963,530.67944,522.92865,515.1879,507.67554,500.14948,492.71198,485.42203,478.18848,470.96112,463.7431,456.38937,449.04562,441.67093,434.32034,426.9796,419.3852,411.33044,402.45462]
    wl_vnir = np.array(wl_vnir)
    
    ScaleFactor_swir = 100
    Offset_swir = 0
    wl_swir = [2496.877, 2490.0305, 2483.593, 2476.7944, 2469.4182, 2462.8157, 2456.3328, 2448.9263, 2442.1904, 2435.3296, 2428.4036, 2420.9905, 2414.159, 2407.3418, 2399.804, 2392.8442, 2385.8118, 2378.5127, 2371.339, 2364.3738, 2357.026, 2349.578, 2342.5918, 2335.2395, 2327.6096, 2320.6433, 2312.9158, 2305.505, 2298.3381, 2290.5762, 2283.2778, 2275.7732, 2268.0452, 2260.6309, 2252.8518, 2245.2166, 2237.651, 2229.7563, 2222.1958, 2214.345, 2206.6006, 2198.8901, 2190.8389, 2183.1863, 2175.0745, 2167.2554, 2159.2913, 2151.1292, 2143.2288, 2135.2388, 2127.097, 2118.9587, 2110.8193, 2102.5515, 2094.4104, 2086.1042, 2077.7546, 2069.542, 2061.1409, 2052.7603, 2044.4255, 2035.9985, 2027.479, 2019.0529, 2010.4133, 2001.8461, 1993.2917, 1984.5515, 1975.7843, 1967.0625, 1958.3737, 1949.6423, 1940.834, 1931.9949, 1923.0927, 1914.051, 1904.6321, 1895.8495, 1886.8143, 1878.4668, 1868.0199, 1859.3008, 1850.2821, 1841.0529, 1831.7446, 1822.1523, 1812.8234, 1803.3687, 1793.6924, 1784.4453, 1774.9119, 1765.2563, 1755.5525, 1745.933, 1736.2584, 1726.4401, 1716.6006, 1706.811, 1697.0128, 1687.1749, 1677.127, 1666.9692, 1656.7743, 1646.9758, 1636.8574, 1626.7844, 1616.587, 1606.2463, 1595.9828, 1585.6343, 1575.396, 1565.1245, 1554.5829, 1544.0325, 1533.567, 1523.0099, 1512.4164, 1501.7822, 1491.2006, 1480.6229, 1469.7032, 1459.0706, 1448.9243, 1438.1736, 1427.1089, 1416.3131, 1405.3914, 1394.5273, 1383.0116, 1372.7213, 1360.8453, 1349.6013, 1338.9486, 1328.0923, 1317.0206, 1306.0494, 1295.1996, 1284.2821, 1273.3088, 1262.3367, 1250.7979, 1240.0587, 1229.0012, 1217.698, 1207.1168, 1196.1769, 1185.4003, 1174.5281, 1163.4866, 1152.4738, 1141.8761, 1131.145, 1120.497, 1109.7427, 1099.1104, 1088.5919, 1078.0413, 1067.6122, 1057.3828, 1047.4341, 1037.7673, 1028.8129, 1017.9914, 1008.17413, 998.38153, 988.4178, 978.74365, 969.39557, 959.5276, 951.01624, 942.96277, 0.0, 0.0]
    wl_swir = np.array(wl_swir)

    vnir = vnir[:].astype(float)/ScaleFactor_vnir-Offset_vnir
    swir = swir[:].astype(float)/ScaleFactor_swir-Offset_swir
    lat_vnir = lat_vnir[:,:]
    lon_vnir = lon_vnir[:,:]
    lat_swir = lat_swir[:,:]
    lon_swir = lon_swir[:,:]

    return vnir, swir, lat_vnir, lon_vnir, lat_swir, lon_swir, wl_vnir, wl_swir

def stack_rgb(cube,r_wl,g_wl,b_wl,ls_pct=0):
    
    from skimage import exposure
    
    b_ind = np.argmin(np.abs(wl_vnir-b_wl))
    g_ind = np.argmin(np.abs(wl_vnir-g_wl))
    r_ind = np.argmin(np.abs(wl_vnir-r_wl))

    red   = cube[:,r_ind,:]/cube.max()
    green = cube[:,g_ind,:]/cube.max()
    blue  = cube[:,b_ind,:]/cube.max()

    stackedRGB = np.stack((red,green,blue),axis=2)
    
    # linear contrast stretch
    pLow, pHigh = np.percentile(stackedRGB[~np.isnan(stackedRGB)], (ls_pct,100-ls_pct))
    stackedRGB = exposure.rescale_intensity(stackedRGB, in_range=(pLow,pHigh))

    return stackedRGB

def plot_geo(img,lats,lons,lat_plot_limits=[-90,90],lon_plot_limits=[-180,180],mosaic_flag=False,one_channel=False):
	# solution taken from: 
	# https://stackoverflow.com/questions/41389335/how-to-plot-geolocated-rgb-data-faster-using-python-basemap
	# https://github.com/matplotlib/matplotlib/issues/4277

    if one_channel:
    	img = img.transpose()
    else:
    	img[:,:,0] = img[:,:,0].transpose()
    	img[:,:,1] = img[:,:,1].transpose()
    	img[:,:,2] = img[:,:,2].transpose()

    lats = lats.transpose()
    lons = lons.transpose()

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    lat_0 = np.mean(lats)
    lat_range = [np.min(lats), np.max(lats)]

    lon_0 = np.mean(lons)
    lon_range = [np.min(lons), np.max(lons)]

    map_kwargs = dict(projection='cyl', resolution='i',
                      llcrnrlat=lat_plot_limits[0], urcrnrlat=lat_plot_limits[1],
                      llcrnrlon=lon_plot_limits[0], urcrnrlon=lon_plot_limits[1],
                      lat_0=lat_0, lon_0=lon_0)
    if mosaic_flag:
        plt.gcf()
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    m = Basemap(**map_kwargs)

    if one_channel:
        m.pcolormesh(lons, lats, img, latlon=True,label='_nolegend_')
    else:
        mesh_rgb = img[:-1,:-1,:] # pcolormesh cut off the last row and column. This fixs it!
        mesh_rgb.shape
        colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)

        # ADDED THIS LINE
        colorTuple = np.insert(colorTuple,3,1.0,axis=1)

        # What you put in for the image doesn't matter because of the color mapping
        m.pcolormesh(lons, lats, lons, latlon=True,color=colorTuple,label='_nolegend_')

    meridianinterval = np.linspace(lon_plot_limits[0],lon_plot_limits[1],5) # 5 = number of "ticks"
    m.drawmeridians(meridianinterval,labels=[0,0,0,1], dashes=[6,900], color='w',label='_nolegend_')
    parallelinterval = np.linspace(lat_plot_limits[0],lat_plot_limits[1],5) # 5 = number of "ticks"
    m.drawparallels(parallelinterval,labels=[1,0,0,0], dashes=[6,900], color='w',label='_nolegend_')    
    m.drawcoastlines()
    # m.drawcountries()

    plt.show()
#%%


#%% Open file L1
path_to_image = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Images/PRISMA'
# file_name = 'PRS_L1_STD_OFFL_20200208101029_20200208101033_0001.he5'
file_name = 'PRS_L1_STD_OFFL_20200116101454_20200116101458_0001.he5'
path_to_file = os.path.join(path_to_image,file_name)

vnir, swir, lat_vnir, lon_vnir, lat_swir, lon_swir, wl_vnir, wl_swir = open_PRISMA(path_to_file)

# list the structure of SWIR data
swir.shape
# list the structure of VNIR data
vnir.shape
# print portions of the SWIR and VNIR bands
# band 0
swir[0:9,0,0:9]
vnir[0:9,0,0:9]
# band 170
swir[990:999,170,990:999]
# band 60
vnir[990:999,60,990:999]            

# from prismaread/R/convert_prisma.R
#  # Get wavelengths and fwhms ----
#   wl_vnir    <- hdf5r::h5attr(,"List_Cw_Vnir")
#   order_vnir <- order(wl_vnir)
#   wl_vnir <- wl_vnir[order_vnir]


#%%
plt.figure()
data = vnir[:,3,:]
lats = lat_vnir[:,:]
lons = lon_vnir[:,:]

m = Basemap(llcrnrlat=lats.min(),urcrnrlat=lats.max(),\
	        	llcrnrlon=lons.min(),urcrnrlon=lons.max(), resolution='l')
# m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(-180., 181., 45.), labels=[0, 0, 0, 1])
m.imshow(data,cmap='rainbow')
#%%
r_wl = 620
g_wl = 550
b_wl = 460
cube = vnir
stackedRGB = stack_rgb(cube,r_wl,g_wl,b_wl,ls_pct=5)
# plt.figure()
# m = Basemap(llcrnrlat=45.2,urcrnrlat=45.8,\
# 	        	llcrnrlon=lons.min(),urcrnrlon=lons.max(), resolution='l',projection='cyl')
# # m.drawcoastlines(linewidth=0.5)
# meridianinterval = np.linspace(lons.min(),lons.max(),5) # 5 = number of "ticks"
# m.drawmeridians(meridianinterval,labels=[0,0,0,1], dashes=[6,900], color='w')
# parallelinterval = np.linspace(lats.min(),lats.max(),5) # 5 = number of "ticks"
# m.drawparallels(parallelinterval,labels=[1,0,0,0], dashes=[6,900], color='w')
# m.imshow(stackedRGB)
# plt.plot(lons[820,133],lats[820,133],'ob')
# plt.plot(lons[999,999],lats[999,999],'or')
# plt.plot(lons[100,800],lats[100,800],'og')
# plt.title('PRISMA RGB Image')

sp1_x = 338
sp1_y = 601

sp2_x = 800
sp2_y = 560

sp3_x = 900
sp3_y = 100

sp4_x = 349
sp4_y = 717

fig = plt.figure()
plt.suptitle(file_name)
plt.subplot(1,2,1)
plt.imshow(stackedRGB,origin='lower')
plt.plot(sp1_x,sp1_y,'ok')
plt.plot(sp2_x,sp2_y,'or')
plt.plot(sp3_x,sp3_y,'ob')
plt.plot(sp4_x,sp4_y,'og')


#% plot spectrum
sp1_vnir = vnir[sp1_y,3:,sp1_x]
sp1_swir = swir[sp1_y,:-2,sp1_x]

sp2_vnir = vnir[sp2_y,3:,sp2_x]
sp2_swir = swir[sp2_y,:-2,sp2_x]

sp3_vnir = vnir[sp3_y,3:,sp3_x]
sp3_swir = swir[sp3_y,:-2,sp3_x]

sp4_vnir = vnir[sp4_y,3:,sp4_x]
sp4_swir = swir[sp4_y,:-2,sp4_x]

plt.subplot(1,2,2)
plt.plot(wl_vnir[3:],sp1_vnir,'.-k')
plt.plot(wl_swir[:-2],sp1_swir,'.-k',label='_nolegend_')

plt.plot(wl_vnir[3:],sp2_vnir,'.-r')
plt.plot(wl_swir[:-2],sp2_swir,'.-r',label='_nolegend_')

plt.plot(wl_vnir[3:],sp3_vnir,'.-b')
plt.plot(wl_swir[:-2],sp3_swir,'.-b',label='_nolegend_')

plt.plot(wl_vnir[3:],sp4_vnir,'.-g')
plt.plot(wl_swir[:-2],sp4_swir,'.-g',label='_nolegend_')

plt.legend(['City','Lagoon','Sea','Vegetation'])

plt.xlabel('Wavelength (nm)')
plt.ylabel('TOA Radiance ($W/m^2/sr/{\mu}m$)')

#%% Open file L2D
path_to_image = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Images/PRISMA'
file_name = 'PRS_L2D_STD_20200220101716_20200220101720_0001.he5'
path_to_file = os.path.join(path_to_image,file_name)

vnir, swir, lat_vnir, lon_vnir, lat_swir, lon_swir, wl_vnir, wl_swir = open_PRISMA(path_to_file)

#%%
plt.figure()
band_wl = 560
band_ind = np.argmin(np.abs(wl_vnir-band_wl))
data = vnir[:,band_ind,:]
lats = lat_vnir[:,:]
lons = lon_vnir[:,:]

plot_geo(data,lats,lons,lat_plot_limits=[-90,90],lon_plot_limits=[-180,180],one_channel=True)
# m = Basemap(llcrnrlat=lats.min(),urcrnrlat=lats.max(),\
# 	        	llcrnrlon=lons.min(),urcrnrlon=lons.max(), resolution='l')
# # m.drawcoastlines(linewidth=0.5)
# m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(-180., 181., 45.), labels=[0, 0, 0, 1])
# m.imshow(data,cmap='rainbow')
#%%
r_wl = 620
g_wl = 550
b_wl = 460
stackedRGB = stack_rgb(vnir,r_wl,g_wl,b_wl)

sp1_x = 142
sp1_y = 630

sp2_x = 148
sp2_y = 630

sp3_x = 483
sp3_y = 909

sp4_x = 698
sp4_y = 530

fig = plt.figure()
plt.suptitle(file_name)
plt.subplot(1,2,1)
plt.imshow(stackedRGB,origin='lower')
plt.plot(sp1_x,sp1_y,'ok')
plt.plot(sp2_x,sp2_y,'or')
plt.plot(sp3_x,sp3_y,'ob')
plt.plot(sp4_x,sp4_y,'og')


#% plot spectrum
sp1_vnir = vnir[sp1_y,3:,sp1_x]
sp1_swir = swir[sp1_y,:-2,sp1_x]

sp2_vnir = vnir[sp2_y,3:,sp2_x]
sp2_swir = swir[sp2_y,:-2,sp2_x]

sp3_vnir = vnir[sp3_y,3:,sp3_x]
sp3_swir = swir[sp3_y,:-2,sp3_x]

sp4_vnir = vnir[sp4_y,3:,sp4_x]
sp4_swir = swir[sp4_y,:-2,sp4_x]

plt.subplot(1,2,2)
plt.plot(wl_vnir[3:],sp1_vnir,'.-k')
plt.plot(wl_swir[:-2],sp1_swir,'.-k',label='_nolegend_')

plt.plot(wl_vnir[3:],sp2_vnir,'.-r')
plt.plot(wl_swir[:-2],sp2_swir,'.-r',label='_nolegend_')

plt.plot(wl_vnir[3:],sp3_vnir,'.-b')
plt.plot(wl_swir[:-2],sp3_swir,'.-b',label='_nolegend_')

plt.plot(wl_vnir[3:],sp4_vnir,'.-g')
plt.plot(wl_swir[:-2],sp4_swir,'.-g',label='_nolegend_')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (%)')

    
#%% Open file L2C
path_to_image = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Images/PRISMA'
file_name = 'PRS_L2C_STD_20200116101454_20200116101458_0001.he5'
path_to_file = os.path.join(path_to_image,file_name)

vnir, swir, lat_vnir, lon_vnir, lat_swir, lon_swir, wl_vnir, wl_swir = open_PRISMA(path_to_file)

#%%
band_wl = 440
band_ind = np.argmin(np.abs(wl_vnir-band_wl))
data = vnir[:,band_ind,:]
lats = lat_vnir[:,:]
lons = lon_vnir[:,:]

plot_geo(data,lats,lons,lat_plot_limits=[37,47],lon_plot_limits=[5,20],one_channel=True)

# m = Basemap(llcrnrlat=lats.min(),urcrnrlat=lats.max(),\
# 	        	llcrnrlon=lons.min(),urcrnrlon=lons.max(), resolution='l')
# # m.drawcoastlines(linewidth=0.5)
# m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(-180., 181., 45.), labels=[0, 0, 0, 1])
# m.imshow(data,cmap='rainbow')
plt.figure()
plt.imshow(data)
#%%
r_wl = 620
g_wl = 550
b_wl = 460
stackedRGB = stack_rgb(vnir,r_wl,g_wl,b_wl,ls_pct=10)
# plt.figure()
# m = Basemap(llcrnrlat=45.2,urcrnrlat=45.8,\
# 	        	llcrnrlon=lons.min(),urcrnrlon=lons.max(), resolution='l',projection='cyl')
# # m.drawcoastlines(linewidth=0.5)
# meridianinterval = np.linspace(lons.min(),lons.max(),5) # 5 = number of "ticks"
# m.drawmeridians(meridianinterval,labels=[0,0,0,1], dashes=[6,900], color='w')
# parallelinterval = np.linspace(lats.min(),lats.max(),5) # 5 = number of "ticks"
# m.drawparallels(parallelinterval,labels=[1,0,0,0], dashes=[6,900], color='w')
# m.imshow(stackedRGB)
# plt.plot(lons[820,133],lats[820,133],'ob')
# plt.plot(lons[999,999],lats[999,999],'or')
# plt.plot(lons[100,800],lats[100,800],'og')
# plt.title('PRISMA RGB Image')

sp1_x = 142
sp1_y = 630

sp2_x = 148
sp2_y = 630

sp3_x = 483
sp3_y = 909

sp4_x = 698
sp4_y = 530

fig = plt.figure()
plt.suptitle(file_name)
plt.subplot(1,2,1)
plt.imshow(stackedRGB,origin='lower')
plt.plot(sp1_x,sp1_y,'ok')
plt.plot(sp2_x,sp2_y,'or')
plt.plot(sp3_x,sp3_y,'ob')
plt.plot(sp4_x,sp4_y,'og')


#% plot spectrum
sp1_vnir = vnir[sp1_y,3:,sp1_x]
sp1_swir = swir[sp1_y,:-2,sp1_x]

sp2_vnir = vnir[sp2_y,3:,sp2_x]
sp2_swir = swir[sp2_y,:-2,sp2_x]

sp3_vnir = vnir[sp3_y,3:,sp3_x]
sp3_swir = swir[sp3_y,:-2,sp3_x]

sp4_vnir = vnir[sp4_y,3:,sp4_x]
sp4_swir = swir[sp4_y,:-2,sp4_x]


plt.subplot(1,2,2)
plt.plot(wl_vnir[3:],sp1_vnir,'.-k')
plt.plot(wl_swir[:-2],sp1_swir,'.-k',label='_nolegend_')

plt.plot(wl_vnir[3:],sp2_vnir,'.-r')
plt.plot(wl_swir[:-2],sp2_swir,'.-r',label='_nolegend_')

plt.plot(wl_vnir[3:],sp3_vnir,'.-b')
plt.plot(wl_swir[:-2],sp3_swir,'.-b',label='_nolegend_')

plt.plot(wl_vnir[3:],sp4_vnir,'.-g')
plt.plot(wl_swir[:-2],sp4_swir,'.-g',label='_nolegend_')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (%)')    
#%%
plot_geo(stackedRGB,lats,lons,lat_plot_limits=[37,47],lon_plot_limits=[5,20],one_channel=False)

#%% Open file L1
path_to_image = '/Users/javier.concha/Desktop/Javier/2019_Roma/CNR_Research/Images/PRISMA//dati_PRISMA_ADR/'


file1_name = 'PRS_L1_STD_OFFL_20200220101707_20200220101711_0001.he5'
file2_name = 'PRS_L1_STD_OFFL_20200220101711_20200220101716_0001.he5'
file3_name = 'PRS_L1_STD_OFFL_20200220101716_20200220101720_0001.he5'
file4_name = 'PRS_L1_STD_OFFL_20200220101724_20200220101729_0001.he5'
file5_name = 'PRS_L1_STD_OFFL_20200220101729_20200220101733_0001.he5'


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
#%%
path_to_file = os.path.join(path_to_image,file2_name)
vnir, swir, lat_vnir, lon_vnir, lat_swir, lon_swir, wl_vnir, wl_swir = open_PRISMA(path_to_file)

r_wl = 620
g_wl = 550
b_wl = 460
stackedRGB = stack_rgb(vnir,r_wl,g_wl,b_wl,ls_pct=10)
plot_geo(stackedRGB,lat_vnir,lon_vnir,lat_plot_limits=[44,46],lon_plot_limits=[12,14],mosaic_flag=False,one_channel=False)
#%%
plt.plot(12.59883333,45.02,'ro')
plt.plot(12.75733333,45.1725,'go')
plt.plot(12.50483333,45.31216667,'bo')
plt.plot(12.47116667,45.37983333,'ko')



plt.legend(['St 008','St 009','St 010','St 011'])

from matplotlib.lines import Line2D

colors = ['red', 'green', 'blue','black']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='None',marker='o') for c in colors]
labels = ['St 008','St 009','St 010','St 011']
plt.legend(lines, labels)

#%%
r1, c1 = common_functions.find_row_column_from_lat_lon(lat_vnir,lon_vnir,45.02,12.59883333)
r2, c2 = common_functions.find_row_column_from_lat_lon(lat_vnir,lon_vnir,45.1725,12.75733333)
#%%
plt.figure()
#% plot spectrum
sp1_vnir = vnir[r1,3:,c1]
sp1_swir = swir[r1,:-2,c1]

sp2_vnir = vnir[r2,3:,c2]
sp2_swir = swir[r2,:-2,c2]

plt.plot(wl_vnir[3:],sp1_vnir,'.-r')
plt.plot(wl_swir[:-2],sp1_swir,'.-r',label='_nolegend_')

plt.plot(wl_vnir[3:],sp2_vnir,'.-g')
plt.plot(wl_swir[:-2],sp2_swir,'.-g',label='_nolegend_')

plt.xlabel('Wavelength (nm)')
plt.ylabel('TOA Radiance ($W/m^2/sr/{\mu}m$)')
plt.legend(['St 008','St 009'])
#%%
plt.figure()
plt.imshow(stackedRGB)
plt.plot(r2,c2,'go')