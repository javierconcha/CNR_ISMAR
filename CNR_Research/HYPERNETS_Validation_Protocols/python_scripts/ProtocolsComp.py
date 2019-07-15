#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:04:37 2019

@author: Javier A. Concha
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
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
#%% Plotting
x = np.array([0.1, 0.2, 0.3, 0.4],dtype=np.float32)
y = np.array([-0.1, 0.21, 0.39, 0.45],dtype=np.float32)
e_x = np.array([0.01, 0.03, 0.01, 0.05],dtype=np.float32)
e_y = np.array([0.05, 0.07, 0.08, 0.1],dtype=np.float32)

plt.figure
plt.errorbar(x, y, xerr=e_x, yerr=e_y, fmt='or')
plt.axis([-0.5, 1.0, -0.5, 1.0])
plt.gca().set_aspect('equal', adjustable='box')
# plot 1:1 line
xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()
plt.plot([xmin,xmax],[ymin, ymax],'--k')

# Generated linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
line = slope*np.array([xmin,xmax],dtype=np.float32)+intercept
plt.plot([xmin,xmax], line)
plt.legend(['1:1','Regression Line'])
plt.xlabel('$L^{PRS}_{WN}$')
plt.ylabel('$L^{OLCI}_{WN}$')
if (xmin<0 or ymin<0):
    plt.plot([xmin,xmax],[0, 0],'--k',linewidth = 0.7)
    
plt.show()