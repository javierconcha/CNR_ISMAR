#!/usr/bin/env python3
# coding: utf-8
"""
Created on Fri Oct 11 15:49:25 2019

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

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

# blue_star = mlines.Line2D([], [], color='blue', marker='*', linestyle='None',
#                           markersize=10, label='Blue stars')
# red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
#                           markersize=10, label='Red squares')
# purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
#                           markersize=10, label='Purple triangles')

# plt.legend(handles=[blue_star, red_square, purple_triangle])

# plt.show()
#%% S2A
fig = plt.figure()
ax = fig.add_subplot(111) # 

legend_Venise = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='r', label='Venise')
legend_Gloria = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='g', label='Gloria')
legend_Galata_Platform = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='b', label='Galata_Platform')
legend_Helsinki_Lighthouse = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='m', label='Helsinki_Lighthouse')
legend_Gustav_Dalen_Tower = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='c', label='Gustav_Dalen_Tower')
legend_Lake_Erie = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='brown', label='Lake_Erie')
legend_LISCO = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='lime', label='LISCO')
legend_Palgrunden = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='dodgerblue', label='Palgrunden')
legend_Thornton_C_power = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='crimson', label='Thornton_C_power')
legend_USC_SEAPRISM = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='chocolate', label='USC_SEAPRISM')
legend_USC_SEAPRISM_2 = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='darkviolet', label='USC_SEAPRISM_2')
legend_WaveCIS_Site_CSI_6 = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='turquoise', label='WaveCIS_Site_CSI_6')

leg1 = ax.legend(handles=[legend_Venise,legend_Galata_Platform,legend_Helsinki_Lighthouse,legend_Gustav_Dalen_Tower,\
legend_Lake_Erie,legend_LISCO,legend_Palgrunden,legend_Thornton_C_power,legend_USC_SEAPRISM_2,legend_WaveCIS_Site_CSI_6 ],\
loc='upper center',bbox_to_anchor=(0.5, 1.0),ncol=3,frameon=False,fontsize=12)

legend_BW06 = mlines.Line2D([],[], linestyle='None', markersize=10, marker='x', color='k', label='BW06    ')
legend_IPK19 = mlines.Line2D([],[], linestyle='None', markersize=10, marker='+', color='k', label='IPK19    ')
legend_V10 = mlines.Line2D([],[], linestyle='None', markersize=10, marker='o', color='k', markerfacecolor='none', label='V19')

leg2 = ax.legend(handles=[legend_BW06, legend_IPK19, legend_V10],loc='upper center',\
                 bbox_to_anchor=(0.5, 1.11),ncol=3,frameon=False,fontsize=12)

ax.add_artist(leg1)

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()

#%% S3A/OLCI
fig = plt.figure()
ax = fig.add_subplot(111) # 

legend_Venise = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='r', label='Venise')
legend_Gloria = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='g', label='Gloria')
legend_Galata_Platform = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='b', label='Galata_Platform')
legend_Helsinki_Lighthouse = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='m', label='Helsinki_Lighthouse')
legend_Gustav_Dalen_Tower = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='c', label='Gustav_Dalen_Tower')

leg1 = ax.legend(handles=[legend_Venise,legend_Gloria,legend_Galata_Platform,legend_Helsinki_Lighthouse,legend_Gustav_Dalen_Tower],\
loc='upper center',bbox_to_anchor=(0.5, 1.0),ncol=3,frameon=False,fontsize=12)

legend_BW06 = mlines.Line2D([],[], linestyle='None', markersize=10, marker='x', color='k', label='BW06    ')
legend_Z09 = mlines.Line2D([],[], linestyle='None', markersize=10, marker='+', color='k', label='Z09    ')

leg2 = ax.legend(handles=[legend_BW06, legend_Z09],loc='upper center',\
                 bbox_to_anchor=(0.5, 1.11),ncol=2,frameon=False,fontsize=12)

ax.add_artist(leg1)

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()

#%% S3A with color for all stations
fig = plt.figure()
ax = fig.add_subplot(111) # 

legend_Venise = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='r', label='Venise')
legend_Gloria = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='g', label='Gloria')
legend_Galata_Platform = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='b', label='Galata_Platform')
legend_Helsinki_Lighthouse = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='m', label='Helsinki_Lighthouse')
legend_Gustav_Dalen_Tower = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='c', label='Gustav_Dalen_Tower')
legend_All_Stations = mlines.Line2D([], [], linestyle='None', markersize=10, marker='s', color='k', label='All Stations')

leg1 = ax.legend(handles=[legend_Venise,legend_Gloria,legend_Galata_Platform,legend_Helsinki_Lighthouse,legend_Gustav_Dalen_Tower,legend_All_Stations],\
loc='upper center',bbox_to_anchor=(0.5, 1.0),ncol=3,frameon=False,fontsize=12)

legend_BW06 = mlines.Line2D([],[], linestyle='None', markersize=10, marker='x', color='k', label='BW06    ')
legend_Z09 = mlines.Line2D([],[], linestyle='None', markersize=10, marker='+', color='k', label='Z09    ')

leg2 = ax.legend(handles=[legend_BW06, legend_Z09],loc='upper center',\
                 bbox_to_anchor=(0.5, 1.11),ncol=2,frameon=False,fontsize=12)

ax.add_artist(leg1)

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()