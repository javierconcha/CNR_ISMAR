#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:18:55 2019

@author: javier.concha
"""
#%%
import argparse

parser = argparse.ArgumentParser(description="This is example of using args")
parser.add_argument("-s", "--station" , nargs=1, help="The Aeronet OC station", required=True, type=str)
args = parser.parse_args()

station_name  = args.station[0]

print('The station name is: '+station_name)

