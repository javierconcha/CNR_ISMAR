#!/bin/bash
# bash script to check upload of file to CMEMS server
# created by Javier Concha
# 2020-10-06
source ~/Processing/OC_PROC_EIS201912/s3olciProcessing/s3olciProcessing_ENV_on_BLADES_202007.sh

echo '----------------------------'
echo 'Checking NRT 1 day:'
ckdu_EiS202007.py -d $(date +%Y%m%d -d "1 days ago")  -m NRT -v
echo '----------------------------'
echo 'Checking DT 8 days:'
ckdu_EiS202007.py -d $(date +%Y%m%d -d "8 days ago") -s OLCI -m DT -v
echo '----------------------------'
echo 'Checking DT 20 days:'
ckdu_EiS202007.py -d $(date +%Y%m%d -d "20 days ago") -m DT -v
echo '----------------------------'
echo 'Checking DT 24 days:'
ckdu_EiS202007.py -d $(date +%Y%m%d -d "24 days ago") -m DT -v
echo '----------------------------'
echo 'Checking EUR 27 days:'
ckdu_EiS202007.py -d $(date +%Y%m%d -d "27 days ago") --EUR -v
