#!/bin/bash
# bash script to check upload of file to CMEMS server
# created by Javier Concha
# 2020-10-07

subject="Report OCTAC $(date)"
./check_upload.sh | mail -s "$subject" Javier.Concha@artov.ismar.cnr.it
