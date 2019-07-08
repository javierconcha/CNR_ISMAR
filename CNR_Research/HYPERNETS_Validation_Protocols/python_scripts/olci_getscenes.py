import os, sys
import csv
import datetime

def olci_get(date):
	if (date-datetime.datetime(2017,11,29)).total_seconds()<0:
		site="codarep.eumetsat.int"
	else:
		site="coda.eumetsat.int"
	year=str(date.year)
	month='0'*(2-len(str(date.month)))+str(date.month)	
	day='0'*(2-len(str(date.day)))+str(date.day)	
	datestart=year+'-'+month+'-'+day
	dateend=datestart

	os.system("./dhusget.sh -u marcobra -p cannicoda31 -d "+site+" -m Sentinel-3 -i OLCI -T OL_2_WFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00 -F"+"'timeliness:"+\
	'"Non Time Critical"'+"'"+" -C /home/Marco.Bracaglia/progetti/VGOCS/OLCI/csv_file/"+datestart.replace('-','')+"_"+dateend.replace('-','')+".csv -L /home/Marco.Bracaglia/dh/")
	#os.system('rm '+'/home/Marco.Bracaglia/progetti/VGOCS/OLCI/xml_file/'+datestart.replace(':','')+"_"+dateend.replace(':','')+'.xml')
	csv_f='/home/Marco.Bracaglia/progetti/VGOCS/OLCI/csv_file/'+datestart.replace('-','')+"_"+dateend.replace('-','')+'.csv'
	file_list=[]
	i=0
	with open(csv_f) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			if i==0 or i%2==0:
				file_olci=row[0]
				file_list.append(file_olci)
				i+=1
	print (file_list)
	os.system('rm '+csv_f)

	return file_list	


def olci_get_L1(date):
    if (date-datetime.datetime(2017,11,29)).total_seconds()<0:
            site="codarep.eumetsat.int"
    else:
            site="coda.eumetsat.int"
    year=str(date.year)
    month='0'*(2-len(str(date.month)))+str(date.month)
    day='0'*(2-len(str(date.day)))+str(date.day)
    datestart=year+'-'+month+'-'+day
    dateend=datestart
    #os.system("./dhusget.sh -u marcobra -p cannicoda31 -d coda.eumetsat.int -m Sentinel-3 -i OLCI -T OL_2_WFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00 -F"+' timeliness:"Non Time Critical"'\
    #+" -q /home/Marco.Bracaglia/progetti/VGOCS/OLCI/xml_file/"+datestart.replace(':','')+"_"+dateend.replace(':','')+".xml"+\
    #" -C /home/Marco.Bracaglia/progetti/VGOCS/OLCI/xls_file/"+datestart.replace(':','')+"_"+dateend.replace(':','')+".xls")
	print ("./dhusget.sh -u marcobra -p cannicoda31 -d "+site+" -m Sentinel-3 -i OLCI -T OL_1_EFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00"+" -F'timeliness:"+'"Non Time Critical"'+"'"+" -o product -O . -C pippo.csv")
	os.system("./dhusget.sh -u marcobra -p cannicoda31 -d "+site+" -m Sentinel-3 -i OLCI -T OL_1_EFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00"+" -F'timeliness:"+'"Non Time Critical"'+"'"+" -o product -O /home/Marco.Bracaglia/progetti/VGOCS/OLCI/L1_temp/ -C /home/Marco.Bracaglia/progetti/VGOCS/OLCI/csv_file/"+datestart.replace('-','')+"_"+dateend.replace('-','')+"_L1.csv")

    """os.system("./dhusget.sh -u marcobra -p cannicoda31 -d "+site+" -m Sentinel-3 -i OLCI -T OL_1_EFR___ -S "+datestart+"T00:00:00Z -E "+dateend+"T23:59:59Z -c 12.0,44.0:14.0,46.00 -F"+"'timeliness:"+\
    '"Non Time Critical"'+"'"+" -C /home/Marco.Bracaglia/progetti/VGOCS/OLCI/csv_file/"+datestart.replace('-','')+"_"+dateend.replace('-','')+".csv -o 'all' -O home/Marco.Bracaglia/progetti/VGOCS/OLCI/L1_temp/")"""
    #os.system('rm '+'/home/Marco.Bracaglia/progetti/VGOCS/OLCI/xml_file/'+datestart.replace(':','')+"_"+dateend.replace(':','')+'.xml')
    csv_f='/home/Marco.Bracaglia/progetti/VGOCS/OLCI/csv_file/'+datestart.replace('-','')+"_"+dateend.replace('-','')+'_L1.csv'
    file_list=[]
    i=0
    with open(csv_f) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                    if i==0 or i%2==0:
                            file_olci=row[0]
                            file_list.append(file_olci)
                            i+=1
    print (file_list)
	os.system('rm '+csv_f)
    return file_list

