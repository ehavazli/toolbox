#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
from ftplib import FTP

def main(argv):
    try:
        dl_folder = argv[1]
        year = argv[2]
        start_date = int(argv[3])
        end_date = int(argv[4])
        station_id = argv[5]
    except:
        print '''
    *******************************************

       Usage: dload_unr.py [donwload_folder] [year] [start_date] [end_date] [station_id]

              donwload_folder: location of the folder for the data to be downloaded
              year: year of the data to donwload
              start_date: Julian date of the first day to download
              end_date: Julian date of the last day to download
              station_id: 4 character station id

    *******************************************
    '''
        sys.exit(1)

    ftps = FTP('gneiss.nbmg.unr.edu')
    ftps.login()

    for i in xrange(start_date,end_date+1):
        ftps.cwd('/ultras_5min/kenv/'+year+'/'+str(i))
        filename = ftps.nlst('*'+station_id+'*.kenv')
        local_filename = os.path.join(dl_folder, filename[0])
        gFile = open(local_filename, "wb")
        ret1 = 'RETR '+str(filename[0])
        ftps.retrbinary(ret1, gFile.write)
        gFile.close()
        print 'Downloading file: '+ str(filename)
    ftps.quit()

if __name__ == '__main__':
    main(sys.argv[:])
