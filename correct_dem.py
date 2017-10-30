#! /usr/bin/env python

import os
import sys
from numpy import *
import h5py
from pysar import _readfile
from pysar import _writefile

def main(argv):
  try:
     dem_file = argv[1]
     dem_error = argv[2]
     operation = argv[3]
  except:
    print '''
    *******************************************
       
       Usage: correct_dem.py demFile geo_demErrorFile Operation(add or subtract)
       Example:
              correct_dem.py $DEMDIR/Socorro-30/Socorro_30.dem geo_DEM_error.h5 add
              correct_dem.py $DEMDIR/Socorro-30/Socorro_30.dem geo_DEM_error.h5 subtract

    *******************************************         
    '''
    sys.exit(1)

  

  dem, demrsc = _readfile.read_dem(dem_file)
  g = h5py.File(dem_error,'r')
  dset  = g['dem'].get('dem')
  dem_error = dset[0:dset.shape[0]]
  column,row = dem.shape
  xmax = (column - 1)
  ymax = (row - 1)
  DIR = os.getcwd()
  if operation == 'add':
        print 'Adding estimated errors to DEM'
  	sum = dem + dem_error
        _writefile.write_dem(sum,'DEM_+_error.dem')
        
        rsc_file = open('DEM_+_error.dem.rsc','w')
        for k in demrsc.keys():
            rsc_file.write(k+'	'+demrsc[k]+'\n')
        rsc_file.close
        
        date12_file=open('111111-222222_baseline.rsc','w')
        date12_file.write('P_BASELINE_TOP_ODR'+'     '+ '000')
        date12_file.close

  if operation == 'subtract':
        print 'Subtracting estimated errors from DEM'
	diff = dem - dem_error
        _writefile.write_dem(diff,'DEM_-_error.dem')
        
        rsc_file = open('DEM_-_error.dem.rsc','w')
        for k in demrsc.keys():
            rsc_file.write(k+'  '+demrsc[k]+'\n')
        rsc_file.close

        date12_file=open('111111-222222_baseline.rsc','w')
        date12_file.write('P_BASELINE_TOP_ODR'+'     '+ '000')
        date12_file.close

##########
if __name__ == '__main__':
   main(sys.argv[:])
