#! /usr/bin/env python2
#Author: Emre Havazli

import os
import sys
import h5py
import pysar._readfile as readfile
import pysar._datetime as ptime

def main(argv):
  try:
    timeseries_file = argv[1]
    print timeseries_file
  except:
    print '''
        *******************************************
           Calculate incremental changes between timeseries epochs

           Usage: ts_inc.py [timeseries_file]

        *******************************************
        '''
    sys.exit(1)


#  timeseries_file = 'timeseries.h5'
  atr = readfile.read_attribute(timeseries_file)
  k = atr['FILE_TYPE']
  print 'input '+k+' file: '+timeseries_file
  if not k == 'timeseries':
    sys.exit('ERROR: input file is not timeseries!')
  h5file = h5py.File(timeseries_file)

#####################################
  ## Date Info
  dateListAll = sorted(h5file[k].keys())
  print '--------------------------------------------'
  print 'Dates from input file: '+str(len(dateListAll))
  print dateListAll

  dateNum = len(dateListAll)
  h5file_inc = 'timeseries_inc.h5'
  f = h5py.File(h5file_inc)
  gg = f.create_group('timeseries')

  width = int(atr['WIDTH'])
  length = int(atr['FILE_LENGTH'])

  for i in range(dateNum):
      if i+1 >= dateNum:
          print 'End of time series file'
      else:
          date_1 = dateListAll[i]
          date_2 = dateListAll[i+1]
          title = str(date_2)+'-'+str(date_1)
          print 'Calculating '+ title
          ts_1 = h5file[k].get(date_1)[:].flatten()
          ts_2 = h5file[k].get(date_2)[:].flatten()
          inc = ts_2 - ts_1
          inc = inc.reshape(length,width)
          dset = gg.create_dataset(title, data=inc, compression='gzip')

          for key , value in atr.iteritems():
              gg.attrs[key]=value



############################################################################
if __name__ == '__main__':
    main(sys.argv[:])
