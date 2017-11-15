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
    epochList = argv[2]
    print 'EPOCHS: ' + epochList
    if epochList == 'all':
        epochs = 'all'
    else:
        epochs = argv[2].split()
        print 'Epochs to be subtracted: '+ str(epochs)

  except:
    print '''
        *******************************************
           Calculate incremental changes between timeseries epochs

           Usage: ts_inc.py [timeseries_file] [epochs or all]

           ts_inc.py timeseries_tropHgt_demErr_refDate.h5 '20170725 20170824 20170923 20171023'
           ts_inc.py timeseries_tropHgt_demErr_refDate.h5 'all'

        *******************************************
        '''
    sys.exit(1)

  atr = readfile.read_attribute(timeseries_file)
  k = atr['FILE_TYPE']
  print 'input '+k+' file: '+timeseries_file
  if not k == 'timeseries':
    sys.exit('ERROR: input file is not timeseries!')
  h5file = h5py.File(timeseries_file)

  if epochs == 'all':
#####################################
  ## Date Info
      dateListAll = sorted(h5file[k].keys())
      print '--------------------------------------------'
      print 'Dates from input file: '+str(len(dateListAll))
      print dateListAll

      dateNum = len(dateListAll)
      h5file_inc = timeseries_file[0:-3]+'_inc.h5'
      f = h5py.File(h5file_inc,'w')
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

  elif len(epochs) > 1:
      h5file_diff = timeseries_file[0:-3]+'_diff.h5'
      f = h5py.File(h5file_diff,'w')
      gg = f.create_group('timeseries')
      width = int(atr['WIDTH'])
      length = int(atr['FILE_LENGTH'])

      dateNum = len(epochs)
      for i in range(dateNum):
          if i+1 >= dateNum:
              print 'End of date list'
          else:
              date_1 = epochs[i]
              date_2 = epochs[i+1]
              title = str(date_2)+'-'+str(date_1)
              print 'Calculating '+ title
              ts_1 = h5file[k].get(date_1)[:].flatten()
              ts_2 = h5file[k].get(date_2)[:].flatten()
              inc = ts_2 - ts_1
              inc = inc.reshape(length,width)
              dset = gg.create_dataset(title, data=inc, compression='gzip')

              for key , value in atr.iteritems():
                  gg.attrs[key]=value
  else:
            print '''
                *******************************************
                   Calculate incremental changes between timeseries epochs

                   Usage: ts_inc.py [timeseries_file] [epochs or all]

                *******************************************
                '''
            sys.exit(1)




############################################################################
if __name__ == '__main__':
    main(sys.argv[:])
