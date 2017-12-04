#! /usr/bin/env python2
#Author: Emre Havazli

import os
import sys
from numpy import *
import h5py

def main(argv):
  try:
     first_velocity_file = argv[1]
     second_velocity_file = argv[2]
  except:
    print '''
    *******************************************

       Usage: residual_velocity.py velocityFile1 velocityFile2 (File1 - File2)

              if you want to add the residuals to a velocity file introduce add argument

              residual_velocity.py velocityFile1 velocityFile2 add
    *******************************************
    '''
    sys.exit(1)


  f = h5py.File(first_velocity_file,'r')
  g = h5py.File(second_velocity_file,'r')
  dset = f['velocity'].get('velocity')
  first_velocity = dset[0:dset.shape[0]]
  dset  = g['velocity'].get('velocity')
  second_velocity = dset[0:dset.shape[0]]
  residual_val = first_velocity - second_velocity
#  if argv[3] == 'add':
#     residual_val = first_velocity - second_velocity
#  else: pass
  k = h5py.File('residual_velocity.h5','w')
  gg = k.create_group('velocity')
  dset = gg.create_dataset('velocity', data=residual_val, compression='gzip')

  for key , value in f['velocity'].attrs.iteritems():
     gg.attrs[key]=value

##########
if __name__ == '__main__':
   main(sys.argv[:])
