#! /usr/bin/env python2
#@author: Emre Havazli

import sys
from numpy import *
import h5py
import pysar
import pysar._readfile as readfile
import pysar._pysar_utilities as ut

def main(argv):
    try:
        velocityfile = argv[0]
        y = int(argv[1])
        x = int(argv[2])
        radius = int(argv[3])
    except:
        print '''
    *******************************************

       Usage: average_circle.py h5file y x radius

    *******************************************
    '''
        sys.exit(1)
    
    f = h5py.File(velocityfile,'r')
    dset = f['velocity'].get('velocity')    
    velocity = asarray(dset)
    ref_vel = velocity[y][x]
    print 'REF VEL: ' + str(ref_vel)
    atr = readfile.read_attribute(velocityfile)
    crc_mask = ut.circle_index(atr,(y,x,radius))
    masked_vel = velocity[crc_mask]
#    diff_masked_vel = ref_vel - masked_vel
    average_crc = average(masked_vel)
    std_masked = std(masked_vel)
#    print 'Masked Vel: ' + str(masked_vel)
#    print 'Referenced Vel: ' +str(diff_masked_vel)
    print 'Average in the circle: ' +str(average_crc)
    print 'Standard Dev of Circle: ' +str(std_masked)
###########################
if __name__ == '__main__':
    main(sys.argv[1:])


