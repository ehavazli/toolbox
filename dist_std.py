#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
from numpy import *
import glob
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def main(argv):
    try:
        directory = argv[0]
        s_y = int(argv[1])
        s_x = int(argv[2])
        e_y = int(argv[3])
        e_x = int(argv[4])
    except:
        print '''
    *******************************************

       Usage: dist_std.py [directory] [start-y] [start-x] [end-y] [end-x]

    *******************************************
    '''
        sys.exit(1)

    files = sorted(glob.glob(directory+'/average_std_*_years.h5'))
    x_res = float(183.25)
    y_res = float(333)
    color=iter(cm.jet(linspace(0,1,8)))
    for k in files:
        file_name = k.split('_')[-2:]
        years = file_name[0]
        f = h5py.File(k,'r')
        dset = f['velocity'].get('velocity')
        std = asarray(dset)
        ref_std = std[s_y][s_x]
        std_pix = []
        c=next(color)
        for y in xrange(s_y,e_y):
            for x in xrange(s_x,e_x):
                std_val = std[y][x]-ref_std
                std_pix.append(std_val)
                dist = sqrt(((s_y-y)*y_res)**2+((s_x-x)*x_res)**2)
                plt.ylabel('Uncertainty (mm/yr)')
                plt.xlabel('Distance (km)')
                plt.axis([0, 350,0, 2])
                plt.plot(dist/1000.0,std_val*10000.0, '*',c=c)
    plt.savefig(directory+'/STDvsDIST.png',bbox_inches="tight")
    plt.close()

            # out_file = open(directory+'std_dst_'+years+'_'+str(y)+'_'+str(s_y)+'-''+str(s_x)+'.txt', 'w')
            # std_pix_ar = array(std_pix)*float(10000.0)
            # for item in std_pix_ar:
            #     out_file.write("%s\n" % item)
            # out_file.close




###########################
if __name__ == '__main__':
    main(sys.argv[1:])
