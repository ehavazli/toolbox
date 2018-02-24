#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
from numpy import *
import glob
import h5py
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def main(argv):
    try:
        directory = argv[0]
        e_y = int(argv[1])
        e_x = int(argv[2])
    except:
        print '''
    *******************************************

       Usage: dist_std.py [directory] [end_y] [end_x]
       e_y = end iteration at this y
       e_x = end iteration at this x

    *******************************************
    '''
        sys.exit(1)

    files = sorted(glob.glob(directory+'/average_std_*_years.h5'))
    x_res = float(183.25)
    y_res = float(333)
    std_pix = {}
    for k in files:
        file_name = k.split('_')[-2:]
        years = file_name[0]
        print 'Working on '+str(years)+' years long time series'
        f = h5py.File(k,'r')
        dset = f['velocity'].get('velocity')
        stda = asarray(dset)
        std_lst = []
        dist = []
        for i in xrange(0,10):
            s_y = random.randint(0,999)
            s_x = random.randint(0,999)
            ref_std = stda[s_y][s_x]
            # print str(s_y)+' '+str(s_x)
            for y in xrange(0,e_y):
                for x in xrange(0,e_x):
                    std_val = abs(stda[y][x]-ref_std)
                    std_lst.append(std_val)
                    dist.append(sqrt(((s_y-y)*y_res)**2+((s_x-x)*x_res)**2))
        # print years+' '+str(len(std_lst))+' '+str(len(dist))
        std_pix[int(years)] = [std_lst,dist]

    x = arange(0,360000,10000)
    color=iter(cm.jet(linspace(0,1,8)))

    for key, value in sorted(std_pix.iteritems()):
        print 'Working on plots of '+str(key)+' years data'
        c=next(color)
        avg = []
        avg_std=[]
        avg_er = []
        for n in xrange(0,35):
            print 'Distances between '+str(x[n])+' and '+str(x[n+1])
            plt.ylabel('Uncertainty (mm/yr)')
            plt.xlabel('Distance (km)')
            for i, q in enumerate(value[1]):
                if x[n] <= value[1][i] <= x[n+1]:
                    avg_std.append(value[0][i])
            avg.append(nanmean(avg_std))
            avg_er.append(std(avg_std))
#        avg.insert(0,0)
#        avg_er.insert(0,0)
        x_plt = x[1:]
        print 'X: '+str(x)
        print 'AVG: '+str(len(avg))
        # print 'AVG_ER: '+str(avg_er)
        # plt.scatter(x/1000.0,array(avg)*10000.0,c=c,label=(str(key)+' years'))
        plt.errorbar(x_plt/1000.0,array(avg)*100000.0,array(avg_er)*100000.0,c=c,marker='o',xerr = None,ls='none',label=(str(key)+' years'))
    plt.legend()
    plt.savefig(directory+'/STDvsDIST.png',bbox_inches="tight",dpi=600)





#     color=iter(cm.jet(linspace(0,1,8)))
#     for key, value in sorted(std_pix.iteritems()):
#         c=next(color)
#         plt.ylabel('Uncertainty (mm/yr)')
#         plt.xlabel('Distance (km)')
# #        plt.axis([0, 350,0,2])
#         plots = plt.plot(array(value[1])/1000.0,array(value[0])*10000.0, '*',c=c,label=(str(key)+' years'))
# #        plt.plot(sol,c=c,label=(str(key)+' years'))
#         plt.legend()
#     plt.savefig(directory+'/STDvsDIST.png',bbox_inches="tight",dpi=600)
# #    plt.close()

            # out_file = open(directory+'std_dst_'+years+'_'+str(y)+'_'+str(s_y)+'-''+str(s_x)+'.txt', 'w')
            # std_pix_ar = array(std_pix)*float(10000.0)
            # for item in std_pix_ar:
            #     out_file.write("%s\n" % item)
            # out_file.close




###########################
if __name__ == '__main__':
    main(sys.argv[1:])
