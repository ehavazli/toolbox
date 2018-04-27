#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
from numpy import *
import matplotlib.pyplot as plt
import scipy.io
import math
#from scipy.optimize import brute
from shapely.geometry import MultiLineString, Point,LineString

def main(argv):
    try:
        directory = argv[0]
        model = argv[1]
        offset = float(argv[2])
    except:
        print '''
    *******************************************

       Usage: atan_bestfit_creep.py directory model offset

            directory: directory to transect.mat file
            model: 'interseismic' or 'creep' or 'both'
            offset: if there is an offset of creep from the fault (in meters)

    *******************************************
    '''
        sys.exit(1)
# def line(p1, p2):
#     A = (p1[1] - p2[1])
#     B = (p2[0] - p1[0])
#     C = (p1[0]*p2[1] - p2[0]*p1[1])
#     return A, B, -C
#
# def intersection(L1, L2):
#     D  = L1[0] * L2[1] - L1[1] * L2[0]
#     Dx = L1[2] * L2[1] - L1[1] * L2[2]
#     Dy = L1[0] * L2[2] - L1[2] * L2[0]
#     if D != 0:
#         x = Dx / D
#         y = Dy / D
#         return x,y
#     else:
#         return False
#
# def distance(origin, destination):
#     lat1, lon1 = origin
#     lat2, lon2 = destination
#     radius = 6373.0 # km
#
#     dlat = math.radians(lat2-lat1)
#     dlon = math.radians(lon2-lon1)
#     a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
#         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
#     d = radius * c
#
#     return d

    transectmat = scipy.io.loadmat(directory+'/transect.mat')
    transect = transectmat['dataset'][0][0][1]
    avgInSAR = nanmean(transect,axis=1)
#avgInSAR = avgInSAR[:-8]

#avgInSAR = nanmean(transectmat['dataset'][0][0][1],axis=1)
    stdInSAR = nanstd(transect,axis=1)
#stdInSAR = stdInSAR[:-8]
#stdInSAR = nanstd(transectmat['dataset'][0][0][1],axis=1)
#transect = avgInSAR*1000.0
#std_transect = stdInSAR*1000.0

    transect_dist = transectmat['dataset'][0][0][2]
    transect_lat_first = transectmat['dataset'][0][0][0][0][25]
    transect_lon_first = transectmat['dataset'][0][0][3][0][25]
    transect_first = Point(transect_lon_first,transect_lat_first)
    transect_lat_end = transectmat['dataset'][0][0][0][-1][25]
    transect_lon_end = transectmat['dataset'][0][0][3][-1][25]
    transect_end = Point(transect_lon_end,transect_lat_end)

#transect_middle_line = line([transect_lat_first,transect_lon_first],[transect_lat_last,transect_lon_last])
    transect_middle_line = LineString([(transect_first),(transect_end)])
#fault_line = line([40.6033690,26.834260],[40.7506800,27.335392])
    fault_line = LineString([(26.835038363171353,40.600641025641025),
(26.849744245524295,40.605769230769226),
(26.86253196930946,40.610256410256405),
(26.875319693094628,40.61474358974359),
(26.884271099744243,40.61730769230769),
(26.896419437340153,40.621153846153845),
(26.90920716112532,40.625),
(26.929028132992325,40.631410256410255),
(26.93542199488491,40.63525641025641),
(26.95012787723785,40.64038461538461),
(26.96547314578005,40.64487179487179),
(26.975703324808183,40.648717948717945),
(26.982097186700766,40.651282051282045),
(26.992966751918157,40.65448717948718),
(27.005754475703323,40.65961538461538),
(27.01918158567775,40.66474358974359),
(27.028772378516624,40.66858974358974),
(27.033887468030688,40.67115384615384),
(27.04092071611253,40.674358974358974),
(27.04923273657289,40.67820512820513),
(27.058823529411764,40.68141025641025),
(27.062020460358056,40.682051282051276),
(27.072250639386187,40.68589743589743),
(27.083759590792837,40.68846153846154),
(27.09846547314578,40.69102564102564),
(27.108056265984654,40.69358974358974),
(27.121483375959077,40.69615384615384),
(27.13107416879795,40.69807692307692),
(27.141943734015346,40.70128205128205),
(27.160485933503836,40.70641025641025),
(27.186700767263424,40.71282051282051),
(27.19757033248082,40.715384615384615),
(27.20843989769821,40.71730769230769),
(27.224424552429667,40.72115384615385),
(27.239769820971865,40.725641025641025),
(27.251918158567772,40.72948717948718),
(27.26790281329923,40.73461538461538),
(27.280690537084396,40.73782051282051)])

    intersect = transect_middle_line.intersection(fault_line)
#intersect = intersection(transect_middle_line,fault_line)
    dist = (transect_first.distance(intersect))*100000.0
    dist2 = (intersect.distance(transect_end))*100000.0


##1-D space##
    x = linspace(-dist,dist2,num=len(avgInSAR),endpoint=True)
    xp = (x/1000.)

##Linear Component##
    G = ones([len(avgInSAR),2])
    G[:,0] = xp
    G_inv = dot(linalg.inv(dot(G.T,G)), G.T)
    G_inv = array(G_inv, float32)
    sol = dot(G_inv,avgInSAR)
    k = dot(G,sol)

    D = arange(100.,20000.,100.)
    V = arange(-0.001,-0.10,-0.001)
    rmse=[]

# for d1 in D:
#     for d2 in D:
#         for d in D:
#           for s in V:
#               V = (s/pi)*(arctan(x/d1)-arctan(x/d2)+arctan(d))
#               residual = V - avgInSAR
#               rms = sqrt((sum((residual)**2,0))/len(transect))
#               rmse.append([d,d1,d2,s,rms])
# rmse = array(rmse,float32)
# idx = argmin(rmse[:,4])
# rmse_min = rmse[idx]
# print rmse_min
# m_c = (rmse_min[3]/pi)*(arctan(x/rmse_min[1])-arctan(x/rmse_min[2])+arctan(rmse_min[0]))


####################################################
    if model == 'interseismic':
        for d in D:
            for s in V:
                v2 = sol[0]+((s/pi)*arctan(x/d))+sol[1]
                residual = v2 - avgInSAR
                rms = sqrt((sum((residual)**2,0))/len(transect))
                rmse.append([d, s, rms])
#        print 'RMSE of '+ str(d)+' meters ' + str(s)+' m/yr: '+str(rms)



        rmse = array(rmse,float32)
        idx = argmin(rmse[:,2])
        rmse_min = rmse[idx]
        print rmse_min
        v2_rms_min = ((rmse_min[1]/pi)*arctan(x/rmse_min[0]))
##PLOT average and standard deviation##
        fig = plt.figure()
        plt.rcParams.update({'font.size': 22})
        fig.set_size_inches(20,8)
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax=fig.add_subplot(111)

        ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
        for i in arange(0.0,1.01,0.01):
            ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
        for i in arange(0.0,1.01,0.01):
            ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
        ax.plot(xp,avgInSAR*1000.0,'r-',label = 'Average velocity')
##PLOT average and standard deviation##

#ax.plot(xp,(v1_rms_min*1000.),'k--',label='inversion')
#
#label1 = 'Interseismic: '+str(int(rmse_min[0]))+' meters '+str(int(abs(round(rmse_min[1]*1000.,2))))+r'$\pm$'+str(int(round(rmse_min[2]*1000.)))+'mm/yr'
        label1 = 'Interseismic only: Locking depth: '+str(int(rmse_min[0]))+' meters - Slip rate: '+str(int(abs(round(rmse_min[1]*1000.,2))))+' mm/yr'

        ax.plot(xp,((sol[0]+v2_rms_min+sol[1])*1000.),'b--',label=label1)
        ax.legend(loc='lower left')
        plt.ylabel('Velocity (mm/yr)')
        plt.xlabel('Distance (km)')
        fig.savefig(directory+'atan_best_'+str(rmse_min[0])+'.png')
        plt.close()

    elif model == 'creep':
            for d in D:
                for s in V:
                    v2 = sol[0]+((s/pi)*arctan(x/d))+sol[1]
                    residual = v2 - avgInSAR
                    rms = sqrt((sum((residual)**2,0))/len(transect))
                    rmse.append([d, s, rms])
        #        print 'RMSE of '+ str(d)+' meters ' + str(s)+' m/yr: '+str(rms)



            rmse = array(rmse,float32)
            idx = argmin(rmse[:,2])
            rmse_min = rmse[idx]
            print rmse_min
            v2_rms_min = ((rmse_min[1]/pi)*arctan(x/rmse_min[0]))
##Atan plus creep##
            x = x+offset
            rmse_c=[]
            Q = arange(0.00001,rmse_min[0],100)
# for d1 in Q:
    # for d2 in Q:
    #       for s in V:
    #         v1 = sol[0]+sol[1]+v2_rms_min+((s/pi)*(arctan(x/d1)-arctan(x/d2)))
    #         residual = v1 - avgInSAR
    #         rms = sqrt((sum((residual)**2,0))/len(transect))
    #         rmse_c.append([d1,d2, s, rms])
#            print 'RMSE of '+ str(d)+'_'+str(d+100.)+' meters ' + str(s)+' m/yr: '+str(rms)
            d1=0.01

            for d2 in D:
                for s in V:
                    v1 = sol[0]+sol[1]+v2_rms_min+((s/pi)*(arctan(x/d1)-arctan(x/d2)))
                    residual = v1 - avgInSAR
                    rms = sqrt((sum((residual)**2,0))/len(transect))
                    rmse_c.append([d1,d2, s, rms])

            rmse_c = array(rmse_c,float32)
            idx = argmin(rmse_c[:,3])
            rmse_c_min = rmse_c[idx]
            print rmse_c_min
            v1_rms_min = (rmse_c_min[2]/pi)*(arctan(x/rmse_c_min[0])-arctan(x/rmse_c_min[1]))
# m_c = +sol[0]+v1_rms_min+sol[1]
            m_c =v1_rms_min+v2_rms_min+sol[0]+sol[1]

####################
            fig = plt.figure()
            plt.rcParams.update({'font.size': 22})
            fig.set_size_inches(20,8)
            ax = plt.Axes(fig, [0., 0., 1., 1.], )
            ax=fig.add_subplot(111)

##PLOT average and standard deviation##
            ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
            for i in arange(0.0,1.01,0.01):
                ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
            for i in arange(0.0,1.01,0.01):
                ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
            ax.plot(xp,avgInSAR*1000.0,'r-',label = 'Average velocity')
##PLOT average and standard deviation##

#ax.plot(xp,(v1_rms_min*1000.),'k--',label='inversion')
#
#label1 = 'Interseismic: '+str(int(rmse_min[0]))+' meters '+str(int(abs(round(rmse_min[1]*1000.,2))))+r'$\pm$'+str(int(round(rmse_min[2]*1000.)))+'mm/yr'
            # label1 = 'Interseismic only model: Locking depth: '+str(int(rmse_min[0]))+' meters - Slip rate: '+str(int(abs(round(rmse_min[1]*1000.,2))))+' mm/yr'
            #
            # ax.plot(xp,((sol[0]+v2_rms_min+sol[1])*1000.),'b--',label=label1)

#ax.plot(xp,((sol[0]+v2_rms_min+sol[1])*1000.),xp,(m_c*1000.),'b--')
# ax.plot(xp,((sol[0]+v1_rms_min+sol[1])*1000.),'b-',label=label)

#label2 = 'Interseismic+Creep: '+str(int(rmse_c_min[0]))+' - '+str(int(rmse_c_min[1]))+' meters '+' Slip: '+str(int(abs(round(rmse_c_min[2]*1000.,2))))+r'$\pm$'+str(int(round(rmse_c_min[3]*1000.)))+' mm/yr'
            label2 = 'Interseismic model with creep: Creep depth: 0 - '+str(int(rmse_c_min[1]))+' meters - Creep rate: '+str(int(abs(round(rmse_c_min[2]*1000.,2))))+' mm/yr'

            ax.plot(xp,(m_c*1000.),'k-',label=label2)

            ax.legend(loc='lower left')
            plt.ylabel('Velocity (mm/yr)')
            plt.xlabel('Distance (km)')
            fig.savefig(directory+'atan_best_'+str(rmse_c_min[0])+'_'+str(rmse_c_min[1])+'.png')
            plt.close()
    elif model == 'both':
                    for d in D:
                        for s in V:
                            v2 = sol[0]+((s/pi)*arctan(x/d))+sol[1]
                            residual = v2 - avgInSAR
                            rms = sqrt((sum((residual)**2,0))/len(transect))
                            rmse.append([d, s, rms])
                #        print 'RMSE of '+ str(d)+' meters ' + str(s)+' m/yr: '+str(rms)



                    rmse = array(rmse,float32)
                    idx = argmin(rmse[:,2])
                    rmse_min = rmse[idx]
                    print rmse_min
                    v2_rms_min = ((rmse_min[1]/pi)*arctan(x/rmse_min[0]))
        ##Atan plus creep##
                    x = x+offset
                    rmse_c=[]
                    Q = arange(0.00001,rmse_min[0],100)
        # for d1 in Q:
            # for d2 in Q:
            #       for s in V:
            #         v1 = sol[0]+sol[1]+v2_rms_min+((s/pi)*(arctan(x/d1)-arctan(x/d2)))
            #         residual = v1 - avgInSAR
            #         rms = sqrt((sum((residual)**2,0))/len(transect))
            #         rmse_c.append([d1,d2, s, rms])
        #            print 'RMSE of '+ str(d)+'_'+str(d+100.)+' meters ' + str(s)+' m/yr: '+str(rms)
                    d1=0.01

                    for d2 in D:
                        for s in V:
                            v1 = sol[0]+sol[1]+v2_rms_min+((s/pi)*(arctan(x/d1)-arctan(x/d2)))
                            residual = v1 - avgInSAR
                            rms = sqrt((sum((residual)**2,0))/len(transect))
                            rmse_c.append([d1,d2, s, rms])

                    rmse_c = array(rmse_c,float32)
                    idx = argmin(rmse_c[:,3])
                    rmse_c_min = rmse_c[idx]
                    print rmse_c_min
                    v1_rms_min = (rmse_c_min[2]/pi)*(arctan(x/rmse_c_min[0])-arctan(x/rmse_c_min[1]))
        # m_c = +sol[0]+v1_rms_min+sol[1]
                    m_c =v1_rms_min+v2_rms_min+sol[0]+sol[1]

        ####################
                    fig = plt.figure()
                    plt.rcParams.update({'font.size': 22})
                    fig.set_size_inches(20,8)
                    ax = plt.Axes(fig, [0., 0., 1., 1.], )
                    ax=fig.add_subplot(111)

        ##PLOT average and standard deviation##
                    ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
                    for i in arange(0.0,1.01,0.01):
                        ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
                    for i in arange(0.0,1.01,0.01):
                        ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
                    ax.plot(xp,avgInSAR*1000.0,'r-',label = 'Average velocity')
        ##PLOT average and standard deviation##

        #ax.plot(xp,(v1_rms_min*1000.),'k--',label='inversion')
        #
        #label1 = 'Interseismic: '+str(int(rmse_min[0]))+' meters '+str(int(abs(round(rmse_min[1]*1000.,2))))+r'$\pm$'+str(int(round(rmse_min[2]*1000.)))+'mm/yr'
                    label1 = 'Interseismic only: Locking depth: '+str(int(rmse_min[0]))+' meters - Slip rate: '+str(int(abs(round(rmse_min[1]*1000.,2))))+' mm/yr'

                    ax.plot(xp,((sol[0]+v2_rms_min+sol[1])*1000.),'b--',label=label1)

        #ax.plot(xp,((sol[0]+v2_rms_min+sol[1])*1000.),xp,(m_c*1000.),'b--')
        # ax.plot(xp,((sol[0]+v1_rms_min+sol[1])*1000.),'b-',label=label)

        #label2 = 'Interseismic+Creep: '+str(int(rmse_c_min[0]))+' - '+str(int(rmse_c_min[1]))+' meters '+' Slip: '+str(int(abs(round(rmse_c_min[2]*1000.,2))))+r'$\pm$'+str(int(round(rmse_c_min[3]*1000.)))+' mm/yr'
                    label2 = 'Interseismic with creep: Creep depth: 0 - '+str(int(rmse_c_min[1]))+' meters - Creep rate: '+str(int(abs(round(rmse_c_min[2]*1000.,2))))+' mm/yr'

                    ax.plot(xp,(m_c*1000.),'k-',label=label2)

                    ax.legend(loc='lower left')
                    plt.ylabel('Velocity (mm/yr)')
                    plt.xlabel('Distance (km)')
                    fig.savefig(directory+'atan_best_'+str(rmse_c_min[0])+'_'+str(rmse_c_min[1])+'.png')
                    plt.close()

    else:
                print '''
            *******************************************

               Usage: atan_bestfit_creep.py directory model offset

                    directory: directory to transect.mat file
                    model: 'interseismic' or 'creep'
                    offset: if there is an offset of creep from the fault

            *******************************************
            '''
                sys.exit(1)
###########################
if __name__ == '__main__':
    main(sys.argv[1:])
