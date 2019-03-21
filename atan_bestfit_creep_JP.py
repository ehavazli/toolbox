#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse
import h5py

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def enu2los(e, n, u, inc_angle=34., head_angle=-12.873):
    """
    Parameters: e : np.array or float, displacement in east-west direction, east as positive
                n : np.array or float, displacement in north-south direction, north as positive
                u : np.array or float, displacement in vertical direction, up as positive
                inc_angle  : np.array or float, local incidence angle from vertical
                head_angle : np.array or float, satellite orbit from the north in clock-wise direction as positive
    For AlosA: inc_angle = 34, head_angle = -12.873
    For AlosD: inc_angle = 34, head_angle = -167.157
    For SenD: inc_angle = 34, head_angle = -168
    """
    # if input angle is azimuth angle
    # if (head_angle + 180.) > 45.:
    #     head_angle = azimuth2heading_angle(head_angle)

    inc_angle *= np.pi/180.
    head_angle *= np.pi/180.
    v_los = (-1 * e * np.cos(head_angle) * np.sin(inc_angle)
             + n * np.sin(head_angle) * np.sin(inc_angle)
             + u * np.cos(inc_angle))
    return v_los

def main(argv):
    try:
        h5File = argv[0]
        directory = argv[1]+'/'
        model = argv[2]
        noit = int(argv[3])
    except:
        print '''
    *******************************************
       Usage: atan_bestfit_creep.py [file] [output_directory] [model] [number of MC iterations] [slip rate]* [contour level]*
            directory: directory to transect.mat file
            model: 'interseismic' or 'creep' or 'both'
            slip rate: slip rate to be fixed in m/yr (optional)
            contour level: number of contours (optional)
    *******************************************
    '''
        sys.exit(1)
    f = h5py.File(h5File,'r')
    transect = np.array(f.get('transect'))
    x = np.array(f.get('distance'))

    avgInSAR = np.nanmean(transect,axis=1)
    stdInSAR = np.nanstd(transect,axis=1)


#     transect_dist = transectmat['dataset'][0][0][2]
#     transect_lat_first = transectmat['dataset'][0][0][0][0][25]
#     transect_lon_first = transectmat['dataset'][0][0][3][0][25]
#     transect_first = Point(transect_lon_first,transect_lat_first)
#     transect_lat_end = transectmat['dataset'][0][0][0][-1][25]
#     transect_lon_end = transectmat['dataset'][0][0][3][-1][25]
#     transect_end = Point(transect_lon_end,transect_lat_end)
#
#     transect_middle_line = LineString([(transect_first),(transect_end)])
#     fault_line = LineString([(26.835038363171353,40.600641025641025),
# (26.849744245524295,40.605769230769226),
# (26.86253196930946,40.610256410256405),
# (26.875319693094628,40.61474358974359),
# (26.884271099744243,40.61730769230769),
# (26.896419437340153,40.621153846153845),
# (26.90920716112532,40.625),
# (26.929028132992325,40.631410256410255),
# (26.93542199488491,40.63525641025641),
# (26.95012787723785,40.64038461538461),
# (26.96547314578005,40.64487179487179),
# (26.975703324808183,40.648717948717945),
# (26.982097186700766,40.651282051282045),
# (26.992966751918157,40.65448717948718),
# (27.005754475703323,40.65961538461538),
# (27.01918158567775,40.66474358974359),
# (27.028772378516624,40.66858974358974),
# (27.033887468030688,40.67115384615384),
# (27.04092071611253,40.674358974358974),
# (27.04923273657289,40.67820512820513),
# (27.058823529411764,40.68141025641025),
# (27.062020460358056,40.682051282051276),
# (27.072250639386187,40.68589743589743),
# (27.083759590792837,40.68846153846154),
# (27.09846547314578,40.69102564102564),
# (27.108056265984654,40.69358974358974),
# (27.121483375959077,40.69615384615384),
# (27.13107416879795,40.69807692307692),
# (27.141943734015346,40.70128205128205),
# (27.160485933503836,40.70641025641025),
# (27.186700767263424,40.71282051282051),
# (27.19757033248082,40.715384615384615),
# (27.20843989769821,40.71730769230769),
# (27.224424552429667,40.72115384615385),
# (27.239769820971865,40.725641025641025),
# (27.251918158567772,40.72948717948718),
# (27.26790281329923,40.73461538461538),
# (27.280690537084396,40.73782051282051)])
#
#     intersect = transect_middle_line.intersection(fault_line)
#intersect = intersection(transect_middle_line,fault_line)
    # dist = (transect_first.distance(intersect))*100000.0
    # dist2 = (intersect.distance(transect_end))*100000.0

################################################################################
###GPS LOCATIONS
    # bgnt = [26.570, 40.932]
    # doku = [26.706, 40.739]
    # kvam = [26.871, 40.601]
    # kvam2 = [26.871, 40.601]
    # sev2 = [26.880, 40.396]
    # transect_lat = transectmat['dataset'][0][0][0][0]
    # transect_lon = transectmat['dataset'][0][0][3][0]
    # print(transect_lat[-1])
    # print(transect_lon[0])
    # transect_lat_end = transectmat['dataset'][0][0][0][-1]
    # transect_lon_end = transectmat['dataset'][0][0][3][-1]
    # print(transect_lat_end[0])
    # print(transect_lon_end[-1])
    # sys.exit()
################################################################################
    depth = []
    slip = []
    McMinRms = []
    varModel = []
##1-D space##
    # x = np.linspace(-dist,dist2,num=len(avgInSAR),endpoint=True)

    xp = (x/1000.)
################################################################################
    # avgInSAR = avgInSAR-avgInSAR[0]
    # transect = transect-transect[0]

    G = np.ones([len(transect),2])
    G[:,0] = xp
    G_inv = np.dot(np.linalg.inv(np.dot(G.T,G)), G.T)
    G_inv = np.array(G_inv, np.float32)
    sol = np.dot(G_inv,avgInSAR)
    k = np.dot(G,sol)

    D = np.arange(0.0001,30000.,100.)
    V = np.arange(0.000,0.03,0.0005)
    Q = np.arange(-5000,5000,100)

    rmse_inv = []

    try:
        s = np.float(argv[4])
        print "s is fixed: " + str(s)
        for d in D:
            for i in V:
                for j in Q:
                    DistMo = x+j
                    v2 = sol[0]+((s/np.pi)*np.arctan(DistMo/d))+sol[1]
                    residual = v2 - avgInSAR
                    rms = np.sqrt((sum((residual)**2,0))/len(transect))
                    rmse_inv.append([d, s, rms, j])

    except:
        for d in D:
            for s in V:
                for j in Q:
                    DistMo = x+j
                    v2 = sol[0]+((s/np.pi)*np.arctan(DistMo/d))+sol[1]
                    residual = v2 - avgInSAR
                    rms = np.sqrt((sum((residual)**2,0))/len(transect))
                    rmse_inv.append([d, s, rms, j])



    # rmse_inv=[[9000, -0.020, 0]]
    rmse_inv = np.array(rmse_inv,np.float32)
    idx = np.argmin(rmse_inv[:,2])
    rmse_inv_min = rmse_inv[idx]
    print ('Inversion: Locking Depth = ' +str(np.round(rmse_inv_min[0]))+
    ' meters'+' Slip rate: '+str(np.round(rmse_inv_min[1]*1000.,2))+' mm/yr ' + 'RMSE: '+
    str(np.round(rmse_inv_min[2],3))+' Offset: '+str(rmse_inv_min[3])+' meters')
    v2_rms_min_inv =sol[0]+sol[1]+((rmse_inv_min[1]/np.pi)*np.arctan((x+rmse_inv_min[3])/rmse_inv_min[0]))
################################################################################
    if model == 'interseismic':

#######Monte-Carlo Error Bounds#################################################
            for q in xrange(noit):
                avgVel = []
                for i in xrange(len(avgInSAR)):
                    avgVel.append(np.random.normal(avgInSAR[i], stdInSAR[i]))
                avgVel = np.array(avgVel)

                G = np.ones([len(avgVel),2])
                G[:,0] = xp
                G_inv = np.dot(np.linalg.inv(np.dot(G.T,G)), G.T)
                G_inv = np.array(G_inv, np.float32)
                sol = np.dot(G_inv,avgVel)
                k = np.dot(G,sol)

                # D = np.arange(1000.,10000.,100.)
                # V = np.arange(-0.001,-0.030,-0.0005)
                rmse_Mc=[]

                try:
                    s = np.float(argv[4])
                    for d in D:
                        for i in V:
                            DistMo = x+rmse_inv_min[3]
                            v2 = sol[0]+((s/np.pi)*np.arctan(DistMo/d))+sol[1]
                            residual = v2 - avgInSAR
                            rms = np.sqrt((sum((residual)**2,0))/len(transect))
                            rmse_Mc.append([d, s, rms])

                except:
                    for d in D:
                        for s in V:
                            DistMo = x+rmse_inv_min[3]
                            v2 = sol[0]+((s/np.pi)*np.arctan(DistMo/d))+sol[1]
                            residual = v2 - avgInSAR
                            rms = np.sqrt((sum((residual)**2,0))/len(transect))
                            rmse_Mc.append([d, s, rms])



                # for d in D:
                #     for s in V:
                #         v2 = sol[0]+((s/np.pi)*np.arctan(x/d))+sol[1]
                #         residual = v2 - avgVel
                #         rms = np.sqrt((sum((residual)**2,0))/len(transect))
                #         rmse_Mc.append([d, s, rms])

#######Monte-Carlo Error Bounds#################################################
                rmse_Mc_inv = np.array(rmse_Mc,np.float32)
                idx = np.argmin(rmse_Mc_inv[:,2])
                rmse_Mc_inv_min = rmse_Mc_inv[idx]
                print str(q)+' '+ str(rmse_Mc_inv_min)
                depth.append(rmse_Mc_inv_min[0])
                slip.append(rmse_Mc_inv_min[1])
                # r.append(rmse_min[2])
                varModel.append(rmse_Mc_inv_min[2]**2)
                McMinRms.append(rmse_Mc_inv_min[2])

            stdModel = np.sqrt(sum(varModel))
            slip[:] = [x * -1000.0 for x in slip]

            stdDepth = roundup(np.std(depth))
            stdSlip = math.ceil(np.std(slip))

################################################################################
##PLOT average and standard deviation##
            fig = plt.figure()
            plt.rcParams.update({'font.size': 24})
            fig.set_size_inches(12,12)
            ax = plt.Axes(fig, [0., 0., 1., 1.], )
            ax=fig.add_subplot(111)

            plt.scatter(slip,depth,c='blue',marker='.')
            plt.scatter(rmse_inv_min[1]*-1000.,rmse_inv_min[0], s=200, c='red', marker='*')

            depth.append(rmse_inv_min[0])
            slip.append(rmse_inv_min[1])
            McMinRms.append(rmse_inv_min[2]*1000.)

            Xi, Yi = np.meshgrid(-V*1000.,D)
            z = np.array(np.reshape(rmse_Mc_inv[:,2],(len(D),len(V)))*1000.)
            # z = np.array(McMinRms)*1000
            # contours = plt.contour(Xi, Yi, z*1000.,int(np.amax(z*1000.)-np.amin(z*1000.)))
            # contours = plt.contour(Xi, Yi, z,500,colors='black')
            # contours = plt.contour(Xi, Yi, z,levels=(2.50,2.75,3,3.25,3.50,3.75,4,4.25,4.50,4.75,5),colors='black',linewidths=0.5)
            try:
                cLevel = np.int(argv[6])
            except:
                cLevel = 100

            contours = plt.contour(Xi, Yi, z,cLevel,colors='black',linewidths=0.5)

            plt.clabel(contours, inline=True, fontsize=18,fmt='%.2f')

            cov = np.cov(slip, depth)
            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            for nstd in xrange(1,2):        ####nstd is 1sigma, 2sigma, 3sigma
                w, h = 2 * nstd * np.sqrt(vals)
                ell = Ellipse(xy=(np.mean(slip), np.mean(depth)),
                            width=w, height=h,
                            angle=theta, color='black')
                ell.set_facecolor('none')
                ell.set_linestyle('--')
                ell.set_linewidth('2')
                ax.add_artist(ell)
                ax.annotate('68%',xy=((np.mean(slip)),(np.mean(depth)-(w))),xycoords='data')

            print 'Mean Slip '+ str(np.mean(slip))
            print 'Mean Depth '+ str(np.mean(depth))
            print 'W: '+ str(w)
            print 'H: '+ str(h)

            plt.ylabel('Locking Depth (m)')
            plt.xlabel('Slip (mm/yr)')
            plt.xlim((np.min(slip)-3),(np.max(slip)+3))
            if np.min(depth) < 1000:
                plt.ylim(0,(np.max(depth)+1000))
            else:
                plt.ylim((np.min(depth)-1000.),(np.max(depth)+1000))
            fig.savefig(directory+'depth_vs_slip.png')
            plt.close()
################################################################################
            fig = plt.figure()
            plt.rcParams.update({'font.size': 20})
            fig.set_size_inches(20,8)
            ax = plt.Axes(fig, [0., 0., 1., 1.], )
            ax=fig.add_subplot(111)

            ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
            for i in np.arange(0.0,1.01,0.01):
                ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
            for i in np.arange(0.0,1.01,0.01):
                ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
            ax.plot(xp,avgInSAR*1000.0,'r-',label = 'Average velocity')

            label1 = 'Interseismic only: Locking depth: '+str(int(rmse_inv_min[0]))+'$\pm$'+str(int(stdDepth))+'m - Slip rate: '+str(int(abs(round(rmse_inv_min[1]*1000.,2))))+'$\pm$'+str(int(stdSlip))+' mm/yr'
            # label1 = 'D: '+str(int(rmse_inv_min[0]))+'$\pm$'+str(int(stdDepth))+'m - V: '+str(int(abs(round(rmse_inv_min[1]*1000.,2))))+'$\pm$'+str(int(stdSlip))+'mm/yr'
            label2 = 'Offset from fault: ' + str(rmse_inv_min[3])+' meters'
            ax.plot(xp,((v2_rms_min_inv)*1000.),'b--',label=label1)
            ax.legend(loc='lower left')
            # ax.text(0.8,0.8,'Offset from fault: ' + str(rmse_inv_min[3])+' meters',bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10},transform=ax.transAxes)
            ax.text(-0.9,18.5,'Offset from fault: ' + str(rmse_inv_min[3]*-1)+' meters',bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 10})

            # ax.legend(loc='upper right')
            plt.ylabel('LOS Velocity (mm/yr)')
            plt.xlabel('Distance from fault (km)')
            fig.savefig(directory+'atan_best_'+str(rmse_inv_min[0])+'.png')
            plt.close()
################################################################################
    elif model == 'creep':
            x = x+offset
            rmse_c_inv=[]
            # Q = np.arange(0.00001,rmse_inv[0],100)

            d1=0.01

            for d2 in D:
                for s in V:
                    v1 = v2_rms_min_inv+((s/np.pi)*(np.arctan(x/d1)-np.arctan(x/d2)))
                    residual = v1 - avgInSAR
                    rms = np.sqrt((sum((residual)**2,0))/len(transect))
                    rmse_c_inv.append([d1,d2, s, rms])

            rmse_c_inv = np.array(rmse_c_inv,np.float32)
            idx = np.argmin(rmse_c_inv[:,3])
            rmse_c_inv_min = rmse_c_inv[idx]
            print 'Creep Inversion: '+str(rmse_c_inv_min)
            v1_rms_min =(rmse_c_inv_min[2]/np.pi)*(np.arctan(x/rmse_c_inv_min[0])-np.arctan(x/rmse_c_inv_min[1]))
            m_c_inv =v1_rms_min+v2_rms_min_inv
#######Monte-Carlo Error Bounds#################################################
            for q in xrange(noit):
                avgVel = []
                rmse_c=[]
                for i in xrange(len(avgInSAR)):
                    avgVel.append(np.random.normal(avgInSAR[i], stdInSAR[i]))
                avgVel = np.array(avgVel)

                for d2 in D:
                    for s in V:
                        v1 = v2_rms_min_inv+((s/np.pi)*(np.arctan(x/d1)-np.arctan(x/d2)))
                        residual = v1 - avgVel
                        rms = np.sqrt((sum((residual)**2,0))/len(transect))
                        rmse_c.append([d1,d2, s, rms])

                rmse_c = np.array(rmse_c,np.float32)
                idx = np.argmin(rmse_c[:,3])
                rmse_c_min = rmse_c[idx]
                print str(q)+' '+ str(rmse_c_min)
                # v1_rms_c_min = (rmse_c_min[2]/np.pi)*(np.arctan(x/rmse_c_min[0])-np.arctan(x/rmse_c_min[1]))
                # m_c =v1_rms_c_min+v2_rms_min_inv+sol[0]+sol[1]

                depth.append(rmse_c_min[1])
                slip.append(rmse_c_min[2])
                varModel.append(rmse_c_min[3]**2)
                McMinRms.append(rmse_c_min[3])


            stdModel = np.sqrt(sum(varModel))
            slip[:] = [x * -1000.0 for x in slip]
#######Monte-Carlo Error Bounds Plot STD Ellipses###############################
            fig = plt.figure()
            plt.rcParams.update({'font.size': 24})
            fig.set_size_inches(12,12)
            ax = plt.Axes(fig, [0., 0., 1., 1.], )
            ax=fig.add_subplot(111)

            plt.scatter(slip,depth,c='blue',marker='.')
            plt.scatter(rmse_c_inv_min[2]*-1000.,rmse_c_inv_min[1], s=200, c='red', marker='*')

            depth.append(rmse_c_inv_min[1])
            slip.append(rmse_c_inv_min[2]*-1000.)
            McMinRms.append(rmse_c_inv_min[3]*1000.)

            Xi, Yi = np.meshgrid(-V*1000.,D)
            z = np.array(np.reshape(rmse_c[:,3],(len(V),len(D)))*1000.)
            # z = np.array(McMinRms)*1000
            # contours = plt.contour(Xi, Yi, z*1000.,int(np.amax(z*1000.)-np.amin(z*1000.)))
            # contours = plt.contour(Xi, Yi, z,500,colors='black')
            # contours = plt.contour(Xi, Yi, z,levels=(2.50,2.75,3,3.25,3.50,3.75,4,4.25,4.50,4.75,5),colors='black',linewidths=0.5)
            # contours = plt.contour(Xi, Yi, z,700,colors='black',linewidths=0.5)
            #
            # plt.clabel(contours, inline=True, fontsize=18,fmt='%.2f')

            cov = np.cov(slip, depth)
            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            for nstd in xrange(1,2):        ####nstd is 1sigma, 2sigma, 3sigma
                w, h = 2 * nstd * np.sqrt(vals)
                ell = Ellipse(xy=(np.mean(slip), np.mean(depth)),
                            width=w, height=h,
                            angle=theta, color='black')
                ell.set_facecolor('none')
                ell.set_linestyle('--')
                ell.set_linewidth('2')
                ax.add_artist(ell)
                ax.annotate('68%',xy=((np.mean(slip)),(np.mean(depth)+(w/2))),xycoords='data')

            print 'Mean Slip '+ str(np.mean(slip))
            print 'Mean Depth '+ str(np.mean(depth))
            print 'W: '+ str(w)
            print 'H: '+ str(h)

            plt.ylabel('Depth (m)')
            plt.xlabel('Slip (mm/yr)')
            # plt.xlim(0,10)
            # plt.ylim(200,1200)
            fig.savefig('depth_vs_slip.png')
            plt.close()
            # fig = plt.figure()
            # plt.rcParams.update({'font.size': 22})
            # fig.set_size_inches(10,10)
            # ax = plt.Axes(fig, [0., 0., 1., 1.], )
            # ax=fig.add_subplot(111)
            #
            # plt.scatter(slip,depth)
            # plt.scatter(rmse_c_min[2]*1000.,rmse_c_min[1], s=200, c='red', marker='*')
            # plt.ylabel('Depth (m)')
            # plt.xlabel('Slip (mm/yr)')
            # # plt.xlim(1,30)
            # # plt.ylim(1000,10000)
            # fig.savefig('depth_vs_slip.png')
            # plt.close()
            #
            # depth.append(rmse_inv_min[0])
            # slip.append(rmse_inv_min[1])
            # stdDepth = roundup(np.std(depth))
            # stdSlip = math.ceil(np.std(slip))
            #
            # fig = plt.figure()
            # plt.rcParams.update({'font.size': 22})
            # fig.set_size_inches(15,15)
            # ax = plt.Axes(fig, [0., 0., 1., 1.], )
            # ax=fig.add_subplot(111)
            # InvRms = rmse_c[:,3]
            # z = np.array(np.reshape(InvRms,(len(V),len(D))))
            # Xi, Yi = np.meshgrid(-V*1000.,D)
            # # plt.scatter(-rmse_inv[:,1]*1000.,rmse_inv[:,0],c='blue')
            # plt.scatter(rmse_c_min[2]*1000.,rmse_c_min[1], s=500, c='red', marker='*')
            # contours = plt.contour(Xi, Yi, z*1000,int(np.amax(z*1000.)-np.amin(z*1000.)), colors='black')
            # # contours = plt.contour(Xi, Yi, z*1000.,int(np.amax(z*1000.)-np.amin(z*1000.)), cmap='RdGy')
            # plt.clabel(contours, inline=True, fontsize=10)
            # # CS = plt.contourf(Xi, Yi, z*1000.,50, cmap='RdGy')
            # # plt.colorbar();
            # plt.ylabel('Depth (m)')
            # plt.xlabel('Slip (mm/yr)')
            # # plt.xlim(-1,20)
            # # plt.ylim(-100,20000)
            # fig.savefig('Contour_depth_vs_slip.png')
            # plt.close()

####################################Error Ellipses##############################
            # fig = plt.figure()
            # plt.rcParams.update({'font.size': 22})
            # fig.set_size_inches(10,10)
            # ax = plt.Axes(fig, [0., 0., 1., 1.], )
            # ax=fig.add_subplot(111)
            #
            # cov = np.cov(slip, depth)
            # vals, vecs = eigsorted(cov)
            # theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            # for nstd in xrange(1,4):        ####nstd is 1sigma, 2sigma, 3sigma
            #     w, h = 2 * nstd * np.sqrt(vals)
            #     ell = Ellipse(xy=(np.mean(slip), np.mean(depth)),
            #                 width=w, height=h,
            #                 angle=theta, color='black')
            #     ell.set_facecolor('none')
            #     ax.add_artist(ell)

            # plt.scatter(slip,depth)
            # plt.scatter(rmse_c_min[2]*1000.,rmse_c_min[1], s=60, c='red', marker='*')
            # plt.ylabel('Depth (m)')
            # plt.xlabel('Slip (mm/yr)')
            # # plt.xlim(1,30)
            # # plt.ylim(1000,10000)
            # fig.savefig('depth_vs_slip.png')
            # plt.close()
####################################Error Ellipses##############################
##Monte-Carlo Error Bounds Plot STD Ellipses####################################

            # depth.append(rmse_c_inv_min[0])
            # slip.append(rmse_c_inv_min[1])
            stdDepth = roundup(np.std(depth))
            stdSlip = math.ceil(np.std(slip))

################################################################################
            fig = plt.figure()
            plt.rcParams.update({'font.size': 22})
            fig.set_size_inches(20,8)
            ax = plt.Axes(fig, [0., 0., 1., 1.], )
            ax=fig.add_subplot(111)

            ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
            for i in np.arange(0.0,1.01,0.01):
                ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
            for i in np.arange(0.0,1.01,0.01):
                ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
            ax.plot(xp,avgInSAR*1000.0,'r-',label = 'Average velocity')

            label2 = 'Creep depth: 0 - '+str(int(rmse_c_inv_min[1]))+'$\pm$'+str(stdDepth)+' m - Creep rate: '+str(int(abs(round(rmse_c_inv_min[2]*1000.,2))))+'$\pm$'+str(int(stdSlip))+' mm/yr'

            ax.plot(xp,(m_c_inv*1000.),'k-',label=label2)

            ax.legend(loc='lower left')
            plt.ylabel('Velocity (mm/yr)')
            plt.xlabel('Distance (km)')
            fig.savefig(directory+'atan_best_'+str(rmse_c_inv_min[0])+'_'+str(rmse_c_inv_min[1])+'.png')
            plt.close()
################################################################################

    elif model == 'both':
        transect = transect-transect[0]
        avgInSAR = avgInSAR-avgInSAR[0]
        rmse = []
        for d in D:
            for s in V:
                v2 = sol[0]+((s/np.pi)*np.arctan(x/d))+sol[1]
                residual = v2 - avgInSAR
                rms = np.sqrt((sum((residual)**2,0))/len(transect))
                rmse.append([d, s, rms])
                #        print 'RMSE of '+ str(d)+' meters ' + str(s)+' m/yr: '+str(rms)



        rmse = np.array(rmse,np.float32)
        idx = np.argmin(rmse[:,2])
        rmse_min = rmse[idx]
        print rmse_min
        v2_rms_min = ((rmse_min[1]/np.pi)*np.arctan(x/rmse_min[0]))
        ##Atan plus creep##
        x = x+offset
        rmse_c=[]
#        Q = arange(0.00001,rmse_min[0],100)

        d1=0.01

        rmse_c = []
        for d2 in D:
            for s in V:
                v1 = sol[0]+sol[1]+v2_rms_min+((s/np.pi)*(np.arctan(x/d1)-np.arctan(x/d2)))
                residual = v1 - avgInSAR
                rms = np.sqrt((sum((residual)**2,0))/len(transect))
                rmse_c.append([d1,d2, s, rms])

        rmse_c = np.array(rmse_c,np.float32)
        idx = np.argmin(rmse_c[:,3])
        rmse_c_min = rmse_c[idx]
        print rmse_c_min
        v1_rms_min = (rmse_c_min[2]/np.pi)*(np.arctan(x/rmse_c_min[0])-np.arctan(x/rmse_c_min[1]))

        m_c =v1_rms_min+v2_rms_min+sol[0]+sol[1]

        ####################
        fig = plt.figure()
        plt.rcParams.update({'font.size': 22})
        fig.set_size_inches(20,8)
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax=fig.add_subplot(111)

        ##PLOT average and standard deviation##
        ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
        for i in np.arange(0.0,1.01,0.01):
            ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
        for i in np.arange(0.0,1.01,0.01):
            ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
        ax.plot(xp,avgInSAR*1000.0,'r-',label = 'Average velocity')
        label1 = 'Interseismic only: Locking depth: '+str(int(rmse_min[0]))+' meters - Slip rate: '+str(int(abs(round(rmse_min[1]*1000.,2))))+' mm/yr'

        ax.plot(xp,((sol[0]+v2_rms_min+sol[1])*1000.),'b--',label=label1)

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
