#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Ellipse
import scipy.io
import math
###############################################################################

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6373.0 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def gps2los(g_vel, look_angle):
    L = sin(look_angle)*g_vel

    return L

def eigsorted(cov):
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

transectmat = scipy.io.loadmat('transect.mat')
transect = transectmat['dataset'][0][0][1]
avgInSAR = nanmean(transect,axis=1)
stdInSAR = nanstd(transect,axis=1)



# ##GPS##
# bgnt = [26.570, 40.932]
# doku = [26.706, 40.739]
# kvam = [26.871, 40.601]
# kvam2 = [26.871, 40.601]
# sev2 = [26.880, 40.396]
# #velocities are in mm/yr##
# bgnt_vel = 3.11
# doku_vel = 2.89
# kvam_vel = -9.62
# kvam2_vel = -11.08
# sev2_vel = -16.52
# ##GPS vel 2 LOS
# bgnt_los = gps2los(bgnt_vel,34.3)
# doku_los = gps2los(doku_vel,34.3)
# kvam_los = gps2los(kvam_vel,34.3)
# kvam2_los = gps2los(kvam2_vel,34.3)
# sev2_los = gps2los(sev2_vel,34.3)
# print 'KVAM2_LOS: '+str(kvam2_los)
##
transect_dist = transectmat['dataset'][0][0][2]
transect_lat_first = transectmat['dataset'][0][0][0][0][25]
transect_lon_first = transectmat['dataset'][0][0][3][0][25]
transect_lat_last = transectmat['dataset'][0][0][0][-1][25]
transect_lon_last = transectmat['dataset'][0][0][3][-1][25]

transect_middle_line = line([transect_lat_first,transect_lon_first],[transect_lat_last,transect_lon_last])
fault_line = line([40.6033690,26.834260],[40.7506800,27.335392])
intersect = intersection(transect_middle_line,fault_line)
dist = distance([transect_lat_first,transect_lon_first],intersect)*1000.0
dist2 = distance(intersect,[transect_lat_last,transect_lon_last])*1000.0
##
# dist_bgnt = distance(bgnt,intersect)/1000.
# dist_doku = distance(doku,intersect)/1000.
# dist_kvam = distance(kvam,intersect)/1000.
# dist_kvam2 = distance(kvam2,intersect)/1000.
# dist_sev2 = distance(sev2,intersect)/1000.

depth = []
slip = []
r=[]
varModel = []
for q in xrange(5):
    avgVel = []
    for i in xrange(len(avgInSAR)):
        avgVel.append(random.normal(avgInSAR[i], stdInSAR[i]))
    avgVel = array(avgVel)

##1-D space##
    x = linspace(-dist,dist2,num=len(transect),endpoint=True)
    xp = x/1000.

##Linear Component##
    G = ones([len(transect),2])
    G[:,0] = xp
    G_inv = dot(linalg.inv(dot(G.T,G)), G.T)
    G_inv = array(G_inv, float32)
    sol = dot(G_inv,avgVel)
    k = dot(G,sol)

    D = arange(100.,20000.,5.)
    V = arange(-0.010,-0.10,-0.0005)
    rmse=[]
    for d in D:
        for s in V:
            v2 = sol[0]+((s/pi)*arctan(x/d))+sol[1]
            residual = v2 - avgVel
            rms = sqrt((sum((residual)**2,0))/len(transect))
            rmse.append([d, s, rms])
#        print 'RMSE of '+ str(d)+' meters ' + str(s)+' m/yr: '+str(rms)


##PLOT##
        # fig = plt.figure()
        # fig.set_size_inches(10,4)
        # ax = plt.Axes(fig, [0., 0., 1., 1.], )
        # ax=fig.add_subplot(111)
        # ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
        # for i in arange(0.0,1.01,0.01):
        #    ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
        # for i in arange(0.0,1.01,0.01):
        #    ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
        # ax.plot(xp,avgInSAR*1000.0,'r-')
        # # ax.plot(xp,k*1000.,'k-')
        # ax.plot(xp,v2*1000.,'b--')
        # plt.ylabel('Velocity (mm/yr)')
        # plt.xlabel('Distance (km)')
        # fig.savefig('atan_best_'+str(d)+'_'+str(s)+'.png')
        # plt.close()

    rmse = array(rmse,float32)
    idx = argmin(rmse[:,2])
    rmse_min = rmse[idx]
    print str(q)+' '+ str(rmse_min)
    depth.append(rmse_min[0])
    slip.append(rmse_min[1])
    r.append(rmse_min[2])
    varModel.append(rmse_min[2]**2)
#    v2_rms_min = sol[0]+((rmse_min[1]/pi)*arctan(x/rmse_min[0]))+sol[1]
# print "MEAN DEPTH: "+str(mean(depth))
# print "MEAN SLIP: "+str(mean(slip))
# print "STD DEPTH: "+str(std(depth))
# print "STD SLIP: "+str(std(slip))
stdModel = sqrt(sum(varModel))
slip[:] = [x * 1000.0 for x in slip]
d_s_r = array(zip(depth,slip,r))

###############################################################################

###############################################################################


##1-D space##
x = linspace(-dist,dist2,num=len(transect),endpoint=True)
xp = x/1000.

##Linear Component##
G = ones([len(transect),2])
G[:,0] = xp
G_inv = dot(linalg.inv(dot(G.T,G)), G.T)
G_inv = array(G_inv, float32)
sol = dot(G_inv,avgInSAR)
k = dot(G,sol)

D = arange(100.,20000.,5.)
V = arange(-0.010,-0.10,-0.0005)
rmse=[]
for d in D:
    for s in V:
        v2 = sol[0]+((s/pi)*arctan(x/d))+sol[1]
        residual = v2 - avgInSAR
        rms = sqrt((sum((residual)**2,0))/len(transect))
        rmse.append([d, s, rms])

rmse = array(rmse,float32)
idx = argmin(rmse[:,2])
rmse_min = rmse[idx]
print "Inversion: " + str(rmse_min)
v2_rms_min = sol[0]+((rmse_min[1]/pi)*arctan(x/rmse_min[0]))+sol[1]

fig = plt.figure()
plt.rcParams.update({'font.size': 22})
fig.set_size_inches(10,10)
ax = plt.Axes(fig, [0., 0., 1., 1.], )
ax=fig.add_subplot(111)

cov = cov(slip, depth)
vals, vecs = eigsorted(cov)
theta = degrees(arctan2(*vecs[:,0][::-1]))
for nstd in xrange(1,3):        ####nstd is 1sigma, 2sigma, 3sigma
    w, h = 2 * nstd * sqrt(vals)
    ell = Ellipse(xy=(mean(slip), mean(depth)),
              width=w, height=h,
              angle=theta, color='black')
    ell.set_facecolor('none')
    ax.add_artist(ell)

plt.scatter(slip,depth)
plt.scatter(rmse_min[1]*1000.,rmse_min[0], s=60, c='red', marker='*')
plt.ylabel('Depth (m)')
plt.xlabel('Slip (mm/yr)')
# plt.xlim(10,20)
# plt.ylim(4000,7000)
fig.savefig('depth_vs_slip.png')
plt.close()
print "PLOTTED"


depth.append(rmse_min[0])
slip.append(rmse_min[1])
stdDepth = math.ceil(std(depth))
stdSlip = math.ceil(std(slip))


fig = plt.figure()
plt.rcParams.update({'font.size': 22})
fig.set_size_inches(20,8)
ax = plt.Axes(fig, [0., 0., 1., 1.], )
ax=fig.add_subplot(111)
ax.plot(xp,transect*1000.0,'o',ms=1,mfc='Black', linewidth='0')
# for i in arange(0.0,1.01,0.01):
#     ax.plot(xp, (avgInSAR-i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
#     for i in arange(0.0,1.01,0.01):
#         ax.plot(xp, (avgInSAR+i*stdInSAR)*1000, '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
#ax.plot(xp,avgInSAR*1000.0,'r-',label = 'Average velocity')
#ax.plot(xp,(k*1000.),'k-',label='inversion')
label = 'Best Fit: '+str(int(rmse_min[0]))+'$\pm$'+str(int(stdDepth))+'m; '+str(int(abs(round(rmse_min[1]*1000.,2))))+'$\pm$'+str(int(stdSlip))+'mm/yr'
# for i in arange(0.0,1.01,0.01):
#     ax.plot(xp, ((v2_rms_min*1000.)-(i*stdModel)*1000.), '-',color='#DCDCDC',alpha=0.5)#,color='#DCDCDC')#'LightGrey')
#     for i in arange(0.0,1.01,0.01):
#         ax.plot(xp, ((v2_rms_min*1000.)+(i*stdModel)*1000.), '-',color='#DCDCDC',alpha=0.5)#'LightGrey')
ax.plot(xp,(v2_rms_min*1000.),'b--',label=label)

#ax.plot(dist_kvam2,kvam2_los,'o',label='KVM2')

ax.legend(loc='upper right')
plt.ylabel('Velocity (mm/yr)')
plt.xlabel('Distance (km)')
fig.savefig('atan_best_'+str(rmse_min[0])+'_'+str(rmse_min[1])+'.png')
plt.close()
#     print "PLOTTED"
