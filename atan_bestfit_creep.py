#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
from numpy import *
import matplotlib.pyplot as plt
import scipy.io
import math
from scipy.optimize import brute

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

transectmat = scipy.io.loadmat('transect.mat')
transect = transectmat['dataset'][0][0][1]
avgInSAR = nanmean(transect,axis=1)
#avgInSAR = nanmean(transectmat['dataset'][0][0][1],axis=1)
stdInSAR = nanstd(transect,axis=1)
#stdInSAR = nanstd(transectmat['dataset'][0][0][1],axis=1)
#transect = avgInSAR*1000.0
#std_transect = stdInSAR*1000.0

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

D = arange(100.,20000.,100.)
V = arange(-0.010,-0.10,-0.001)
rmse=[]

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
v2_rms_min = sol[0]+((rmse_min[1]/pi)*arctan(x/rmse_min[0]))+sol[1]


##Atan plus creep##
#D = arange(100.,20000.,100.)
#V = arange(-0.005,-0.100,-0.001)
rmse_c=[]

for d1 in D:
    for d2 in D:
          for s in V:
            v1 = sol[0]+(s/pi)*(arctan(x/d1)-arctan(x/d2))+sol[1]
#            v2 = sol[0]+((s/pi)*arctan(x/rmse_min[0]))+sol[1]
#            m = v1+v2_rms_min
            residual = v1 - avgInSAR
            rms = sqrt((sum((residual)**2,0))/len(transect))
            rmse_c.append([d1,d2, s, rms])
#            print 'RMSE of '+ str(d)+'_'+str(d+100.)+' meters ' + str(s)+' m/yr: '+str(rms)

rmse_c = array(rmse_c,float32)
idx = argmin(rmse_c[:,3])
rmse_c_min = rmse_c[idx]
print rmse_c_min
v1_rms_min = sol[0]+(rmse_c_min[2]/pi)*(arctan(x/rmse_c_min[0])-arctan(x/rmse_c_min[1]))+sol[1]
m_c = v1_rms_min+v2_rms_min


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

ax.plot(xp,(v1_rms_min*1000.),'k--',label='inversion')

label = 'Best Fit: '+str(int(rmse_min[0]))+' meters '+str(int(abs(round(rmse_min[1]*1000.,2))))+' mm/yr'
ax.plot(xp,(v2_rms_min*1000.),'b--',label=label)

label = 'Best Fit: '+str(int(rmse_c_min[0]))+' meters '+str(int(rmse_c_min[1]))+' meters '+str(rmse_c_min[2])+' mm/yr'
ax.plot(xp,(m_c*1000.),'k-',label=label)

ax.legend(loc='upper right')
plt.ylabel('Velocity (mm/yr)')
plt.xlabel('Distance (km)')
fig.savefig('atan_best_'+str(rmse_min[0])+'_'+str(rmse_min[1])+'.png')
plt.close()
