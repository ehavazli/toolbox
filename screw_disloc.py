#! /usr/bin/env python2
#@author: Emre Havazli

import os
import sys
from numpy import *
import matplotlib.pyplot as plt
import scipy.io
import math

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
transect_meter = nanmean(transectmat['dataset'][0][0][1],axis=1)
transect = transect_meter*1000.0

transect_dist = transectmat['dataset'][0][0][2]
transect_lat_first = transectmat['dataset'][0][0][0][0][25]
transect_lon_first = transectmat['dataset'][0][0][3][0][25]
transect_lat_last = transectmat['dataset'][0][0][0][-1][25]
transect_lon_last = transectmat['dataset'][0][0][3][-1][25]

transect_middle_line = line([transect_lat_first,transect_lon_first],[transect_lat_last,transect_lon_last])
fault_line = line([40.6033690,26.834260],[40.7506800,27.335392])
intersect = intersection(transect_middle_line,fault_line)
#intersect_dist = sqrt((transect_lat_last-transect_lat_first)**2+(transect_lon_last-transect_lon_first)**2)
dist = distance([transect_lat_first,transect_lon_first],intersect)*1000.0
dist2 = distance(intersect,[transect_lat_last,transect_lon_last])*1000.0
# transect_middle_line_lat = linspace(transect_lat_first, transect_lat_last, num = len(transect_meter), endpoint = True)
# transect_middle_line_lon = linspace(transect_lon_first, transect_lon_last, num = len(transect_meter), endpoint = True)
# transect_middle_line = column_stack((transect_middle_line_lat,transect_middle_line_lon))

# fault_lat = linspace(40.6033690, 40.7506800, num = 100000, endpoint = True)
# fault_lon = linspace(26.834260, 27.335392, num = 10000, endpoint = True)
# fault = column_stack((transect_middle_line_lat,transect_middle_line_lon))
#
# intersection_lat = intersect1d(fault_lat,transect_middle_line_lat)
# print dist
# print dist2
# print dist+dist2
# sys.exit()



V = -0.015 #Constant velocity
D = 12000.
d = 800.
d0 = 200.
##1-D space##
x = linspace(-dist,dist2,num=len(transect),endpoint=True)
xp = x/1000.
# y = arange(-377.5,377.5,1)
# yp = y/1000.

##Linear Component##
G = ones([len(transect),2])
G[:,0] = xp
G_inv = dot(linalg.inv(dot(G.T,G)), G.T)
G_inv = array(G_inv, float32)
sol = dot(G_inv,transect_meter)
k = dot(G,sol)


##Shallow creep between depths of d0 and d
v1 = (V/pi)*(arctan(x/d0)-arctan(x/d))
dv1 = (V/(pi*d0))*1./(1.+(x/d0)**2) - (V/(pi*d))*1./(1.+(x/d)**2)


##Free slip for depths greater than D
v2 = sol[0]+(V/pi)*arctan(x/D)+sol[1]
dv2 = (V/(pi*D))*1./(1.+(x/D)**2)
##PLOT
fig = plt.figure(1)
ax = fig.add_axes([0.1,0.1,0.8,0.8])

plt.plot(xp,((v1+v2)*1000.),'b-',xp,(v2*1000.),'r--')
#plt.plot(xp,((v1+v2)*1000.),'b-')
plt.ylabel('Displacement (mm/yr)')
plt.xlabel('Distance (km)')
# plt.show()
#plt.plot(xp,((v1+v2)*1000.),'b-',xp,(v2*1000.),'r--',xp,transect,'ko')
plt.plot(xp,transect,'ro')
# plt.ylabel('Displacement (mm/yr)')
# plt.xlabel('Distance (km)')
plt.show()

##Compute displacement due to a dipping fault

# V = -10.
# alpha = 30.*pi/180.
# D = 12.
# x = arange(-40,40)
# cosa = cos(alpha)
# sina = sin(alpha)
# num = x*(cosa**2)
# dem = D-(x*sina*cosa)
# vel0 = V*arctan2(x,D)/pi
# vel1 = V*(arctan2(num,dem)/pi-alpha/pi)

# fig = plt.figure(2)
# plt.plot(x,vel0,x,vel1,'r--')
# plt.ylabel('Displacement (mm/yr)')
# plt.xlabel('Distance (km)')
# plt.show()
