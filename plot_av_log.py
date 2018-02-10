#! /usr/bin/env python2
#@author: Emre Havazli

import sys
from numpy import *
import matplotlib.pyplot as plt

filepath = 'average.log'
vel=[]
with open(filepath) as fp:
   line = fp.readline()
   cnt=1
   while line:
       line = fp.readline()
       cnt += 1
       words = line.split(" ")
       if words[0] == 'Average':
          vel.append(words[-1])
       else: pass
f = open('vel.list','w')
for x in vel:
    f.write(x)
f.close
dist = arange(len(vel))
for i in range(len(vel)):
    plt.plot(dist[i],float(vel[i])*10000.0, 'ro')
plt.axis([0, 1000,0, 2])
plt.savefig('STDvsDIST.png',bbox_inches="tight")
