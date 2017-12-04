#! /usr/bin/env python

##########################
#Author : Heresh Fattahi #
##########################

import os
import sys
import glob
from numpy import sum,isnan,float,size,flipud,pi,shape,ones,zeros
import numpy as np
import h5py
from scipy.io import netcdf
import matplotlib.pyplot as plt
import argparse
import datetime
import time
from datetime import timedelta, date


class timeseries(object):

    def __init__(self, file=None):
        self.file = file

    def open(self):
        self.h5 = h5py.File(self.file,'r')
        self.dateList = self.h5['timeseries'].keys()
        dset = self.h5['timeseries'].get(self.dateList[0])
        self.rows,self.cols = dset.shape
        self.numPixels = self.rows*self.cols
        self.numDates = len(self.dateList)
        self.lat_first = float(self.h5['timeseries'].attrs['Y_FIRST'])
        self.lon_first = float(self.h5['timeseries'].attrs['X_FIRST'])
        self.lat_step = float(self.h5['timeseries'].attrs['Y_STEP'])
        self.lon_step = float(self.h5['timeseries'].attrs['X_STEP'])

        self.lat = np.arange(self.lat_first, self.lat_first + self.rows*self.lat_step, self.lat_step)
        self.lon = np.arange(self.lon_first, self.lon_first + self.cols*self.lon_step, self.lon_step)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.lon = self.lon.flatten()
        self.lat = self.lat.flatten()
                
    def close(self):
        self.h5.close()
    
    def load(self):
        self.Data = np.zeros((len(self.dateList),self.rows*self.cols),np.float32)
        lenDates = len(self.dateList)
    
        for i in range(lenDates):
            self.Data[i] = np.reshape(self.h5['timeseries'].get(self.dateList[i]),(1,self.numPixels))

   
    def sample(self , numSamples = 500 , mask=None):
        import random  
        print '*******************************************'
        print 'Number of pixels : ' + str(self.numPixels)
        print 'Sample size : ' + str(numSamples)
        print '*******************************************'
        if numSamples > self.numPixels:
           print """Number of samples is larger than the number of pixels.
                    Fixed the number of samples to the number of pixels"""
           numSamples = self.numPixels

        if mask == None:
           mask = np.ones((self.numPixels))
         
        idx = np.arange(self.numPixels)
        idx_sampled = random.sample(idx[mask==1],numSamples)
        self.Data = self.Data[:,idx_sampled]
        self.lon = self.lon[idx_sampled]
        self.lat = self.lat[idx_sampled]
        self.idx = idx_sampled
        
    def std_timeseries(self,ref):
        
        numDates = self.Data.shape[0]
        dataRef = self.Data
        for i in range(numDates):
             dataRef[i,:] = dataRef[i,:] - dataRef[i,ref]
        
      #  self.Data = self.Data - self.Data[:,ref]
        self.relative_std = np.nanstd(dataRef,0)
        
    def std_velocity(self, sar_dates):

        t=[]
        for d in sar_dates:
          ti = datetime.datetime.strptime(d, "%Y%m%d")
          t.append(ti.year + (ti.month-1)/12. + ti.day/365.)

        t_bar = np.mean(t)
        tt=np.sqrt(np.sum((t-t_bar)**2))
        self.relative_std_velocity = self.relative_std/tt

    def distance(self,i):
        import pyproj
        geod = pyproj.Geod(ellps='WGS84')
        lon1 = self.lon[i]*np.ones(self.lon.shape)
        lat1 = self.lat[i]*np.ones(self.lon.shape)

        angle1,angle2,self.dist = geod.inv(lon1, lat1, self.lon, self.lat)
      #  angle1,angle2,dist = geod.inv(lon1, lat1, lon2, lat2)


    def uncertainty_vs_distance(self,sar_dates):
        numSamples = np.shape(self.Data)[1]
        distance = np.zeros((numSamples*numSamples))
        uncertainty = np.zeros((numSamples*numSamples))
        for i in range(numSamples):
            print i 
            self.std_timeseries(i)
            self.std_velocity(sar_dates)
            self.distance(i)
            uncertainty[i*numSamples:(i+1)*numSamples] = self.relative_std_velocity
            distance[i*numSamples:(i+1)*numSamples] = self.dist

        return uncertainty, distance

    def estimate_seasonal(self,inps):

       # h5timeseries = h5py.File(inps.modisTimeseriesFile,'r')
       # dateList = h5timeseries['timeseries'].keys()
        dates=[]
        for d in self.dateList:
            dates.append(datetime.datetime(*time.strptime(d,"%Y-%m-%dT%H:%M:%S")[0:6]))

        datevector=[]
        for i in range(len(dates)):
           datevector.append(np.float(dates[i].year) + np.float(dates[i].month-1)/12 + np.float(dates[i].day-1)/365 +
           float(dates[i].hour)/24.0/365. + float(dates[i].minute)/60.0/24.0/365. +  float(dates[i].second)/3600.0/24.0/365.)


        B=np.ones([len(datevector),5])
        B[:,1]=np.sin(2*np.pi*np.array(datevector))
        B[:,2]=np.cos(2*np.pi*np.array(datevector))
        B[:,3]=np.sin(4*np.pi*np.array(datevector))
        B[:,4]=np.cos(4*np.pi*np.array(datevector))
       # dset = h5timeseries['timeseries'].get(h5timeseries['timeseries'].keys()[0])
       # timeseries = np.zeros((self.numDates,self.rows,self.cols,np.float32)
       # for i in range(len(dateList)):
       #     timeseries[i] = h5timeseries['timeseries'].get(dateList[i])


  #  lt,rows,cols=np.shape(timeseries)
  #  numpixels=rows*cols

   # Data=np.zeros([lt,numpixels])
   # for i in range(lt):
   #    Data[i,:]=np.reshape(timeseries[i],[1,numpixels])

   # numPixels=rows*cols
        x=np.zeros((5,self.numPixels))
        for i in range(self.numPixels):
          ind=~np.isnan(self.Data[:,i])
          Bi=B[ind,:]
          d=self.Data[ind,i]
          try:
            B1 = np.dot(np.linalg.inv(np.dot(Bi.T,Bi)),Bi.T)
            B1 = np.array(B1,np.float32)
            x[:,i]=np.dot(B1,d)
          except:
            print 'skiping '
          if not np.remainder(i,10000): print 'Processing point: %7d of %7d ' % (i,self.numPixels)

  #####################################################
        S1=np.reshape(x[1,:],[self.rows,self.cols])
        C1=np.reshape(x[2,:],[self.rows,self.cols])
        S2=np.reshape(x[3,:],[self.rows,self.cols])
        C2=np.reshape(x[4,:],[self.rows,self.cols])
        intercept=np.reshape(x[0,:],[self.rows,self.cols])

        A1=np.sqrt(S1**2+C1**2)
        #write_to_h5(A1, 'annual_amplitude.h5', 'velocity',self.h5)
        write_to_h5(A1, 'annual_amplitude.h5', 'velocity',self.file)

        phase1=np.arctan2(C1,S1)
        write_to_h5(phase1, 'annual_phase.h5', 'velocity',self.file)

        A2=np.sqrt(S2**2+C2**2);
        write_to_h5(A2, 'semi_annual_amplitude.h5', 'velocity',self.file)

        phase2=np.arctan2(C2,S2);
        write_to_h5(phase2, 'semi_annual_phase.h5', 'velocity',self.file)

        stochastic = self.Data - np.dot(B,x)
   # h5out = h5py.File('stochastic.h5','w')
        h5out = h5py.File(inps.stochasticFile,'w')
        group = h5out.create_group('timeseries')
        for i in range(len(self.dateList)):
           dset = group.create_dataset(self.dateList[i], data=np.reshape(stochastic[i,:],[self.rows,self.cols]), compression='gzip')
      #  for key,value in h5timeseries['timeseries'].attrs.iteritems():
        for key,value in self.h5['timeseries'].attrs.iteritems():
           group.attrs[key] = value
        h5out.close()
    #return 'annual_amplitude.h5', 'annual_phase.h5' ,'semi_annual_amplitude.h5', stochastic

    def statistics(self,inps):
        print '*****************************'
        print 'Calculating the std of the stochastic delay'
        absolute_std = np.nanstd(self.Data,0)
        write_to_h5(np.reshape(absolute_std,[self.rows,self.cols]), 'absolute_std_timeseries.h5','velocity', inps.stochasticFile) #stochastic)
        print '*****************************'
        if inps.ref_Pix is not None:
           print 'Calculating the relative InSAR uncertainty (due to the stoachastic delay) wrt to pixel : ' + inps.ref_Pix
           print 'Incidence angle : ' + str(inps.incidence_angle)
           lat,lon = [float(val) for val in inps.ref_Pix.split()]
           xr = (lon - self.lon_first)/self.lon_step
           yr = (lat - self.lat_first)/self.lat_step
           ref_Pix=yr*self.cols+xr
           self.Data = self.Data/np.cos(inps.incidence_angle*np.pi/180.0)
           print "Calculating the covariance of the stoachstic delat at each pixel wrt the reference pixels "
    #    C=np.zeros((1,numpixels))
    #    d=np.zeros((2,lt))
    #    d[0,:]=Data[:,ref_Pix]

    #    for i in range(numpixels):
    #        d[1,:]=Data[:,i]
    #        C[0,i]=np.cov(d)[0,1]
    #        if not np.remainder(i,10000): print 'Processing point: %7d of %7d ' % (i,numpixels)

    #    print '*****************************'
    #    write_to_h5(np.reshape(C,[rows,cols]), 'covarinace.h5','velocity',stochastic)
    #    print '*****************************'
           print "Calculating the relative standard deviation of the stochastic delay"
           for i in range(len(self.dateList)):
             self.Data[i,:] = self.Data[i,:] - self.Data[i,ref_Pix]

           relative_std = np.nanstd(self.Data,0)
           write_to_h5(np.reshape(relative_std,[self.rows,self.cols]), 'relative_std_timeseries.h5','velocity', inps.stochasticFile)

      #  if inps.sar_dates is not None:
      #     velocity_uncertainty(np.reshape(relative_std,[rows,cols]),inps)

        else:
           C=None
           relative_std = None
        return 'absolute_std_timeseries.h5', 'relative_std_timeseries.h5', 'covarinace.h5'


def write_to_h5(dataset,outName,groupName,h5withAttributes):
    h5out = h5py.File(outName,'w')
    h5 = h5py.File(h5withAttributes,'r')
    group = h5out.create_group(groupName)
    dset = group.create_dataset(groupName, data=dataset, compression='gzip')
    try:
      for key,value in h5['timeseries'].attrs.iteritems():
          group.attrs[key] = value
    except:
      for key,value in h5['velocity'].attrs.iteritems():
          group.attrs[key] = value

    h5out.close()
    h5.close()

