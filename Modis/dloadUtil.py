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
import pyaps as pa

def download_modis(inps):

    S,N,W,E = [val for val in inps.bbox.split()]
    start_year,start_month,start_day=[int(val) for val in inps.start_date.split('-')]
    end_year,end_month,end_day=[int(val) for val in inps.end_date.split('-')]
    start_date = date(start_year,start_month,start_day)
    end_date = date(end_year,end_month,end_day)
    for single_date in daterange(start_date, end_date):
            print single_date.strftime("%Y-%m-%d")
            t = str(single_date.year) + '-' + str(single_date.month) + '-' + str(single_date.day) + 'T' + inps.time
            print t
            cmd = 'get_modis_v3.py -r ' + W + '/' + E + '/' + S + '/' + N + ' -t ' + t + ' -w 18000 -s oscar1 -p terra ' #+ outNameTerra
                #cmd='./get_modis_v3.py -r 63.5/69/25.5/33 -t '+t+' -w 18000 -s oscar1 -o '+outName
            print cmd
            os.system(cmd)

            cmd = 'get_modis_v3.py -r ' + W + '/' + E + '/' + S + '/' + N + ' -t ' + t + ' -w 18000 -s oscar1 -p aqua ' #+ outNameAqua
            print cmd
            os.system(cmd)

    modisPath='./*grd'
    fileList = glob.glob(modisPath)
    fileList.sort()

    date_yrs=[]
    filelistDict={}
    for f in fileList:
       ymd,yr,HMS = get_date(f)
       date_yrs.append(yr)
       filelistDict[str(yr)]=[f,ymd+'T'+HMS]

    print '***************************'
   # h5OutName='modis.h5'
    h5OutName = inps.modisTimeseriesFile
    print 'writing '+ h5OutName
    h5=h5py.File(h5OutName,'w')
    g=h5.create_group('timeseries')
    date_yrs.sort()

    pwv,Y_FIRST,Y_END,X_FIRST,X_END=read_modis(f)
    onesIndx=ones(shape(pwv))
    mask=zeros(shape(pwv))

    for yr in date_yrs:
       f = filelistDict[str(yr)][0]
       ymdHMS = filelistDict[str(yr)][1]
       pwv,Y_FIRST,Y_END,X_FIRST,X_END=read_modis(f)
       zwd=pwv2zwd(pwv)
       idx=~isnan(zwd)
       mask[idx]=mask[idx]+onesIndx[idx]

       dset=g.create_dataset(ymdHMS,data=zwd,compression='gzip')


    g=h5.create_group('mask')
    dset=g.create_dataset('mask',data=mask,compression='gzip')

    LENGTH,WIDTH=mask.shape
    Y_STEP=(Y_END-Y_FIRST)/(LENGTH-1)
    X_STEP=(X_END-X_FIRST)/(WIDTH-1)

    h5['timeseries'].attrs['Y_FIRST']=Y_FIRST
    h5['timeseries'].attrs['Y_END']=Y_END
    h5['timeseries'].attrs['X_FIRST']=X_FIRST
    h5['timeseries'].attrs['X_END']=X_END
    h5['timeseries'].attrs['WAVELENGTH']=1
    h5['timeseries'].attrs['FILE_LENGTH']=LENGTH
    h5['timeseries'].attrs['WIDTH']=WIDTH
    h5['timeseries'].attrs['Y_STEP']=Y_STEP
    h5['timeseries'].attrs['X_STEP']=X_STEP

    h5mask=h5py.File('CloudMask.h5','w')
    g=h5mask.create_group('mask')
    dset=g.create_dataset('mask',data=mask,compression='gzip')
    g.attrs['Y_FIRST']=Y_FIRST
    g.attrs['Y_END']=Y_END
    g.attrs['X_FIRST']=X_FIRST
    g.attrs['X_END']=X_END

    h5mask.close()
    h5.close()
    return 'modis.h5' 



#def download_atmosphereModel(inps, atmSource):
def download_atmosphereModel(inps):  #EMRE
    S,N,W,E = [val for val in inps.bbox.split()] #EMRE
    start_year,start_month,start_day=[int(val) for val in inps.start_date.split('-')] #EMRE
    end_year,end_month,end_day=[int(val) for val in inps.end_date.split('-')] #EMRE
    start_date = date(start_year,start_month,start_day) #EMRE
    end_date = date(end_year,end_month,end_day) #EMRE
    d1 = start_date
    d2 = end_date
 
    dateList=[]
#    d1 = datetime.date(2002,1,1)
#    d2 = datetime.date(2002,2,1)
    diff = d2 - d1
    day_step = 1 #EMRE
    hr = '18'
    for i in range(0,diff.days+1,day_step):
        dd=(d1 + datetime.timedelta(i)).isoformat()
        dateList.append(dd.replace('-',''))

#    if atmSource in ['ecmwf','ECMWF']:
    if inps.data_source in ['ecmwf','ECMWF']:  #EMRE
       gribSource='ECMWF'
       if not os.path.isdir('ECMWF'):
          print 'making directory: ECMWF'
          os.mkdir('ECMWF')

       ecmwf_file=[]
       for d in dateList:
         ecm='./ECMWF/ERA-Int_'+d+'_'+hr+'.grb'
         ecmwf_file.append(ecm)
         print [d]
         if not os.path.isfile(ecm):
            pa.ECMWFdload([d],hr,'./ECMWF/')
         else:
            print ecm + ' already exists.'

#    elif atmSource in ['narr','NARR']:
    elif inps in ['narr','NARR']: #EMRE
       gribSource='NARR'
       if not os.path.isdir('NARR'):
          print 'making directory: NARR'
          os.mkdir('NARR')

       ecmwf_file=[]
       for d in dateList:
         ecm='./NARR/narr-a_221_'+d+'_'+hr+'00_000.grb'
         ecmwf_file.append(ecm)
         print [d]
         if not os.path.isfile(ecm):
            pa.NARRdload([d],hr,'./NARR/')
         else:
            print ecm + ' already exists.'

#    elif atmSource in ['era','ERA']:
    elif inps in ['era','ERA']:
      gribSource='ERA'
      if not os.path.isdir('ERA'):
         print 'making directory: ERA'
         os.mkdir('ERA')
         
      ecmwf_file=[]
      for d in dateList:
        ecm='./ERA/ERA_'+d+'_'+hr+'.grb'
        ecmwf_file.append(ecm)
        print [d]
        if not os.path.isfile(ecm):
           pa.ERAdload([d],hr,'./ERA/')
        else:
           print ecm + ' already exists.'

#    elif atmSource in ['merra','MERRA']:
    elif inps in ['merra','MERRA']:
      gribSource='MERRA'
      if not os.path.isdir('MERRA'):
         print 'making directory: MERRA'
         os.mkdir('MERRA')

      ecmwf_file=[]
      for d in dateList:
        ecm='./MERRA/merra-'+d+'-'+hr+'.hdf'
        ecmwf_file.append(ecm)
        print [d]
        if not os.path.isfile(ecm):
           pa.MERRAdload([d],hr,'./MERRA/')
        else:
           print ecm + ' already exists.'

    else:
#       Usage();
       print 'FAILED' #EMRE
       sys.exit(1)

    print '*******************************************************************************'
    print 'Calculating delay for each epoch.'
    h5phs=h5py.File('aps.h5','w')
    outName='ECMWF.h5'
    #h5apsCor=h5py.File(outName,'w')
   # group=h5apsCor.create_group('timeseries')
    group_phs=h5phs.create_group('timeseries')

    demCoord='geo'
    print ecmwf_file[0]
    #demCoord='radar'
    if demCoord=='radar':
       aps1 = pa.PyAPS_rdr(str(ecmwf_file[0]),demFile,grib=gribSource,verb=True,Del=DelayType)
    else:
       aps1 = pa.PyAPS_geo(str(ecmwf_file[0]),demFile,grib=gribSource,verb=True,Del=DelayType)

    phs1 = np.zeros((aps1.ny,aps1.nx))
    print aps1.ny
    print aps1.nx
   # aps1.getdelay(phs1,inc=inc_angle,wvl=wavelength)
    aps1.getdelay(phs1)
    dset = group_phs.create_dataset(dateList[0], data= phs1, compression='gzip')
#    phs1=(phs1 - phs1[yref,xref])*wavelength/(4*np.pi)

   # dset = group.create_dataset(dateList[0], data= phs1- phs1, compression='gzip')

    for i in range(1,len(ecmwf_file)):
       ecm=ecmwf_file[i]
       print ecm
       if demCoord=='radar':
          aps = pa.PyAPS_rdr(str(ecm),demFile,grib=gribSource,verb=True,Del=DelayType)
       else:
          aps = pa.PyAPS_geo(str(ecm),demFile,grib=gribSource,verb=True,Del=DelayType)
       phs = np.zeros((aps.ny,aps.nx))
     #  aps.getdelay(phs,inc=inc_angle,wvl=wavelength)
       aps.getdelay(phs)
     #  phs=(phs - phs[yref,xref])*wavelength/(4*np.pi)
     #  phs=phs-phs1
       dset = group_phs.create_dataset(dateList[i], data= phs, compression='gzip')
    #   dset1 = h5timeseries['timeseries'].get(dateList[i])
    #   data1 = dset1[0:dset1.shape[0],0:dset1.shape[1]]
    #   dset = group.create_dataset(dateList[i], data= data1+phs, compression='gzip')

    for key,value in h5timeseries['timeseries'].attrs.iteritems():
     # group.attrs[key] = value
        group_phs.attrs[key] = value



def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


def get_date(f):
    fname = os.path.basename(f)
    ymd = fname.split('T')[0].split('_')[-1]
    HMS = fname.split('T')[1].split('.grd')[0]
    yr,m,d = ymd.split('-')
    H,M,S = HMS.split(':')
    if len(m)==1:
      m='0'+m
    if len(d)==1:
      day='0'+d
    yy = float(yr) + (float(m)-1.)/12. + float(d)/365 + float(H)/24.0/365. + float(M)/60.0/24.0/365. +  float(S)/3600.0/24.0/365.
    print ymd,yy,HMS 
    return ymd,yy,HMS
    
def pwv2zwd(pwv):

    zwd=6.2*pwv/100. # for more accurate results the Mapping factor should be calculated
    return zwd
    
def zwd2swd(zwd,theta):
    theta=theta*pi/180.
    swd=zwd/cos(theta)
    return swd
    
def read_modis(file):
    f= netcdf.netcdf_file(file,'r')
    lat=f.variables['y']
    lon=f.variables['x']
    pwv=flipud(f.variables['z'][:,:])
    f.close()
    return pwv,lat[-1],lat[0],lon[0],lon[-1]




