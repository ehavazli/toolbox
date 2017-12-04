#! /usr/bin/env python

##########################
#Author : Heresh Fattahi #
##########################

import os
import sys
from numpy import sqrt, mean
import numpy as np
import h5py
#import matplotlib.pyplot as plt
import argparse
import datetime
import time
from datetime import timedelta, date
import dloadUtil as dl
from delayTimeseries import timeseries


def createParser():
    parser = argparse.ArgumentParser( description='estimate InSAR uncertainty due to the tropospheric delay')

    parser.add_argument('-d', '--source', dest='data_source', type=str, required=True,
            help='source of the data to compute uncertainty')
    parser.add_argument('-s', '--start', dest='start_date', type=str, required=True,
            help='start date of the troposphere time-series data')
    parser.add_argument('-e', '--end', dest='end_date', type=str, default=None,
            help='end date of the troposphere time-series data')
    parser.add_argument('-t', '--time', dest='time', type=str, required=True,
            help='UTC time')
    parser.add_argument('-w', '--time_window', dest='time_windwo', type=str, default=18000,
            help='time window in seconds to search for MODIS acquisitions. Default is 18000 seconds.')
    parser.add_argument('-b', '--bbox', dest='bbox', type=str, required=True, default=None,
    help='Lat/Lon Bounding box: SNWE (South North West East)')
    parser.add_argument('-l', '--sar_dates', dest='sar_dates', type=str, default=None,
    help='List of the SAR acquisition dates')
    parser.add_argument('-i', '--incidence_angle', dest='incidence_angle', type=float, default=None,
            help='incidence angle to project the zenith delay to the slant direction.')
    parser.add_argument('-r', '--reference_pixel', dest='ref_Pix', type=str, default=None,
            help='reference pixel to calculate the relative uncertainty: lat lon')
    parser.add_argument('-S','--steps', dest='steps', type=str, default='all',
            help='steps of processing : download,seasonal,uncertainty | Default: all')
    parser.add_argument('-z','--sample_size', dest='sampleSize', type=int, default=500,
            help='sample size to calculate uncertainty vs distance | Default: 500')

    parser.add_argument('--modis_timeseriesFile', dest='modisTimeseriesFile', type=str, default='modis.h5',
            help='Name of the h5 file that contains the timeseries of modis wet delay. Default: modis.h5')

    parser.add_argument('--stochastic_file', dest='stochasticFile', type=str, default='stochastic.h5',
            help='Name of the h5 file that contains the timeseries of the stochastic wet delay. Default: stochastic.h5')

    #parser.add_argument('-w', '--wavelength', dest='wavelength', type=str, default=None,
    #        help='SAR wavelength')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    inps = parser.parse_args(args=iargs)
    inps.steps = inps.steps.split()
    if 'all' in inps.steps:
        inps.steps = ['download','seasonal','uncertainty']
    return inps


def velocity_uncertainty_vs_distance(inps):

    ###################
    if os.path.exists(inps.sar_dates_list):
       sar_dates=[]
       for line in open(inps.sar_dates_list):
          sar_dates.append(line.strip())
    else:
       sar_dates = inps.sar_dates_list.split(',')
    ################### 

  #  tsObj = timeseries(stochasticFile)
    tsObj = timeseries(inps.stochasticFile)
   # tsObj = timeseries()
    tsObj.open()
    tsObj.load()
    tsObj.sample(inps.sampleSize)
    uncertainty,distance = tsObj.uncertainty_vs_distance(sar_dates)    
     
    ###################
    #plot
    import matplotlib.pyplot as plt
    distance = distance/1000.0
    uncertainty = uncertainty*1000.0
    figName='uncertainty_vs_dist.png'
    plt.plot(distance,uncertainty,'k+',ms=2)
    plt.show()
    

def statistics(inps):
    tsObj = timeseries(inps.stochasticFile)
    tsObj.open()
    tsObj.load()
    absStd, relStd, covFile = tsObj.statistics(inps)
    tsObj.close()
    return absStd, relStd, covFile

def estimate_seasonal(inps):
    tsObj = timeseries(inps.modisTimeseriesFile)
    tsObj.open()
    tsObj.load()
    tsObj.estimate_seasonal(inps)
    tsObj.close()


def velocity_uncertainty(realtive_std_file, inps):
    from delayTimeseries import write_to_h5
    h5 = h5py.File(realtive_std_file,'r')
    dset = h5['velocity'].get('velocity')
    realtive_std = dset[0:dset.shape[0],0:dset.shape[1]]
    if os.path.exists(inps.sar_dates):
       sar_dates=[]
       for line in open(inps.sar_dates): 
          sar_dates.append(line.strip())
    else:
       sar_dates = inps.sar_dates.split(',')

    t=[]
    for d in sar_dates:
       ti = datetime.datetime.strptime(d, "%Y%m%d")
       t.append(ti.year + (ti.month-1)/12. + ti.day/365.)

    t_bar = mean(t)
    tt = sqrt(np.sum((t-t_bar)**2))
    relative_std_velocity = realtive_std/tt
    write_to_h5(relative_std_velocity,'relative_std_velocity.h5','velocity',realtive_std_file)

def download(inps):
    if inps.data_source == 'MODIS':
        h5 = dl.download_modis(inps)
    elif inps.data_source in ['ERAI','ECMWF','MERRA','NARR']:
        h5 = dl.download_atmosphereModel(inps)
    elif inps.data_source == 'GPS':
        h5 = dl.download_GPS(inps)
    else:
        print 'Data source is not known. supported options are : MODIS, ERAI, GPS'
        sys.exit(1)
    return h5


##############################################################

def main(iargs=None):

  inps = cmdLineParse(iargs)
  if 'download' in inps.steps or inps.steps == 'all':
      h5 = download(inps)

  if 'seasonal' in inps.steps or inps.steps == 'all':
      estimate_seasonal(inps)

  if 'uncertainty' in inps.steps or inps.steps == 'all':
      statistics(inps)

      absolute_std, relative_std, covarinace = statistics(inps)
      if inps.sar_dates is not None:
          relative_std_velocity = velocity_uncertainty(relative_std, inps)

  if 'uncertainty_vs_distance' in inps.steps or inps.steps == 'all':
      velocity_uncertainty_vs_distance(inps) #, inps.sar_dates, inps.sampleSize)


if __name__ == "__main__":

  # Main engine
  main()
