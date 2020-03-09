#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Emre Havazli
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from shapely.geometry import Point
from shapely.ops import transform
from functools import partial
from osgeo import ogr
import pandas as pd
import numpy as np
import subprocess
import os, sys
import shutil
import timeit
import pyproj
import glob
import h5py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def createParser():
    '''
        Workflow to generate MintPy ready ARIA product stacks.
    '''

    import argparse
    parser = argparse.ArgumentParser(description='Generate mask and create stacks')
    # parser.add_argument('-f', '--file', dest='imgfile', type=str, required=True, help='ARIA file')
    parser.add_argument('-w', '--workdir', dest='workdir', default='./', help='Specify directory to deposit all outputs. Default is local directory where script is launched.')
    parser.add_argument('-c', '--csv', dest='csvFile', help='CSV file listing GPS stations (SiteID, Lon, Lat) can be obtained from "http://plugandplay.unavco.org:8080/unrgsac/gsacapi/site/form" but small editing is required')
    parser.add_argument('-tr', '--track', dest='trackDir', help='Products folder with the relevant track number')
    # parser.add_argument('-g', '--gps', dest='gpsSite', help='GPS stations given in the order as: SiteID, Lon, Lat')
    parser.add_argument('-d', '--dist', dest='distance', type=int, help='Distance around GPS station for mask generation (in km)')
    parser.add_argument('-s', '--step', dest='step', default='all', help='Choose step to do (all steps are run from scracth if no step is given) [generateMask,download,tsSetup,prepAria,timeseries,bootStrap,dloadGPS,mergeGPSup,skipdownload]')
    parser.add_argument('-t', '--temp', dest='template', default='smallbaselineApp.cfg',help='MintPy template file to used for time series processing')
    parser.add_argument('-ts', '--timeseries', dest='timeseriesFile',default='timeseries_ERA5_demErr.h5',help='MintPy timeseries file for bootstrapping')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    args = parser.parse_args(args=iargs)

    # if not (args.csvFile or args.gpsSite):
    if not (args.csvFile):

        parser.error('No CSV file of GPS station has been provided')
    elif not (args.distance):
        parser.error('No distance given around the GPS station(s)')
    else:
        return args


def geodesic_point_buffer(lat, lon, km):
    #Adapted from https://gis.stackexchange.com/a/289923

    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    proj_wgs84 = pyproj.Proj(init='epsg:4326')
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres

    return transform(project, buf).simplify(0).exterior.coords[:]

def generateMaskfromGPS(csvFile,distance,workdir):
    # Read CSV file with GPS stations and create mask files with given distance around them in kilometers.
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])
    lonList = list(df.iloc[:,1])
    latList = list(df.iloc[:,2])

    dist = distance

    for i in range(len(siteName)):
        siteDir = os.path.join(workdir,siteName[i])
        if not os.path.exists(siteDir):
            print('Creating directory: {0}'.format(siteDir))
            os.makedirs(siteDir)
        else:
            print('Directory {0} already exists.'.format(siteDir))
        maskCoord = geodesic_point_buffer(latList[i],lonList[i],dist)
        fileName = os.path.join(siteDir,siteName[i]+'.geojson')
        #Adapted from https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html#write-geometry-to-geojson
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for i in range(len(maskCoord)):
            ring.AddPoint(maskCoord[i][0],maskCoord[i][1])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        poly.FlattenTo2D()
        # Create the output Driver
        outDriver = ogr.GetDriverByName('GeoJSON')
        # Create the output GeoJSON
        outDataSource = outDriver.CreateDataSource(fileName)
        outLayer = outDataSource.CreateLayer(fileName, geom_type=ogr.wkbPolygon )
        # Get the output Layer's Feature Definition
        featureDefn = outLayer.GetLayerDefn()
        # create a new feature
        outFeature = ogr.Feature(featureDefn)
        # Set new geometry
        outFeature.SetGeometry(poly)
        # Add new feature to output Layer
        outLayer.CreateFeature(outFeature)
        # dereference the feature
        outFeature = None
        # Save and close DataSources
        outDataSource = None
        print('Created mask file: ',fileName)

    # return siteName

def ARIAdownload(csvFile,workdir):
    print('Download ARIA products using generated mask files')
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])
    prodDir = os.path.join(os.path.abspath(workdir),'products')

    for i in siteName:
        mask = os.path.abspath(os.path.join(workdir,i,i+'.geojson'))
        print('Running: ','ariaDownload.py', '-b', mask,'-w',prodDir)
        subprocess.run(['ariaDownload.py', '-b', mask,'-w',prodDir])
        print('Finished downloading data for: ',i)

    print('Creating directories with track numbers and moving products')
    fileList = glob.glob(os.path.join(prodDir+'/*.nc'))
    trackDirProdList = []
    for i in fileList:
        prod = i.split('/')[-1]
        trackNo = prod.split('-')[4]
        trackDir = os.path.abspath(os.path.join(prodDir,trackNo))
        trackDirProdList.append(trackDir)
        if not os.path.exists(trackDir):
            print('Creating directory: {0}'.format(trackDir))
            os.makedirs(trackDir)
            print('Moving product',i,'to',trackDir)
            shutil.move(i,trackDir)
            prodLoc = os.path.join(trackDir,prod)
            os.symlink(prodLoc,prodDir)
            # subprocess.run(['ln','-s',i, trackDir])
        else:
            print('Moving product',i,'to',trackDir)
            try:
                shutil.move(i,trackDir)
            except shutil.Error:
                print(i,'already in',trackDir)
            prodLoc = os.path.join(trackDir,prod)
            sym = os.path.join(prodDir,prodLoc)
            try:
                os.symlink(prodLoc,sym)
            except FileExistsError:
                print('Link to',i,'exists')
            # dst = os.path.join(trackDir,i)
            # try:
            #     os.symlink(prod,workdir+'/'+i)
            # except FileExistsError:
            #     print('Link to',i,'exists')
            # subprocess.run(['ln','-s',i, trackDir])
    return trackDirProdList


def ARIAtsSetup(csvFile,workdir,*args):
    print('Run ariaTSsetup to crop, extract and prepare stacks for each GPS station')
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])
    prodDir = os.path.join(os.path.abspath(workdir),'products')

    try:
        trackDirList

    except UnboundLocalError:
        trackDirList = []
        for x in range(len(list(os.walk(prodDir))[0][1])):
            trackDirList.append(os.path.join(os.path.abspath(prodDir),list(os.walk(prodDir))[0][1][x]))
        print('trackDirList: ',trackDirList)

    for i in siteName:
        for j in trackDirList:
            trackNo = str(j.split('/')[-1])
            siteDir = os.path.abspath(os.path.join(workdir,i,trackNo))
            print('Working in: ',siteDir)
            mask = os.path.abspath(os.path.join(workdir,i,i+'.geojson'))
            products = os.path.join(os.path.abspath(prodDir),trackNo,'*.nc')
            print('Running: ','ariaTSsetup.py','-f',"'{0}'".format(products),'--mask','download','--bbox',mask,'-w',siteDir)
            subprocess.run(['ariaTSsetup.py','-f',products,'--mask','download','--bbox',mask,'-w',siteDir])
            print('Finished time series setup for: ',i)
    return trackDirList

def prepAria(csvFile,templateFile,workdir):
    print('Run prep_aria.py to prepare for MintPy processing (reference point is chaged to GPS lat/lon)')
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])
    lonList = list(df.iloc[:,1])
    latList = list(df.iloc[:,2])

    for i in range(len(siteName)):
        siteLoc = os.path.abspath(os.path.join(workdir,siteName[i]))
        if 'DEM' not in list(os.walk(siteLoc))[0][1]:
            trackDirList = list(os.walk(siteLoc))[0][1]
        # else:
        #     trackDirList = siteLoc

        for x in trackDirList:
            try:
                siteDir = os.path.join(siteLoc,x)
                incDir = os.path.join(siteDir,'incidenceAngle/')
                incFile = glob.glob(incDir+'/*.vrt')[0]
                azDir = os.path.join(siteDir,'azimuthAngle/')
                azFile = glob.glob(azDir+'/*.vrt')[0]
            except IndexError:
                print('The station and surrounding area does not correspond to track or there is a problem with tsSetup:',x,'\n')
                continue

            mintpyDir = os.path.join(siteDir,'mintpy')
            stackDir = os.path.join(siteDir,'stack')
            demDir = os.path.join(siteDir,'DEM/SRTM_3arcsec.dem')
            incDir = os.path.join(siteDir,'incidenceAngle/')
            incFile = glob.glob(incDir+'/*.vrt')[0]
            azDir = os.path.join(siteDir,'azimuthAngle/')
            azFile = glob.glob(azDir+'/*.vrt')[0]
            maskDir = os.path.join(siteDir,'mask/watermask.msk')

            template = os.path.abspath(templateFile)
            ifgramFile = os.path.join(mintpyDir,'inputs/ifgramStack.h5')

            print('Running: ','prep_aria.py','-w',mintpyDir,'-s',stackDir,'-d',demDir,'-i',incFile,'-a',azFile,'--water-mask',maskDir)
            subprocess.run(['prep_aria.py','-w',mintpyDir,'-s',stackDir,'-d',demDir,'-i',incFile,'-a',azFile,'--water-mask',maskDir])
            print('Running: ','cp',template,mintpyDir)
            subprocess.run(['cp',template,mintpyDir])
            print('Template file', template, 'copied to:', mintpyDir)
            # print('Running: ','reference_point.py',ifgramFile,'--lat',latList[i],'--lon',lonList[i])
            # subprocess.run(['reference_point.py',ifgramFile,'--lat',str(latList[i]),'--lon',str(lonList[i])])

def timeseries(csvFile,templateFile,workdir):
    print('ACTIVATE YOUR MINTPY ENVIRONMENT BEFORE RUNNING THIS STEP')
    print('Run MintPy processing')
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])
    template = os.path.abspath(templateFile)

    for i in siteName:
        siteLoc = os.path.abspath(os.path.join(workdir,i))
        if 'DEM' not in list(os.walk(siteLoc))[0][1]:
            trackDirList = list(os.walk(siteLoc))[0][1]
        else:
            trackDirList = siteLoc

        for x in trackDirList:
            try:
                siteDir = os.path.join(siteLoc,x)
                mintpyDir = os.path.join(siteDir,'mintpy')
                subprocess.run(['smallbaselineApp.py',template],cwd=mintpyDir)
            except FileNotFoundError:
                print('No mintpy folder under:',siteDir)

def los2up(velFile,outName):
    from mintpy.utils import readfile, writefile

    VelInsar,atrVel = readfile.read(velFile,datasetName='velocity')
    StdInsar,atrStd = readfile.read(velFile,datasetName='velocityStd')

    incAng = np.float(atrVel['incidenceAngle'])*(np.pi/180)
    losU = VelInsar*(np.cos(incAng))

    dsDict = dict()
    dsDict['velocity'] = losU
    dsDict['velocityStd'] = StdInsar

    writefile.write(datasetDict=dsDict, out_file=outName, metadata=atrVel)

def bootStrap(csvFile,timeseriesFile,workdir):
    print('ACTIVATE YOUR MINTPY ENVIRONMENT BEFORE RUNNING THIS STEP')

    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])
    lonList = list(df.iloc[:,1])
    latList = list(df.iloc[:,2])

    for i in range(len(siteName)):
        print('**************************************************************')
        print('Working on:',siteName[i])
        siteLoc = os.path.abspath(os.path.join(workdir,siteName[i]))
        try:
            trackDirList = list(os.walk(siteLoc))[0][1]
        except:
            print('No track folder found under:',siteLoc)

        # if 'DEM' not in list(os.walk(siteLoc))[0][1]:
        #     trackDirList = list(os.walk(siteLoc))[0][1]
        # else:
        #     trackDirList = siteLoc

        for x in trackDirList:
            try:
                siteDir = os.path.join(siteLoc,x)
                # mintpyDir = os.path.join(siteDir,'mintpy')
                print('Change reference point to station:',siteName[i])
                subprocess.run(['reference_point.py',timeseriesFile,'--lat',str(latList[i]),'--lon',str(lonList[i])],cwd=siteDir)

                print('Run bootstrapping for:',siteName[i])
                subprocess.run(['bootStrap.py','-f',timeseriesFile,'-o',siteName[i]+'_'+x+'_bootVel.h5'],cwd=siteDir)

                print('Convert velocity to UP')
                los2up(siteDir+'/'+siteName[i]+'_'+x+'_bootVel.h5',siteDir+'/'+siteName[i]+'_'+x+'_UP_bootVel.h5')

                ##Mask new velocity file with spatial coherence
                print('Mask new velocity file with spatial coherence')
                subprocess.run(['generate_mask.py','avgSpatialCoh.h5','-m','0.7','--base','waterMask.h5','-o','maskSpatialCoh.h5'],cwd=siteDir)
                subprocess.run(['mask.py',siteDir+'/'+siteName[i]+'_'+x+'_UP_bootVel.h5','-m','maskSpatialCoh.h5'],cwd=siteDir)

            except:
                print('No mintpy folder under:',siteDir)

def mergeGPS(csvFile,workdir,GPSdataDir='./GPS'):
    from mintpy.objects.gps import GPS
    from mintpy.utils import readfile, writefile

    velList = sorted(glob.glob(workdir+'/*/*/*_*_UP_bootVel_msk.h5'))
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])
    for i in range(len(siteName)):
        print('**************************************************************')
        print('Station:',siteName[i])
        startDate = readfile.read_attribute(velList[i])['START_DATE']
        try:
            gps_obj = GPS(site=siteName[i], data_dir=GPSdataDir)
            gps_obj.open()
        except:
            print('Station',siteName[i],'cannot be downloaded')
            continue

        print('Reading up displacement of values')
        gps_obj.read_displacement(start_date=startDate)
        dis_u = gps_obj.dis_u
        dates = gps_obj.dates
        dateList = []
        for k in dates:
            dateList.append((float(k.strftime("%j"))-1) / 366 + float(k.strftime("%Y")))

        if len(dateList) < 100:
            print('There are less than 100 displacement values in GPS timeseries')
        elif dateList[-1] - dateList[0] < 3.:
            print('Timeseries length is shorter than 3 years:',str(dateList[-1] - dateList[0]))
        else:
            # Solve for mx+c, coef[0][0] = m, coef[0][1] = c
            coef = np.polyfit(dateList,dis_u,1,rcond=None,full=True)
            velGPS = coef[0][0]
            # Sum of squared residuals
            StdGPS = np.sqrt(coef[1]/len(dateList))
            print('GPS up velocity for',siteName[i],': ',velGPS,StdGPS)
            velInsar,atr = readfile.read(velList[i],datasetName='velocity')
            StdInsar,atrStd = readfile.read(velList[i],datasetName='velocityStd')
            jointVel = np.where(velInsar==0, velInsar,velInsar+velGPS)
            # jointVel = velInsar + velGPS
            jointStd = np.where(velInsar==0, velInsar,np.sqrt(StdInsar**2+StdGPS**2))
            print('Joint Velocity and Standard deviation calculated')
            print('Joint Vel for',siteName[i],': ',jointVel.max(),jointStd.max())

            dsDict = dict()
            dsDict['velocity'] = jointVel
            dsDict['velocityStd'] = jointStd
            outdirList = velList[i].split('/')[0:-1]
            outdir = os.path.join(*outdirList)
            velName = velList[i].split('/')[-1]
            index = velName.find('bootVel_msk.h5')
            outFileName = os.path.abspath(os.path.join(outdir,velName[:index]+'GPSadded_'+velName[index:]))
            writefile.write(datasetDict=dsDict, out_file=outFileName, metadata=atr)

def copyTS(csvFile,workdir):
    # print('ACTIVATE YOUR MINTPY ENVIRONMENT BEFORE RUNNING THIS STEP')
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])

    for i in range(len(siteName)):
        siteDir = os.path.join(workdir,'COPY',siteName[i])
        if not os.path.exists(siteDir):
            print('Creating directory: {0}'.format(siteDir))
            os.makedirs(siteDir)
        else:
            print('Directory {0} already exists.'.format(siteDir))

    for i in siteName:
        print('Copying data from:',i)
        siteLoc = os.path.abspath(os.path.join(workdir,i))
        trackDirList = list(os.walk(siteLoc))[0][1]

        # if 'DEM' not in list(os.walk(siteLoc))[0][1]:
        #     trackDirList = list(os.walk(siteLoc))[0][1]
        # else:
        #     trackDirList = siteLoc

        for x in trackDirList:
            try:
                siteDir = os.path.join(siteLoc,x)
                mintpyDir = os.path.join(siteDir,'mintpy')
                copyDir = os.path.abspath(os.path.join(workdir,'COPY',i,x,))
                if not os.path.exists(copyDir):
                    print('Creating directory: {0}'.format(copyDir))
                    os.makedirs(copyDir)
                    # print('cp', mintpyDir+'/*.h5', copyDir)
                    for file in glob.glob(mintpyDir+'/*.h5'):
                        shutil.copy(file, copyDir)
                else:
                    print('Directory {0} already exists.'.format(copyDir))
            except FileNotFoundError:
                print('No mintpy folder under:',siteDir)

def main(inps=None):
    inps = cmdLineParse()
    # masksDir = os.path.join(inps.workdir,'masks')

    if inps.step == 'generateMask':
        generateMaskfromGPS(inps.csvFile,inps.distance,inps.workdir)
    elif inps.step == 'download':
        ARIAdownload(inps.csvFile,inps.workdir)
    elif inps.step == 'tsSetup':
        ARIAtsSetup(inps.csvFile,inps.workdir)
    elif inps.step == 'prepAria':
        prepAria(inps.csvFile,inps.template,inps.workdir)
    elif inps.step == 'timeseries':
        timeseries(inps.csvFile,inps.template,inps.workdir)
    elif inps.step == 'bootStrap':
        bootStrap(inps.csvFile,inps.timeseriesFile,inps.workdir)
    elif inps.step == 'mergeGPS':
        mergeGPS(inps.csvFile,inps.workdir)
    elif inps.step == 'copyTS':
        copyTS(inps.csvFile,inps.workdir)
    elif inps.step == 'skipdownload':
        generateMaskfromGPS(inps.csvFile,inps.distance,inps.workdir)
        ARIAtsSetup(inps.csvFile,inps.workdir)
        prepAria(inps.csvFile,inps.template,inps.workdir)
    else:
        generateMaskfromGPS(inps.csvFile,inps.distance,inps.workdir)
        trackDirProdList = ARIAdownload(inps.csvFile,inps.workdir)
        ARIAtsSetup(inps.csvFile,inps.workdir,trackDir=trackDirProdList)
        prepAria(inps.csvFile,inps.template,inps.workdir)
        # timeseries(inps.csvFile,inps.template,inps.workdir)

###############################################################################
if __name__ == '__main__':
    inps = cmdLineParse()
    main(inps)
