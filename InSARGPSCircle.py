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
import subprocess
import os, sys
import shutil
import timeit
import pyproj
import glob

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
    parser.add_argument('-c', '--csv', dest='csvFile', help='CSV file listing GPS stations (SiteID, Lon, Lat)')
    parser.add_argument('-tr', '--track', dest='trackDir', help='Products folder with the relevant track number')
    # parser.add_argument('-g', '--gps', dest='gpsSite', help='GPS stations given in the order as: SiteID, Lon, Lat')
    parser.add_argument('-d', '--dist', dest='distance', type=int, help='Distance around GPS station for mask generation (in km)')
    parser.add_argument('-s', '--step', dest='step', default='all', help='Choose step to do (all steps are run from scracth if no step is given) [generateMaskdownload,tsSetup,prepAria,timeseries,skipdownload]')
    parser.add_argument('-t', '--temp', dest='template', default='smallbaselineApp.cfg',help='MintPy template file to used for time series processing')
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
        trackNo = i.split('-')[4]
        trackDir = os.path.abspath(os.path.join(prodDir,trackNo))
        trackDirProdList.append(trackDir)
        if not os.path.exists(trackDir):
            print('Creating directory: {0}'.format(trackDir))
            os.makedirs(trackDir)
            print('Moving product',i,'to',trackDir)
            shutil.move(i,trackDir)
            prod = os.path.join(trackDir,i.split('/')[-1])
            os.symlink(prod,prodDir)
            # subprocess.run(['ln','-s',i, trackDir])
        else:
            print('Moving product',i,'to',trackDir)
            try:
                shutil.move(i,trackDir)
            except shutil.Error:
                print(i,'already in',trackDir)
            prod = os.path.join(trackDir,i.split('/')[-1])
            sym = os.path.join(prodDir,i.split('/')[-1])
            try:
                os.symlink(prod,sym)
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
        if not 'DEM' in list(os.walk(siteLoc))[0][1]:
            trackDirList = list(os.walk(siteLoc))[0][1]
        else:
            trackDirList = siteLoc

        for x in trackDirList:
            siteDir = os.path.join(siteLoc,x)

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

    for i in range(len(siteName)):
        siteLoc = os.path.abspath(os.path.join(workdir,siteName[i]))
        if not 'DEM' in list(os.walk(siteLoc))[0][1]:
            trackDirList = list(os.walk(siteLoc))[0][1]
        else:
            trackDirList = siteLoc

        for x in trackDirList:
            siteDir = os.path.join(siteLoc,x)



    for i in siteName:
        siteLoc = os.path.abspath(os.path.join(workdir,i))

        if not 'DEM' in list(os.walk(siteLoc))[0][1]:
            trackDirList = list(os.walk(siteLoc))[0][1]
        else:
            trackDirList = siteLoc

        for x in trackDirList:
            siteDir = os.path.join(siteLoc,x)
            mintpyDir = os.path.join(siteDir,'mintpy')
            subprocess.run(['smallbaselineApp.py',template],cwd=mintpyDir)

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
