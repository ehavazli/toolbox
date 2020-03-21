#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Emre Havazli, David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os, sys
import pandas as pd
import numpy as np
import glob
import gdal
from scipy.io import netcdf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mintpy.utils import readfile,writefile, utils as ut
from mintpy.utils import plot as pp

def createParser():
    '''
        Workflow to generate weighted averages of bootstrap outputs.
    '''

    import argparse
    parser = argparse.ArgumentParser(description='Generate weighted average maps')
    # parser.add_argument('-f', '--file', dest='imgfile', type=str, required=True, help='ARIA file')
    parser.add_argument('-w', '--workdir', dest='workdir', default='./output', help='Specify directory to deposit all outputs. Default is "./output"')
    parser.add_argument('-i', '--inputdir', dest='inputdir', default='./', type=str, help='Specify directory that includes station output folders. Default is "./"')
    parser.add_argument('-f', '--filewld', dest='filewld', default='*_msk.h5', help='Specify directory that includes station output folders. Default is "./"')
    parser.add_argument('-c', '--csv', dest='csvFile', help='CSV file listing GPS stations (SiteID, Lon, Lat) can be obtained from "http://plugandplay.unavco.org:8080/unrgsac/gsacapi/site/form" but small editing is required')
    # parser.add_argument('-s', '--step', dest='step', default='all', help='Choose step to do (all steps are run from scracth if no step is given) [generateMask,download,tsSetup,prepAria,timeseries,bootStrap,skipdownload]')
    parser.add_argument('-um', '--uncmask', dest='uncmask', type=np.float32,default='0',help='Uncertainty threshold in m/yr')
    # parser.add_argument('-ts', '--timeseries', dest='timeseriesFile',default='timeseries_ERA5_demErr.h5',help='MintPy timeseries file for bootstrapping')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    args = parser.parse_args(args=iargs)

    # if not (args.csvFile or args.gpsSite):
    if not (args.csvFile):
        parser.error('No CSV file of GPS station has been provided')
    # elif not (args.distance):
    #     parser.error('No distance given around the GPS station(s)')
    else:
        return args
    # return args

def merge_tracks_weighted(obs,err,uncmask):
    """Merge in the first dimension through averaging.
    Inputs:
        obs Observations, dimension=(k,m,n)
        err Observation error, dimension=(k,m,n)
    Outputs:
        obs_out    Estimated merged obs
        err_out    Estimated uncertainty merged obs

    Will output the data as arrays with dimension=(m,n) dtype=float32.
    """
    obsFile = gdal.Open(obs)
    obsArray = obsFile.ReadAsArray()
    obsArray = np.nan_to_num(obsArray,copy=True)
    print('Number of Velocity Files:',obsArray.shape[0])
    errFile = gdal.Open(err)
    errArray = errFile.ReadAsArray()
    errArray = np.nan_to_num(errArray,copy=True)
    print('Number of Uncertainty Files:',errArray.shape[0])

    ### generate a mask file of no data
    if uncmask == 0:
        obs_masked = np.ma.masked_where(obsArray==0., obsArray)
        err_masked = np.ma.masked_where(obsArray==0., errArray)
    else:
        obs_masked1 = np.ma.masked_where(obsArray==0., obsArray)
        err_masked1 = np.ma.masked_where(obsArray==0., errArray)
        obs_masked = np.ma.masked_where(errArray>=uncmask, obs_masked1)
        err_masked = np.ma.masked_where(errArray>=uncmask, err_masked1)

    ### compute the weights w from the uncertanties: as w=1/sigma^2
    weights=np.divide(1,np.square(err_masked))

    ### use of numpy masked average function
    obs_out, weights_out = np.ma.average(obs_masked,axis=0,returned=True,weights=weights)

    ### return uncertanties sigma: as w=1/sigma^2
    err_out =np.sqrt(np.divide(1, weights_out))
    del weights_out

    ### returnign the data
    return obs_out, err_out

def write_gmt_simple(lons, lats, z, fname, title='default', name='z', scale=1.0, offset=0, units='meters'):
    """Writes a simple GMT grd file with one array.
    This is based on the gdal2grd.py script found at:
        http://www.vso.cape.com/~nhv/files/python/gdal/gdal2grd.py
    Parameters: lons : 1D Array of lon values
                lats : 1D Array of lat values
                z : 2D slice to be saved
                fname : Output file name
    Kwargs:     title : Title for the grd file
                name : Name of the field in the grd file
                scale : Scale value in the grd file
                offset : Offset value in the grd file
    Returns:    fname
    """

    fid = netcdf.netcdf_file(fname, 'w')

    # Create a dimension variable
    fid.createDimension('side', 2)
    fid.createDimension('xysize', np.prod(z.shape))

    # Range variables
    fid.createVariable('x_range', 'd', ('side',))
    fid.variables['x_range'].units = 'degrees'

    fid.createVariable('y_range', 'd', ('side',))
    fid.variables['y_range'].units = 'degrees'

    fid.createVariable('z_range', 'd', ('side',))
    fid.variables['z_range'].units = units

    # Spacing
    fid.createVariable('spacing', 'd', ('side',))
    fid.createVariable('dimension', 'i4', ('side',))

    fid.createVariable('z', 'f', ('xysize',))
    fid.variables['z'].long_name = name
    fid.variables['z'].scale_factor = scale
    fid.variables['z'].add_offset = offset
    fid.variables['z'].node_offset = 0

    fid.title = title
    fid.source = 'MintPy'

    # Filling in the actual data
    fid.variables['x_range'][0] = lons[0]
    fid.variables['x_range'][1] = lons[-1]
    fid.variables['spacing'][0] = lons[1]-lons[0]

    fid.variables['y_range'][0] = lats[0]
    fid.variables['y_range'][1] = lats[-1]
    fid.variables['spacing'][1] = lats[1]-lats[0]

    # Range
    fid.variables['z_range'][0] = np.nanmin(z)
    fid.variables['z_range'][1] = np.nanmax(z)

    fid.variables['dimension'][:] = z.shape[::-1]
    fid.variables['z'][:] = np.flipud(z).flatten()
    fid.close()
    return fname
    ############################################################
    # Program is part of GIAnT v1.0                            #
    # Copyright 2012, by the California Institute of Technology#
    # Contact: earthdef@gps.caltech.edu                        #
    ############################################################


def get_geo_lat_lon(atr):
    X_FIRST = float(atr['X_FIRST'])
    Y_FIRST = float(atr['Y_FIRST'])
    X_STEP = float(atr['X_STEP'])
    Y_STEP = float(atr['Y_STEP'])
    W = int(atr['WIDTH'])
    L = int(atr['LENGTH'])
    Y_END = Y_FIRST + L*Y_STEP
    X_END = X_FIRST + W*X_STEP

    X = np.linspace(X_FIRST, X_END, W)
    Y = np.linspace(Y_FIRST, Y_END, L)
    #XI,YI = np.meshgrid(X,Y)

    return Y, X


def write_grd_file(data, atr, fname_out=None):
    """Write GMT .grd file for input data matrix, using giant._gmt module.
    Inputs:
        data - 2D np.array in int/float, data matrix to write
        atr  - dict, attributes of input data matrix
        fname_out - string, output file name
    Output:
        fname_out - string, output file name
    """
    # Get 1D array of lats and lons
    lats, lons = get_geo_lat_lon(atr)

    # writing
    print('writing >>> '+fname_out)
    write_gmt_simple(lons, np.flipud(lats), np.flipud(data), fname_out,
                     title='default', name=atr['FILE_TYPE'],
                     scale=1.0, offset=0, units=atr['UNIT'])
    return fname_out

def mintpy2grd(csvFile,inputdir,workdir,filewld):
    df = pd.read_csv(csvFile)
    siteName = list(df.iloc[:,0])

    fileList = sorted(glob.glob(os.path.join(inputdir,filewld)))
    outVelList = []
    outStdList = []
    for i in fileList:
        # 1. Read data
        dataVel, atrVel = readfile.read(i, datasetName='velocity')
        dataStd, atrStd = readfile.read(i, datasetName='velocityStd')

        # 2. Check the reference point
       # refLon = np.float32(readfile.read_attribute(i)['REF_LON']).round(2)
       # refLat = np.float32(readfile.read_attribute(i)['REF_LAT']).round(2)
       # siteName = i.split('/')[-1].split('_')[0]
       # stationLon = np.float32(df[df['SiteID'].str.contains(siteName)]['Lon']).round(2)
       # stationLat = np.float32(df[df['SiteID'].str.contains(siteName)]['Lat']).round(2)
       # if np.isclose(refLon,stationLon):
       #     # print('Lon is close enough',refLon,stationLon)
       #     if np.isclose(refLat,stationLat):
       #         # print('Lat is close enough',refLat,stationLat)
        # 3. Write GMT .grd file
        outbaseVel = os.path.abspath(os.path.join(inps.workdir,'{}.grd'.format(pp.auto_figure_title(i, datasetNames='velocity',inps_dict=vars(inps)))))
        outbaseStd = os.path.abspath(os.path.join(inps.workdir,'{}Std.grd'.format(pp.auto_figure_title(i, datasetNames='velocityStd',inps_dict=vars(inps)))))
        outVelList.append(outbaseVel)
        outStdList.append(outbaseStd)

        outfileVel = write_grd_file(dataVel, atrVel, outbaseVel)
        outfileStd = write_grd_file(dataStd, atrStd, outbaseStd)
        print('Done.')
        #else:
        #    print('Reference point is not on station',siteName,refLon,stationLon)
    return outVelList, outStdList


def main(inps=None):
    inps = cmdLineParse()

    if not os.path.exists(inps.workdir):
        print('Creating directory: {0}'.format(os.path.join(inps.workdir)))
        os.makedirs(inps.workdir)
    else:
        print('Directory {0} already exists.'.format(inps.workdir))

# SAVE h5 files in GRD format
    outVelList, outStdList = mintpy2grd(inps.csvFile,inps.inputdir,inps.workdir,inps.filewld)

# Build VRT by writing each velocity file to a separate band
    outVRTnameVel = os.path.join(inps.workdir,'VelRaster.vrt')
    outVRTnameStd = os.path.join(inps.workdir,'StdRaster.vrt')
    vrt_options = gdal.BuildVRTOptions(separate=True, srcNodata=0)
    dsVel = gdal.BuildVRT(outVRTnameVel, outVelList, options=vrt_options)
    dsVel=None
    dsStd = gdal.BuildVRT(outVRTnameStd, outStdList, options=vrt_options)
    dsStd=None

# Calculate weighted averages over the overlapping regions
    obs_out, err_out = merge_tracks_weighted(outVRTnameVel,outVRTnameStd,inps.uncmask)

# Save gdal and GMT compatible GRD file
    srcDS = gdal.Open(outVRTnameVel)
    transform = srcDS.GetGeoTransform()
    xsize = srcDS.RasterXSize
    ysize = srcDS.RasterYSize
    srcDS = None
    # Velocity
    dstDS = gdal.GetDriverByName('GSBG')
    dataset = dstDS.Create(os.path.join(inps.workdir,'WeightedAvVel.grd'),xsize,ysize)
    dataset.SetGeoTransform(transform)
    q = dataset.GetRasterBand(1)
    q.WriteArray(obs_out)
    q.SetNoDataValue(0)
    q.FlushCache()
    dataset.FlushCache()
    dstDS = None
    # Standard deviation
    dstDS = gdal.GetDriverByName('GSBG')
    dataset = dstDS.Create(os.path.join(inps.workdir,'WeightedAvStd.grd'),xsize,ysize)
    dataset.SetGeoTransform(transform)
    q = dataset.GetRasterBand(1)
    q.WriteArray(err_out)
    q.SetNoDataValue(0)
    q.FlushCache()
    dataset.FlushCache()
    dstDS = None
#
    fig, ax = plt.subplots()
    cbar = plt.colorbar(ax.imshow(obs_out*1000,cmap='jet',vmin=-10,vmax=10))
    # cbar = plt.colorbar(ax.imshow(obs_out,cmap='jet'))
    cbar.set_label('mm/yr')
    plt.savefig(os.path.join(inps.workdir,'AvgVel.png'),format='png',dpi=300,quality=95)
    plt.close()
#
    fig, ax = plt.subplots()
    cbar = plt.colorbar(ax.imshow(err_out*1000,cmap='jet',vmin=-10,vmax=10))
    # cbar = plt.colorbar(ax.imshow(err_out,cmap='jet'))
    cbar.set_label('mm/yr')
    plt.savefig(os.path.join(inps.workdir,'AvgStd.png'),format='png',dpi=300,quality=95)
    plt.close()
###############################################################################
if __name__ == '__main__':
    inps = cmdLineParse()
    main(inps)
