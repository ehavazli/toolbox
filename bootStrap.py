#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Emre Havazli
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np
from sklearn.utils import resample
from mintpy.utils import readfile, writefile


def createParser():
    '''
        Bootstrapping method to estimate velocity and uncertainty
    '''

    import argparse
    parser = argparse.ArgumentParser(description='Bootstrap method for velocity and uncertainty estimation')
    parser.add_argument('-f', '--file', dest='timeseriesFile', type=str, required=True, help='Timeseries File')
    parser.add_argument('-w', '--workdir', dest='workdir', type=str, default='./', help='Specify directory to deposit all outputs. Default is local directory where script is launched.')
    parser.add_argument('-ns', '--nsamples', dest='sampleNo',type=int, default=10, help='Number of samples in the subset (default: 10)')
    parser.add_argument('-nb', '--nboot', dest='bootCount', type=int, default=400, help='Number of bootstrap runs (default: 400)')
    parser.add_argument('-o', '--output', dest='outfile', type=str, default='bootVel.h5', help='Name of output file (default: bootVel.h5)')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def main(iargs=None):
    inps = cmdLineParse(iargs)

    ts_data, atr = readfile.read(inps.timeseriesFile)
    tsData = readfile.timeseries(inps.timeseriesFile)
    if atr['UNIT'] == 'mm':
        ts_data *= 1./1000.

    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    dateList = tsData.get_date_list()

    vel = np.zeros((inps.bootCount,length,width))

    for i in range(inps.bootCount):
        # print('Running boot number: ',i+1)
        bootSamples = list(np.sort(resample(dateList, replace=True, n_samples=inps.sampleNo)))
        # dropList = [x for x in dateList if x not in bootSamples]

        # from ARIAtools import progBar
        # prog_bar = progBar.progressBar(maxValue=inps.bootCount,prefix='Running boot number: '+i)
        # prog_bar.update(i+1)

        bootList = []
        for k in bootSamples:
            bootList.append(dateList.index(k))
        numDate = len(bootList)
        ts_data_sub = ts_data[bootList, :, :].reshape(numDate, -1)

        A = tsData.get_design_matrix4average_velocity(bootSamples)
        X = np.dot(np.linalg.pinv(A), ts_data_sub)
        vel[i] = np.array(X[0, :].reshape(length, width), dtype='float32')

    # prog_bar.close()
    print('Finished calculating resampling and velocity calculation')
    velMean = vel.reshape(inps.bootCount,(length*width)).mean(axis=0)
    velStd = vel.reshape(inps.bootCount,(length*width)).std(axis=0)
    velMean = velMean.reshape(length,width)
    velStd = velStd.reshape(length,width)
    print('Calculated mean velocity and standard deviation')

    atr['FILE_TYPE'] = 'velocity'
    atr['UNIT'] = 'm/year'
    atr['START_DATE'] = bootSamples[0]
    atr['END_DATE'] = bootSamples[-1]
    atr['DATE12'] = '{}_{}'.format(bootSamples[0], bootSamples[-1])

    # write to HDF5 file
    dsDict = dict()
    dsDict['velocity'] = velMean
    dsDict['velocityStd'] = velStd
    if not os.path.exists(inps.workdir):
        print('Creating directory: {0}'.format(os.path.join(inps.workdir)))
        os.makedirs(inps.workdir)
    else:
        print('Directory {0} already exists.'.format(inps.workdir))
    outName = os.path.join(inps.workdir,inps.outfile)
    writefile.write(dsDict, out_file=outName, metadata=atr)

############################################################################
if __name__ == '__main__':
    main()
