#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Emre Havazli
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import h5py
import numpy as np


def createParser():
    '''
        Convert velocity or timeseries file (MintPy output) from LOS to Up component
    '''

    import argparse
    parser = argparse.ArgumentParser(description='LOS to UP')
    parser.add_argument('-g', '--geoFile', dest='geoFile', type=str, default='./inputs/geometryGeo.h5', help='Path and name of the geomertyGeo.h5 file (default: ./inputs/geomertyGeo.h5)')
    parser.add_argument('-v', '--velocity', dest='velFile', type=str, default='./velocity.h5', help='Path and name of the velocity file (default: ./velocity.h5)')
    # parser.add_argument('-t', '--timeseries', dest='tsFile', type=str, default='./timeseries.h5', help='Path and name of the timeseries file (default: ./timeseries.h5)')
    parser.add_argument('-o', '--outname', dest='outName', type=str, default='./velocity_up.h5', help='Path and name of the output file (default: ./velocity_up.h5)')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def main(inps=None):
    inps = cmdLineParse()

    h5Geo = h5py.File(inps.geoFile,'r')
    h5Vel = h5py.File(inps.velFile,'r')
    # h5ts = h5py.File(inps.tsFile,'r')
    h5Out = h5py.File(inps.outName, 'w')

    Iset = h5Geo.get('incidenceAngle').value
    Inc = Iset**(np.pi/180.)
    attrs = h5Vel.attrs
    # attrs = h5ts.attrs

    Vset = h5Vel.get('velocity')
    # Vset = h5ts.get('timeseries')
    # bperp = h5ts.get('bperp')
    # dates = h5ts.get('date')

    losU = Vset/(np.cos(Inc))

    # dset = h5Out.create_dataset('bperp', data=bperp)
    # dset = h5Out.create_dataset('date',data=dates)
    dset = h5Out.create_dataset('velocity', data=losU)


    for key, value in h5Vel.attrs.items():
        dset.attrs[key] = value
    # for key, value in h5ts.attrs.items():
    #     dset.attrs[key] = value
    h5Out.close()
    h5Geo.close()
    h5Vel.close()
    # h5ts.close()

############################################################################
if __name__ == '__main__':
    main()
