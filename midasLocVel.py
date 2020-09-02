#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Emre Havazli
# Copyright 2020, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import argparse
import os, sys, time

def create_parser():
    '''
        Generate a csv file from UNR MIDAS velocities and locations
        [SiteID, Lon, Lat, Vele, Veln, Velu, Sigmae, Sigman, Sigmau]
    '''

    parser = argparse.ArgumentParser(description='Generate a csv file from UNR MIDAS velocities and locations')
    parser.add_argument('-loc', '--locfile', dest='locationFile',default='http://geodesy.unr.edu/NGLStationPages/llh.out',
                                type=str, help='Web link to the file with lat/lon info, default: http://geodesy.unr.edu/NGLStationPages/llh.out')
    parser.add_argument('-vel', '--velfile', dest='velocityFile',default='http://geodesy.unr.edu/velocities/midas.IGS14.txt',
                                type=str, help='Web link to the file with MIDAS velocities, default: http://geodesy.unr.edu/velocities/midas.IGS14.txt')
    parser.add_argument('-o', '--outfile', dest='outFile',default='MIDASVelLoc.csv', type=str, help='Output file name, default: MIDASVelLoc.csv')

    return parser

def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    return parser.parse_args(args=iargs)
############################################################################
## Progress bar class ##
############################################################################

class progressBar:
    """Creates a text-based progress bar. Call the object with
    the simple print command to see the progress bar, which looks
    something like this:
    [=======> 22%       ]
    You may specify the progress bar's min and max values on init.
    note:
        modified from MintPy version 1.2 (https://github.com/insarlab/MintPy/)
        Code originally from http://code.activestate.com/recipes/168639/
    example:
        from ARIAtools import progBar
        prog_bar = progBar.progressBar(maxValue=len(product_dict[0]),prefix='Generating: '+key+' - ')
        for i in enumerate(product_dict[0]):
            prog_bar.update(i[0]+1,suffix=product_dict[1][i[0]][0])
        prog_bar.close()
    """

    def __init__(self, maxValue=100, prefix='', minValue=0, totalWidth=70, print_msg=True):
        self.prog_bar = "[]"  # This holds the progress bar string
        self.min = minValue
        self.max = maxValue
        self.span = maxValue - minValue
        self.suffix = ''
        self.prefix = prefix

        self.print_msg = print_msg
        self.width = totalWidth
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.amount = 0  # When amount == max, we are 100% done
        self.update_amount(0)  # Build progress bar string

    def update_amount(self, newAmount=0, suffix=''):
        """ Update the progress bar with the new amount (with min and max
        values set at initialization; if it is over or under, it takes the
        min or max value as a default. """
        if newAmount < self.min:
            newAmount = self.min
        if newAmount > self.max:
            newAmount = self.max
        self.amount = newAmount

        # Figure out the new percent done, round to an integer
        diffFromMin = np.float(self.amount - self.min)
        percentDone = (diffFromMin / np.float(self.span)) * 100.0
        percentDone = np.int(np.round(percentDone))

        # Figure out how many hash bars the percentage should be
        allFull = self.width - 2 - 18
        numHashes = (percentDone / 100.0) * allFull
        numHashes = np.int(np.round(numHashes))

        # Build a progress bar with an arrow of equal signs; special cases for
        # empty and full
        if numHashes == 0:
            self.prog_bar = '%s[>%s]' % (self.prefix, ' '*(allFull-1))
        elif numHashes == allFull:
            self.prog_bar = '%s[%s]' % (self.prefix, '='*allFull)
            if suffix:
                self.prog_bar += ' %s' % (suffix)
        else:
            self.prog_bar = '[%s>%s]' % ('='*(numHashes-1), ' '*(allFull-numHashes))
            # figure out where to put the percentage, roughly centered
            percentPlace = int(len(self.prog_bar)/2 - len(str(percentDone)))
            percentString = ' ' + str(percentDone) + '% '
            # slice the percentage into the bar
            self.prog_bar = ''.join([self.prog_bar[0:percentPlace],
                                     percentString,
                                     self.prog_bar[percentPlace+len(percentString):]])
            # prefix and suffix
            self.prog_bar = self.prefix + self.prog_bar
            if suffix:
                self.prog_bar += ' %s' % (suffix)
            # time info - elapsed time and estimated remaining time
            if percentDone > 0:
                elapsed_time = time.time() - self.start_time
                self.prog_bar += '%5ds / %5ds' % (int(elapsed_time),
                                                  int(elapsed_time * (100./percentDone-1)))

    def update(self, value, every=1, suffix=''):
        """ Updates the amount, and writes to stdout. Prints a
         carriage return first, so it will overwrite the current
          line in stdout."""
        if value % every == 0 or value >= self.max:
            self.update_amount(newAmount=value, suffix=suffix)
            if self.print_msg:
                sys.stdout.write('\r' + self.prog_bar)
                sys.stdout.flush()

    def close(self):
        """Prints a blank space at the end to ensure proper printing
        of future statements."""
        if self.print_msg:
            print(' ')
############################################################################
## Progress bar class ##
############################################################################

def mergefiles(locationFile,velocityFile):
    locations = pd.read_csv(locationFile,header=0,names=['SiteID','Lat','Lon','Height'],delim_whitespace=True)
    VelUnc = pd.read_csv(velocityFile,header=0,usecols=[0,8,9,10,11,12,13],names=['SiteID','east','north','up','sigmae','sigman','sigmau'],delim_whitespace=True)
    data = []
    prog_bar = progressBar(maxValue=len(VelUnc['SiteID']))
    n = 0
    for i in VelUnc['SiteID']:
        prog_bar.update(n+1,suffix=i)
        try:
            lon = (locations[locations['SiteID'] == i]['Lon'].values[0])
            lat = (locations[locations['SiteID'] == i]['Lat'].values[0])
            vele = (VelUnc[VelUnc['SiteID']== i]['east'].values[0])
            veln = (VelUnc[VelUnc['SiteID']== i]['north'].values[0])
            velu = (VelUnc[VelUnc['SiteID']== i]['up'].values[0])
            sige = (VelUnc[VelUnc['SiteID']== i]['sigmae'].values[0])
            sign = (VelUnc[VelUnc['SiteID']== i]['sigman'].values[0])
            sigu = (VelUnc[VelUnc['SiteID']== i]['sigmau'].values[0])
        except IndexError:
            continue
        data.append({'SiteID': i, 'Lon': lon, 'Lat': lat, 'Vele': vele,'Veln': veln, 'Velu': velu, 'sige': sige, 'sign': sign, 'sigu': sigu})
    prog_bar.close()
    df = pd.DataFrame(data)
    return df
############################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    df = mergefiles(inps.locationFile,inps.velocityFile)
    df.to_csv(inps.outFile,index=False,sep=',')
    print('Check for file',os.path.abspath(inps.outFile))
    return inps.outFile


############################################################################
if __name__ == '__main__':
    main()
