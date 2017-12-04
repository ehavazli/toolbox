#!/usr/bin/env python

# Retrieves MODIS data from OSCAR server.
#
# Lei Pan,     2012-06-06:  original client code (oscarpython_client.py)
# Sang-Ho Yun, 2012-09-24:  modified to take command line input with options and create
#                           a netcdf grd file that can be read with GMT or matlab (grdread2.m).
#
# Copyright: (c) 2011
#            Jet Propulsion Laboratory, California Institute of Technology.
#            ALL RIGHTS RESERVED. U.S. Government Sponsorship acknowledged.

import os
import sys
import urllib
import urllib2
import datetime
import time
import getopt

out = sys.stdout

def usage():
  print ''
  print 'usage: get_modis.py -r minlon/maxlon/minlat/maxlat -t yyyy-mm-ddTHH:MM:SS[Z] [-p platform] [-o outfile] [-w timewindow] [-g gridsize] [-l] [-f] [-v] [-s server]'
  print ''
  print '   -r (--region): lon in [-180,180], lat in [-90,90], mandatory'
  print '   -t (--time): time and date in ISO format, mandatory'
  print '   -p (--platform): terra (default), aqua, or any.'
  print '   -o (--outfile): output filename (default: mod_minlon_maxlon_minlat_maxlat_dateTtime.grd)'
  print '   -w (--timewindow): saerch time window size in second (default: 18000, i.e. time +/-5hrs)'
  print '   -g (--gridsize): grid spacing in degrees (default: 30./3600.=0.008333333...)'
  print '   -l (--localtime): if used, the given time is local time, otherwise UTC'
  print '   -f (--figurefile): downloads the png figure file (same fileroot but with .png)'
  print '   -v (--verbose)'
  print '   -s (--server): specify which server to use - oscar1 (default) or oscar2'
  print ''
  print '   Example:'
  print '   get_modis.py -r 0/10/0/10 -t 2009-04-01T10:30:20 -p terra -s oscar2'
  print '   get_modis.py -r -180/-170/-5/10 -t 2010-10-07T08:10:00 -o q.grd -l -f -v'
  print ''
  sys.exit()

def main():
  try:
    options, remainder = getopt.getopt(sys.argv[1:],'r:t:p:o:w:g:lfvs:',['region=',
                                                                       'time=',
                                                                       'platform=',
                                                                       'outfile=',
                                                                       'timewindow=',
                                                                       'gridsize=',
                                                                       'localtime',
                                                                       'figurefile',
                                                                       'verbose',
                                                                       'server'])
  except getopt.GetoptError, err:
    print str(err)
    usage()
    sys.exit(2)

  n = len(sys.argv)
  if n == 1:
    usage()
    sys.exit()

  # Default values
  platform = 'terra'  # terra data seem better than aqua data.
                      # Use 'any' option with caution. It can be misleading,
                      # because SAMI would merge data from Terra and Aqua.
  timeWindow = 18000  # 5 hours on each side
  grid_spacing = 30./3600.  # 30 arcseconds (~1km) in degrees
  localTime = False
  figFile = False
  verbose = False
  server = 'oscar1'

  for opt, arg in options:
    if opt in ('-r','--region'):
      regionS = arg
    elif opt in ('-t','--time'):
      timeS = arg
    elif opt in ('-p','--platform'):
      platform = arg
    elif opt in ('-o','--outfile'):
      grdFname = arg
    elif opt in ('-w','--timewindow'):
      timeWindow = int(arg)
    elif opt in ('-g','--gridsize'):
      grid_spacing = float(arg)
    elif opt in ('-l','--localtime'):
      localTime = True
    elif opt in ('-f','--figurefile'):
      figFile = True
    elif opt in ('-v','--verbose'):
      verbose = True
    elif opt in ('-s','--server'):
      server = arg

  if not('regionS' in locals()):
    print 'Region option -r should be specified.'
    sys.exit()

  if not('timeS' in locals()):
    print 'Time option -t should be specified.'
    sys.exit()

  # region
  regionL = regionS.split('/')
  minLon = float(regionL[0])
  maxLon = float(regionL[1])
  minLat = float(regionL[2])
  maxLat = float(regionL[3])

  if minLon > maxLon:
    print 'maxLon has to be larger than minLon.'
    sys.exit()

  if minLat > maxLat:
    print 'maxLat has to be larger than minLot.'
    sys.exit()

  # time
  if timeS[-1].lower() == 'z':
    dt = datetime.datetime.strptime(timeS,'%Y-%m-%dT%H:%M:%SZ')
  else:
    dt = datetime.datetime.strptime(timeS,'%Y-%m-%dT%H:%M:%S')

  # If the given time is local, it needs to be converted to UTC
  if localTime:
    meanLon = (minLon + maxLon)/2.
    dh = -meanLon/15.
    dt = dt + datetime.timedelta(hours=dh)

  dtS = dt.isoformat()
  yearS = dtS[0:4]
  monthS = dtS[5:7]
  dayS = dtS[8:10]
  hourS = dtS[11:13]
  minuteS = dtS[14:16]
  secondS = dtS[17:19]

  year = int(yearS)
  month = int(monthS)
  day = int(dayS)
  hour = int(hourS)
  minute = int(minuteS)
  second = int(secondS)

  # platform
  if platform.lower() == 'terra':
    if verbose:
      print ''
      print 'MODIS data from Terra satellite being retrieved...'
      print ''
    pflag = 'T'
    data_src = 'MOD05'
  elif platform.lower() == 'aqua':
    if verbose:
      print ''
      print 'MODIS data from Aqua satellite being retrieved...'
      print ''
    pflag = 'A'
    data_src = 'MYD05'
  elif platform.lower() == 'any':
    if verbose:
      print ''
      print 'MODIS data from either Terra or Aqua satellite being retrieved...'
      print ''
    pflag = ''
    data_src = 'MOD05+MYD05'
  else:
    print 'Platform name not supported.'
    sys.exit()

  # outfile
  #if not('grdFname' in locals()):
#    grdFname = ('M'+'_'+regionS.replace('/','_')+'.grd')
#    grdFname = ('M'+'_'+regionS.replace('/','_')+'_'+dt.isoformat()+'.grd')

  # figurefile
 # if figFile:#
#    pngFname = grdFname.replace('grd','png')

  # python version is {mv}.{v}
  # example: 2.6, mv==2, v==6
  mv = int(sys.version[0])
  v = int(sys.version[2])

  use_simplejson = 1

  # python 3.x or higher, assumed to have json
  if mv >= 3:
    use_simplejson = 0
  # python 2.x
  elif mv == 2:
    # python 2.6 or higher has module json
    if v >= 6:
      use_simplejson = 0

  # decide what to import
  if use_simplejson is 1:
    import simplejson as json
  else:
    import json

  # time of interest
  time1 = {
    'year': year,
    'month': month,
    'day': day,
    'hour': hour,
    'minute': minute,
    'second': second,
  }

  start_date = datetime.date(year, month, day)
  end_date = datetime.date(year, month, day)

  startHour = 'T' + hourS + ':' + minuteS + ':' + secondS

  # used for file names
  sth = startHour.replace(':', '')

  if verbose:
    print >> out, ''
    print >> out, 'Client starting ...'
    print >> out, ''

  d = start_date
  delta = datetime.timedelta(days=1)  # one day increment, can be changed
  while d <= end_date:
    date1 = time.strftime("%Y-%m-%d", d.timetuple())
    date2 = time.strftime("%Y%m%d", d.timetuple())

    startTime = date1 + startHour

    ### queryString = 'difid='+data_src+'&t0='+startTime+'&t1='+timeWindow+ \
  	### '&lon0='+str(minLon)+'&lon1='+str(maxLon)+'&lat0='+str(minLat)+'&lat1='+str(maxLat)

    if verbose:
      print 'Searching time in UTC: ',startTime
      print 'On server: ',server

    query = {
  	  "difid": data_src,
  	  "t": [startTime, timeWindow],
  	  "lon": [minLon, maxLon],
  	  "lat": [minLat, maxLat]
    }

    queryString = json.dumps(query)

    if verbose:
      print >> out, 'queryString: '
      print >> out, queryString

      print >> out, ''
      print >> out, '--------------------------------------------------------------------------'
      print >> out, ''

      print >> out, ''
      print >> out, "queryString: %s" % (queryString)
      print >> out, ''

    #------------------
    # STQ service
    # input: temporal and spatial constraints
    # ouput: list of urls for granules found

    if verbose:
      print >> out, "Calling STQ service passing temporal and spatial constraints as input ..."
      print >> out, ''

    url = 'http://oscar.jpl.nasa.gov/service/stq'
    url = "%s?%s" % (url, urllib.quote(queryString))
    if verbose: print >> out, 'url: ' + url

    result1 = urllib2.urlopen(url).read()
    if verbose: print >> out, '----- STQ result: ' + result1

    d1 = json.loads(result1)

    # collect urls and paths
    urls = d1['result']
    paths = []

    # This is a kluge; the client should have no knowledge
    # of any server directory
    ### prefix1 = 'http://oscar1.jpl.nasa.gov'
    ### prefix2 = '/export'
    ### for i in range(len(urls)):
  	### paths.append(urls[i].replace(prefix1, prefix2))

    listOfGranuleURLs = urls['uris']
    ### listOfGranuleURLs = paths
    average_time = urls['average_time']

    if verbose:
      print >> out, 'paths of granules returned from STQ service:'
      print >> out, listOfGranuleURLs
      print >> out, 'average time of these granules:'
      print >> out, 'End of STQ call.'
      print >> out, ''
    print 'Average data acquisition time: ',average_time

    if not('grdFname' in locals()):
       grdFname = ('M'+'_'+regionS.replace('/','_')+'_'+average_time+'.grd')
  #    grdFname = ('M'+'_'+regionS.replace('/','_')+'.grd')
      #grdFname = ('M'+'_'+regionS.replace('/','_')+'_'+dt.isoformat()+'.grd')

    # figurefile
    if figFile:
      pngFname = grdFname.replace('grd','png')

    if (len(listOfGranuleURLs) <= 0):
  	  print >> out, 'STQ found no granules. No need to go further.'
  	  # increase date by delta day
  	  d += delta
  	  continue  # skip to the next loop

    #------------------
    # SAMI service
    # subset, merge, and interpolation service
    # input: (*) temporal and spatial constraints
    #        (*) list of granule urls
    #        (*) user time of interest
    #        (*) grid spacing of user chosen grid
    #        (*) flag for nearest neighbor algorithm or spatial averaging & time interpolation
    #        (*) flag for subset & merge algorithm (in python) or merge interpolation algorithm (in C)
    # output: url for result in netcdf

    if verbose:
      print >> out, ''
      print >> out, "Calling SAMI service for subset, merge, spatial average, and time interpolation of the Near IR Water Vapor data ..."
      print >> out, ''

    georegion = {
      'lon': (minLon, maxLon),
      'lat': (minLat, maxLat),
    }

    granules = listOfGranuleURLs

    query = {
  	  "georegion":georegion,
  	  "granules":granules,
  	  "time":time1,
  	  "grid_spacing":grid_spacing,
  	  "flag_nearest":False,
  	  "flag_subset":-1
    }

    queryString1 = json.dumps(query)

    if verbose:
      print >> out, 'query input: ', queryString1
      print >> out, ''

    if server.lower() == 'oscar1':
      #prefix2 = 'http://oscar.jpl.nasa.gov:28888'
      prefix2 = 'http://oscar.jpl.nasa.gov'
    elif server.lower() == 'oscar2':
      prefix2 = 'http://oscar2.jpl.nasa.gov:28888'  # this is for JPL internal
    else:
      print 'Server has to be either "oscar1" or "oscar2"'
      sys.exit()

    serviceEndpoint = prefix2 + '/service/modis_sami'

    request = urllib2.Request(url=serviceEndpoint, data=queryString1)
    response = urllib2.urlopen(request).read()

    if verbose:
      print >> out, 'SAMI response:'
      print >> out, response
      print >> out, 'End of SAMI call.'
      print >> out, ''

    #------------------
    # grab result in netcdf and dump as (x,y,z) for GMT

    d1 = json.loads(response)
    pathString = d1["result"]

    pathStringGrd = pathString["grd_uri"]
    if verbose:
      print >> out, 'Parsing SAMI response to get the path of the grd file:'
      print >> out, 'grd file: ', pathStringGrd
    if cmp(pathStringGrd, "") == 0:
  	  print >> out, ''
  	  print >> out, 'SAMI did not create any grd file.'
  	  print >> out, 'Thanks for watching the demo! Client ending ...'
  	  print >> out, ''
    pathGrd = 'tmp_output.grd'
    if verbose:
      print >> out, ''
      print >> out, 'Downloading the grd file to local dir as: ', pathGrd
      print >> out, ''
    urlStringGrd = prefix2 + pathStringGrd
    if verbose:
      print >> out, 'urlStringGrd: ', urlStringGrd

    request = urllib2.Request(url=urlStringGrd)
    response = urllib2.urlopen(request).read()
    f = open(pathGrd, 'w')
    f.write(response)
    f.close()

    if figFile:
      pathStringPng = pathString["png_uri"]
      if verbose:
        print >> out, 'Parsing SAMI response to get the path of the png file:'
        print >> out, 'png file: ', pathStringPng
      if cmp(pathStringPng, "") == 0:
  	    print >> out, ''
  	    print >> out, 'SAMI did not create any png file.'
  	    print >> out, 'Thanks for watching the demo! Client ending ...'
  	    print >> out, ''
  	    sys.exit(-1)
      pathPng = 'PWV.png'
      if verbose:
        print >> out, ''
        print >> out, 'Downloading the png file to local dir as: ', pathPng
        print >> out, ''
      urlStringPng = prefix2 + pathStringPng
      if verbose:
        print >> out, 'urlStringPng: ', urlStringPng

      request = urllib2.Request(url=urlStringPng)
      response = urllib2.urlopen(request).read()
      f = open(pathPng, 'w')
      f.write(response)
      f.close()

    # increase date by delta day
    d += delta

  print >> out, ''
  print >> out, 'Requested grd file stored as "'+grdFname+'".'
  cmd = 'mv tmp_output.grd '+grdFname
  os.system(cmd)

  if figFile:
    print >> out, 'Requested png file stored as "'+pngFname+'".'
    cmd = 'mv PWV.png '+pngFname
    os.system(cmd)

if __name__ == "__main__":
  start_time_main = time.time()
  main()
  time_elapsed = time.time() - start_time_main
  print 'Elapsed time: %8.2f seconds.' % time_elapsed
  print ''
