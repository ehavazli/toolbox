#! /usr/bin/env python
import os
import sys
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
#f=open(timestr,'w')
scratchdir='/famelung/'

def directory_size(path):
    total_size = 0
    seen = set()

    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)

            try:
                stat = os.stat(fp)
            except OSError:
                continue

            if stat.st_ino in seen:
                continue

            seen.add(stat.st_ino)

            total_size += stat.st_size

    return total_size  # size in bytes


def sizeof_fmt(num, suffix=''):
    for unit in ['','Kb','Mb','Gb','Tb','Pb','Eb','Zb']:
        if abs(num) < 1024.00:
            return "%3.2f%s%s" % (num, unit, suffix)
        num /= 1024.00
    return "%.2f%s%s" % (num, 'Yi', suffix)

user_list=os.listdir(scratchdir)

for x in user_list:
	size=(directory_size(scratchdir+x))
	size_hr=sizeof_fmt(size)
	#f.write('\n'+"User: " + x)
	#f.write('\n'+"Size: " + size_hr)
        print "User: " + x
        print "Size: " + size_hr

size_total=(directory_size(scratchdir))
size_total_hr=sizeof_fmt(size_total)
#f.write('\n'+"Total Size: " + size_total_hr)
print "Total Size: " + size_total_hr
#f.close()
