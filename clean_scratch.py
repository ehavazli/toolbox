#! /usr/bin/env python
##########################
#Author : Emre HAVAZLI   #
##########################

import os, time, sys
from datetime import datetime, timedelta
import shutil
import getopt

def Usage():
    print '''
****************************************************************
****************************************************************

  Clean /scratch/projects/insarlab/ directory from folders older than
  the chosen number of days

   Usage: 
         clean_scratch.py <number of days> --remove     "Remove directories older than given number of days"
         clean_scratch.py <number of days>              "List directories older than given number of days"
         clean_scratch.py --remove                      "Remove directories older than 7 days"

****************************************************************
'''

def main(argv):
    total = len(sys.argv)
    if total > 1 : 
        argv_list =(sys.argv)
    else: Usage() ; sys.exit(1)
    
    if total < 3: 
       try:
        x = int(argv_list[1])+1
        nod = int(argv_list[1])
        path = "/scratch/projects/insarlab/"
        user_list = os.listdir(path)

        date_N_days_ago = datetime.now() - timedelta(days=nod)
        now = time.time()
        print "Cleaning /scratch/insarlab for folders last modified before: "+str(date_N_days_ago)

        for user in user_list:
                print user
                path = "/scratch/projects/insarlab/"+user
                for f in os.listdir(path):
                    f = os.path.join(path,f)
                    if 'TESTBENCH' in f:
                        for g in os.listdir(f):
                            g = os.path.join(f,g)
                            if os.stat(g).st_mtime > now - nod * 86400: pass
                            else : print 'TESTBENCH FOLDER older than '+str(nod)+' days: ' + g

                    if os.stat(f).st_mtime > now - nod * 86400: pass
                    else : print 'Folder older than '+str(nod)+' days: '+ f

       except: 
        nod = 7 
        remove = 'yes'
        path = "/scratch/projects/insarlab/"
        user_list = os.listdir(path)

        date_N_days_ago = datetime.now() - timedelta(days=nod)
        now = time.time()
        print "Cleaning /scratch/insarlab for folders last modified before: "+str(date_N_days_ago)

        for user in user_list:
                print user
                path = "/scratch/projects/insarlab/"+user
                for f in os.listdir(path):
                    f = os.path.join(path,f)
                    if 'TESTBENCH' in f: 
                        for g in os.listdir(f):
                            g = os.path.join(f,g)
                            if os.stat(g).st_mtime > now - nod * 86400: pass
                            else : print 'TESTBENCH SUB-DIRS older than '+str(nod)+' days are REMOVED: ' + g
                    else:
                        if os.stat(f).st_mtime > now - nod * 86400: pass
                        else : print 'Directories older than '+str(nod)+' days are REMOVED: ' + f
    if total == 3 :
        try:
         x = int(argv_list[1])+1
         argv_list[2] == '--remove'
         remove = 'yes'
         nod = int(argv_list[1])
         path = "/scratch/projects/insarlab/"
         user_list = os.listdir(path)

         date_N_days_ago = datetime.now() - timedelta(days=nod)
         now = time.time()
         print "Cleaning /scratch/insarlab for folders last modified before: "+str(date_N_days_ago)
 
         for user in user_list:
                print user
                path = "/scratch/projects/insarlab/"+user
                for f in os.listdir(path):
                    f = os.path.join(path,f)
                    if 'TESTBENCH' in f:
                        for g in os.listdir(f):
                            g = os.path.join(f,g)
                            if os.stat(g).st_mtime > now - nod * 86400: pass
                            else : print 'TESTBENCH SUB-DIRS older than '+str(nod)+' days are REMOVED: ' + g
                    else:
                        if os.stat(f).st_mtime > now - nod * 86400: pass
                        else : print 'Directories older than '+str(nod)+' days are REMOVED: ' + f
        except: Usage()

##shutil.rmtree(f, ignore_errors=True)

##########################################################################
if __name__ == '__main__':

  main(sys.argv[1:])
