#! /usr/bin/env python
############################################################
# Author:  Heresh Fattahi                                  #
# Nov 2013
############################################################

import sys
import os
import glob
import readfile
import re
import numpy as np


def main(argv):
  try:
    templateFileString = argv[1]
    referencePairDir = argv[2]


  except:
    print "  ******************************************************************************************************"
    print "  ******************************************************************************************************"
    print "  ******************************************************************************************************"
    print " "
    print "  Coregistration of interferograms and correlation files to a Master date." 
    print " "
    print "  Usage: Coregist_all_To_masterDate.py templateFile referenceIfgramDir"
    print " "
    print "  Example: "
    print "       Coregist_all_To_masterDate.py $TE/SanAndreasT356EnvD.template IFGRAM_SanAndreasT356EnvD_030627-031219_0175_-0317"
    print " "
    print "  ******************************************************************************************************"
    print "  ******************************************************************************************************"
    print "  ******************************************************************************************************"
    sys.exit(1)



  templateContents = readfile.read_template(templateFileString)
  masterDate = templateContents['masterOffsetdate']
  projectName = os.path.basename(templateFileString.partition('.')[0])
  slcDir = os.getenv('SLCDIR') +'/'+ projectName
  processDir = os.getenv('PROCESSDIR')+'/'+ projectName
  referencePairDir=processDir +'/'+referencePairDir

  IFGs= glob.glob(processDir+'/IFG*')
  IFG_Dirs = [x for x in IFGs if '.proc' not in x]
  
  

  IFG_Dirs.remove(referencePairDir)

  os.chdir(processDir)
  f=open('run_CoregistToMasterDate','w')
  for IFG in IFG_Dirs:
     try:
        filt_int=glob.glob(IFG+'/*-*-sim_HDR_*.int')[0]
        filt_cor=glob.glob(IFG+'/*-*-sim_HDR_*.cor')[0]
        radar_dem = glob.glob(IFG+'/radar_*rlks.hgt')[0]

        cmd_filt_int = 'CoregistToMasterdate.py '+ filt_int + ' '+templateFileString+' '+referencePairDir+'\n'
        cmd_filt_cor = 'CoregistToMasterdate.py '+ filt_cor + ' '+templateFileString+' '+referencePairDir+'\n'
#     cmd_radar_dem = 'CoregistToMasterdate.py '+ radar_dem + ' '+templateFileString+' '+referencePairDir+'\n'
        f.write(cmd_filt_int)
     except:
        print 'Warning:'
        print 'no file added from the following folder'
        print IFG
     f.write(cmd_filt_cor)
#     f.write(cmd_radar_dem)    
  f.close()

  fname='Coregist.process'
  f = open(fname,'w')

  f.write('#! /bin/csh')
  f.write('\n#BSUB -J xaa')
  f.write('\n#BSUB -o xaa.o%J')
  f.write('\n#BSUB -e xaa.e%J')
  f.write('\n#BSUB -W 5:00')
  f.write('\n#BSUB -q general')
  f.write('\n#BSUB -n 8')
  f.write('\n#BSUB -R "span[hosts=1]"')
  f.write('\n#BSUB -B')
  f.write('\n#BSUB -N')
  f.write('\ncd '+processDir)
  f.write('\nrunJobs.py run_CoregistToMasterDate 8')
  
  f.close()
  jobCmd='bsub < '+processDir+'/Coregist.process'
  os.system(jobCmd)

#  IFG=os.path.dirname(IFG_Dirs[3])+'/IFGRAM_SanAndreasT356EnvD_031219-050211_0420_00033/'

#  filt_int=glob.glob(IFG+'/filt_*int')[0]
#  print filt_int
#  filt_cor=glob.glob(IFG+'/filt_*cor')[0]
#  print filt_cor

#  cmd_filt_int='CoregistToMasterdate.py '+ filt_int + ' '+templateFileString+' '+referencePairDir
#  print cmd_filt_int
#  os.system(cmd_filt_int)

#  cmd_filt_cor='CoregistToMasterdate.py '+ filt_cor + ' '+templateFileString+' '+referencePairDir
#  print cmd_filt_cor
#  os.system(cmd_filt_cor)

if __name__ == '__main__':
  main(sys.argv[:])


