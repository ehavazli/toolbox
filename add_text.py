#! /usr/bin/env python
import sys
import os
import getopt

def Usage():
  print '''
****************************************************************
****************************************************************
  This function adds text to the begining and end of each line in a text file and
  saves the new file with given name

  -f: source file for the text to be added
  -d: name of the file to be created
  -p: text to add to the begining
  -s: text to add to the end

Example:

  add_text.py -f list.txt -d list_new.txt -p 'prefix' -s 'suffix'

**Consider leaving a space at the end(for prefix) or beginning(for suffix) if you want the new input seperated**
 
****************************************************************
****************************************************************           
  '''

def main(argv):
  try:
    opts, args = getopt.getopt(argv,"f:d:p:s:")

  except getopt.GetoptError:
    Usage() ; sys.exit(1)

  prefix = ''
  suffix = ''

  if opts==[]:
    Usage() ; sys.exit(1)
  for opt,arg in opts:
    if opt in ("-h","--help"):
      Usage()
      sys.exit()
    elif opt == '-f':
      source = str(arg)
    elif opt == '-d':
      dest = str(arg)
    elif opt == '-p':
      prefix = str(arg)
    elif opt == '-s':
      suffix = str(arg)

  with open(source, 'r') as src:
    with open(dest,'w') as dst:
        for line in src:
            dst.write('%s%s%s\n' % (prefix, line.rstrip('\n'),suffix))
##########################################################################
if __name__ == '__main__':

  main(sys.argv[1:])

