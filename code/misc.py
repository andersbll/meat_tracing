import os
import os.path
import fnmatch
import sys
import matplotlib.pyplot as plt


def saveImg(path, img):
  '''Save image to file using matplotlib'''
  plt.imshow(img)
  plt.savefig(path)


def gatherFiles(path, pattern=None):
  ''' Recursively find all files and filter using the given pattern'''
  result = []
  for (root, dirs, files) in os.walk(path):
    if pattern:
      result.extend([os.path.join(root,f) for f in fnmatch.filter(files, pattern)])
    else: 
      result.extend([os.path.join(root,f) for f in files])
  return result


def printProgress(percent, length=50):
  '''Print a progress bar to the command line.
     percent should be a float value between 0 and 1'''
  hashsigns = int(percent*length)
  sys.stdout.write('\r['+'#'*hashsigns + ' '*(length-hashsigns) + ']')
  if percent == 1.0:
    sys.stdout.write('\n')
  sys.stdout.flush()
