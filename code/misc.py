import os
import os.path
import fnmatch


def gatherFiles(path, pattern=None):
  ''' Recursively find all files and filter using the given pattern'''
  result = []
  for (root, dirs, files) in os.walk(path):
    if pattern:
      result.extend([os.path.join(root,f) for f in fnmatch.filter(files, pattern)])
    else: 
      result.extend([os.path.join(root,f) for f in files])
  return result

