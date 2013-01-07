import os
import fnmatch
import sys
import copy
import matplotlib.pyplot as plt

from multiprocessing import Process, Pipe
from itertools import izip


def save_img(path, img):
  '''Save image to file using matplotlib'''
  plt.imshow(img, interpolation='nearest')
  plt.savefig(path)


def gather_files(path, pattern=None):
  ''' Recursively find all files and filter using the given pattern'''
  result = []
  for (root, dirs, files) in os.walk(path):
    if pattern:
      result.extend([os.path.join(root,f) for f in fnmatch.filter(files, pattern)])
    else:
      result.extend([os.path.join(root,f) for f in files])
  return result


def print_progress(iterable, bar_width=50, length=None):
  ''' Wrap iterable in a generator that prints a progress bar to the
      command line as the iterable is traversed.'''
  if length == None:
    total = float(len(iterable))
  else:
    total = float(length)
  for num, obj in enumerate(iterable):
    num = num + 1
    percent = num/total
    hashsigns = int(percent*bar_width)
    sys.stdout.write('\r['+'#'*hashsigns + ' '*(bar_width-hashsigns) + '] %i/%i'%(num,total))
    if percent == 1.0:
      sys.stdout.write('\n')
    sys.stdout.flush()
    yield obj

def dicreplace(dic, path, value):
  diccopy = copy.deepcopy(dic)
  node = diccopy
  for p in path[:-1]:
    node = node[p]
  node[path[-1]] = value
  return diccopy

def dicvariations(dic, variations):
  dics = [(dic, '')]
  if len(variations) == 0:
    dics = [(dic, 'noname')]
  else:
    for name, path, values in reversed(variations):
      dics = [(dicreplace(d, path, v), d_name+name+str(v)+'_')
          for v in values for d, d_name in dics]
    dics = [(d, d_name[:-1]) for d, d_name in dics]
  return dics

#if __name__ == '__main__':
#  woop = {
#    'a': {
#      'aa': 0,
#      'ab': 0,
#    },
#    'b': {
#      'ba': 0,
#      'bb': 0,
#    }
#  }
#  variations = [
#      ('a', ['a', 'aa'], [1, 2]),
#      ('b', ['b', 'bb'], [3, 4]),
#  ]
#  print dicvariations(woop, variations)

##  print vary(woop, ['a'], [35, 1, 2])
##  print w in over
