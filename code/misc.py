import os
import os.path
import fnmatch
import sys
import matplotlib.pyplot as plt


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


def print_progress(iterable, bar_width=50):
  ''' Wrap iterable in a generator that prints a progress bar to the
      command line as the iterable is traversed.'''
  total = float(len(iterable))
  for num, obj in enumerate(iterable):
    num = num + 1
    percent = num/total
    hashsigns = int(percent*bar_width)
    sys.stdout.write('\r['+'#'*hashsigns + ' '*(bar_width-hashsigns) + '] %i/%i'%(num,total))
    if percent == 1.0:
      sys.stdout.write('\n')
    sys.stdout.flush()
    yield obj




