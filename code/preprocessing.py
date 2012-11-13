import os
import os.path
import struct
import numpy as np
import scipy as sp
import scipy.misc
import misc
import matplotlib.pyplot as plt

def getBytes(f, start, end):
  f.seek(start)
  return f.read(end-start)

def intValue(bytes):
  return struct.unpack("<L", bytes)[0]

def shortValue(bytes):
  return struct.unpack("H", bytes)[0]


def make_depth_images(opts):
  filepaths = misc.gatherFiles(opts['datasetPath'], '*dybde*')
  for fileIn in filepaths:
    print fileIn
    fileOut = os.path.splitext(fileIn)[0].replace('input', 'working/preprocessing')
    fileOutDir = os.path.dirname(fileOut)
    if not os.path.exists(fileOutDir):
      os.makedirs(fileOutDir)
    with open(fileIn, 'rb') as f:
      byte = f.read(1)
      i = 0
      startAddress = intValue(getBytes(f,10,14))
      size = intValue(getBytes(f,2,6))
      w = intValue(getBytes(f,18,22))
      h = intValue(getBytes(f,22,26))
      depths = np.zeros((w*h), dtype=np.dtype('H'))
      for i in range(w*h):
        offset = startAddress + i*4
        depths[i] = shortValue(getBytes(f,offset,offset+2))
      mx = np.max(depths)
      mn = np.min(depths)
      depths[depths>=mx]=mn
      depths = mx - depths
      depths = depths.reshape(h,w)
      depths = np.flipud(depths)
      if 'Dag 1' in fileIn:
        depths = depths[137:137+218, 37:37+450]
      elif 'Dag 2' in fileIn:
        depths = depths[136:136+218, 68:68+450]
      np.save(fileOut, depths)
      sp.misc.imsave(fileOut+'.png', depths)

def make_cropped_images(opts):
  filepaths = misc.gatherFiles(opts['datasetPath'], '*_kam*')
  for fileIn in filepaths:
    print fileIn
    fileOut = os.path.splitext(fileIn)[0].replace('input', 'working/preprocessing')
    fileOutDir = os.path.dirname(fileOut)
    if not os.path.exists(fileOutDir):
      os.makedirs(fileOutDir)
    img = sp.misc.imread(fileIn)
    if 'Dag 1' in fileIn:
      img = img[331:331+398, 110:110+821]
    elif 'Dag 2' in fileIn:
      img = img[332:332+398, 167:167+821]
    sp.misc.imsave(fileOut+'.png', img)

def preprocessing(opts):
  make_depth_images(opts)
  make_cropped_images(opts)
  
