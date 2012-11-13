import os
import os.path
import struct
import numpy as np
import scipy as sp
import scipy.misc
import misc

def getBytes(f, start, end):
  f.seek(start)
  return f.read(end-start)

def intValue(bytes):
  return struct.unpack("<L", bytes)[0]

def shortValue(bytes):
  return struct.unpack("H", bytes)[0]


def make_depth_images(opts):
  print '- Extracting and cropping depth maps from BMP files'
  filepaths = misc.gatherFiles(opts['datasetPath'], '*dybde*')
  fileNum = 0
  for fileIn in filepaths:
    fileNum += 1
    misc.printProgress(float(fileNum)/len(filepaths))
    fileOut = os.path.splitext(fileIn)[0].replace('input', 'working/preprocessing')
    fileOutDir = os.path.dirname(fileOut)
    if not os.path.exists(fileOutDir):
      os.makedirs(fileOutDir)
    with open(fileIn, 'rb') as f:
      # Extract BMP header
      startAddress = intValue(getBytes(f,10,14))
      size = intValue(getBytes(f,2,6))
      w = intValue(getBytes(f,18,22))
      h = intValue(getBytes(f,22,26))

      # Extract depth values
      depths = np.zeros((w*h), dtype=np.dtype('H'))
      for i in range(w*h):
        offset = startAddress + i*4
        depths[i] = shortValue(getBytes(f,offset,offset+2))

      # Unknown depth values are represented as the maximal unsigned
      # short value. We overwrite unknown depth values with the depth of
      # the surface on which the meat is placed.
      depths[depths==65535] = opts['boardDepth']
      depths = depths.reshape(h,w)
      depths = np.flipud(depths)
      if 'Dag 1' in fileIn:
        depths = depths[137:137+218, 37:37+450]
      elif 'Dag 2' in fileIn:
        depths = depths[136:136+218, 68:68+450]
      np.save(fileOut, depths)
      sp.misc.imsave(fileOut+'.png', depths)

def make_cropped_images(opts):
  print '- Cropping image files'
  filepaths = misc.gatherFiles(opts['datasetPath'], '*_kam*')
  fileNum = 0
  for fileIn in filepaths:
    fileNum += 1
    misc.printProgress(float(fileNum)/len(filepaths))
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
  print 'Preprocessing'
  make_depth_images(opts)
  make_cropped_images(opts)

