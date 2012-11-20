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

def preprocessing(opts):
  print '# Preprocessing'
  filepaths = misc.gatherFiles(opts['datasetPath'], '*_dybde*')
  fileNum = 0
  for inDepth in filepaths:
    fileNum += 1
    misc.printProgress(fileNum, len(filepaths))
    inImg = inDepth.replace('_dybde', '_kam')
    outImg = inImg.replace('input', 'working/preprocessing').replace('.bmp', '.png')
    outDepth = os.path.splitext(inDepth)[0].replace('input', 'working/preprocessing')
    misc.ensureDir(outDepth)

    # Crop image
    img = sp.misc.imread(inImg)
    if 'Dag 1' in inImg:
      img = img[331:331+398, 110:110+821]
    elif 'Dag 2' in inImg:
      img = img[332:332+398, 167:167+821]
    sp.misc.imsave(outImg, img)

    # Extract and crop depth map
    with open(inDepth, 'rb') as f:
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
      if 'Dag 1' in inDepth:
        depths = depths[137:137+218, 37:37+450]
      elif 'Dag 2' in inDepth:
        depths = depths[136:136+218, 68:68+450]
      np.save(outDepth, depths)

