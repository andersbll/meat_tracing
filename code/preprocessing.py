import struct
import numpy as np
import scipy as sp
import caching

def getBytes(f, start, end):
  f.seek(start)
  return f.read(end-start)

def intValue(bytes):
  return struct.unpack("<L", bytes)[0]

def shortValue(bytes):
  return struct.unpack("H", bytes)[0]

@caching.cache
def preprocess_depth(depth_file, opts):
  # Extract and crop depth map
  with open(depth_file, 'rb') as f:
    # Extract BMP header
    startAddress = intValue(getBytes(f,10,14))
    size = intValue(getBytes(f,2,6))
    w = intValue(getBytes(f,18,22))
    h = intValue(getBytes(f,22,26))

    # Extract depth values
    depth = np.zeros((w*h), dtype=np.dtype('H'))
    for i in range(w*h):
      offset = startAddress + i*4
      depth[i] = shortValue(getBytes(f,offset,offset+2))

    # Unknown depth values are represented as the maximal unsigned
    # short value. We overwrite unknown depth values with the depth of
    # the surface on which the meat is placed.
    depth[depth==65535] = opts['board_depth']
    depth = depth.reshape(h,w)
    depth = np.flipud(depth)
    if 'Dag 1' in depth_file:
      depth = depth[137:137+218, 37:37+450]
    elif 'Dag 2' in depth_file:
      depth = depth[136:136+218, 68:68+450]
    return depth

@caching.cache
def preprocess_image(img_file):
  # Crop image
  img = sp.misc.imread(img_file)
  if 'Dag 1' in img_file:
    img = img[331:331+398, 110:110+821]
  elif 'Dag 2' in img_file:
    img = img[332:332+398, 167:167+821]
  return img
