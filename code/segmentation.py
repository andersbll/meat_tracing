import os
import os.path
import numpy as np
import scipy as sp
import scipy.misc
import misc

import matplotlib.pyplot as plt
from pygco import cut_simple, cut_from_graph

def segmentation(opts):
  filepaths = misc.gatherFiles(os.path.join(opts['workingPath'],'preprocessing'), '*_dybde*.npy')
  #potts pairwise potential used for graph cut algorithm
  pairwise = (-100 * np.eye(2)).astype(np.int32)
  for fileIn in filepaths:
    print fileIn
    fileInImg = fileIn.replace('_dybde','_kam').replace('npy','png')
    fileOut = os.path.splitext(fileIn)[0].replace('preprocessing', 'segmentation')
    fileOutDir = os.path.dirname(fileOut)
    if not os.path.exists(fileOutDir):
      os.makedirs(fileOutDir)
    fileOutSegmentationMask = fileOut.replace('_dybde','_kam_segmask') + '.npy'
    fileOutImgSegmented = fileOut.replace('_dybde','_kam_seg') + '.png'

    depths = np.load(fileIn)
    depths = depths.astype(np.int32)
    depths = depths - opts['segmentation']['boardDepth']
    depths = (np.dstack([depths, -depths]).copy("C")).astype(np.int32)
    segMask = cut_simple(depths.astype(np.int32), pairwise.astype(np.int32))
    img = sp.misc.imread(fileInImg)
    mask = sp.misc.imresize(segMask, (img.shape[0], img.shape[1]))#, interp='nearest')
    img[mask==0] = 0
    sp.misc.imsave(fileOutImgSegmented, img)
    np.save(fileOutSegmentationMask, segMask)

