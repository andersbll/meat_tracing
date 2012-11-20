import os
import os.path
import numpy as np
import scipy as sp
import scipy.misc
import misc

from pygco import cut_simple, cut_from_graph

def segmentation(opts):
  print '# Segmentation'
  filepaths = misc.gatherFiles(os.path.join(opts['workingPath'],'preprocessing'), '*_dybde*.npy')
  # Potts pairwise potential used for graph cut algorithm
  pairwise = (-100 * np.eye(2)).astype(np.int32)
  fileNum = 0
  for inDepth in filepaths:
    fileNum += 1
    misc.printProgress(fileNum, len(filepaths))
    outSegMask = inDepth.replace('preprocessing', 'segmentation').replace('_dybde','_segmask')
    misc.ensureDir(outSegMask)

    depths = np.load(inDepth)
    depths = depths.astype(np.int32)

    depths = depths - opts['boardDepth']
    depths = (np.dstack([depths, -depths]).copy("C")).astype(np.int32)
    segMask = cut_simple(depths.astype(np.int32), pairwise.astype(np.int32))
    # Invert mask
    segMask = (segMask+1) % 2
    np.save(outSegMask, segMask)

