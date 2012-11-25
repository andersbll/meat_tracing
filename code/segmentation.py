import numpy as np
import caching
from preprocessing import preprocess_depth
from pygco import cut_simple, cut_from_graph

@caching.cache
def segmentation(depth_file, opts):
  depth = preprocess_depth(depth_file, opts['preprocess_depth'])
  # Potts pairwise potential used for graph cut algorithm
  pairwise = (-100 * np.eye(2)).astype(np.int32)
  depth = depth.astype(np.int32)
  depth = depth - opts['board_depth']
  depth = (np.dstack([depth, -depth]).copy("C")).astype(np.int32)
  segmask = cut_simple(depth.astype(np.int32), pairwise.astype(np.int32))
  # Invert mask
  segmask = (segmask+1) % 2
  return segmask

