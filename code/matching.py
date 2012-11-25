import numpy as np
import scipy as sp

from sklearn.metrics.pairwise import pairwise_distances

def matching(features1, features2, opts):
  print '# Matching'
  distances = pairwise_distances(features1, features2, metric=opts['metric'])
  return distances
