import numpy as np
import scipy as sp

from sklearn.metrics.pairwise import pairwise_distances
from munkres import Munkres

def matching(features1, features2, opts):
#  print '# Matching'
  distances = pairwise_distances(features1, features2, metric=opts['metric'])

  if opts['bipartite_matching']:
#    print '# Bipartite matching'
    m = Munkres()
    matrix = []
    for i in range(distances.shape[0]):
      row = []
      for j in range(distances.shape[1]):
        row.append(distances[i,j])
      matrix.append(row)
    indexes = m.compute(matrix)
    distances[:] = 1
    for row, column in indexes:
      distances[row,column] = 0

  return distances
