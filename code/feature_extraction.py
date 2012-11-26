import numpy as np
import scipy as sp
import dataset
import caching
import options
import sys
from misc import print_progress
from canonization import canonize

from skimage.measure import regionprops
from skimage.morphology import label
#from skimage.feature import hog
from hog import hog

from sklearn.cluster import KMeans

@caching.cache
def feature_training(opts, params):
  print '# Feature extraction training'
  if 'hog_bow' in opts:
    opts_ = opts['hog_bow']
    descs = []
    files = dataset.training_files(opts_['num_train_images'])
    for img_file, depth_file in print_progress(files):
      img, segmask = canonize(img_file, depth_file, opts['canonization'], params)
      h = hog(img, **opts_['hog'])
      segmask = sp.misc.imresize(segmask, h.shape[:2])
      for i in range(h.shape[0]):
        for j in range(h.shape[1]):
          if segmask[i,j] != 0:
            descs.append(h[i,j,:,:,:].flatten())
    descs = np.vstack(tuple(descs))
    print '# K-means clustering of %i features of dimensionality %i'%(descs.shape[0], descs.shape[1])
    kmeans = KMeans(opts_['num_clusters'], n_jobs=options.num_threads)
    kmeans.fit(descs)
    params['hog_bow_kmeans'] = kmeans
  return params

@caching.cache
def feature_extraction(img_file, depth_file, opts, params):
  img, segmask = canonize(img_file, depth_file, opts['canonization'], params)

  features = np.array([], dtype=float)
  if 'region_properties' in opts:
    opts_ = opts['region_properties']
    if 'grid' in opts_:
      grid = opts_['grid']
      segmask[segmask!=0] = 1
      cellNum = 0
      cellHeight = segmask.shape[0]/grid[0]
      cellWidth = segmask.shape[1]/grid[1]
      for i in range(grid[0]):
        for j in range(grid[1]):
          cellNum += 1
          t = cellHeight*i
          b = cellHeight*(i+1)
          l = cellWidth*j
          r = cellWidth*(j+1)
          segmask[t:b, l:r] = segmask[t:b, l:r] * cellNum
    props = regionprops(segmask.astype(int), properties=opts_['properties'], intensity_image=img)
    featureList = []
    for i in range(len(props)):
      for p in opts_['properties']:
        featureList.append(np.array(props[i][p]).flatten())
    features = np.append(features, np.hstack(tuple(featureList)))


  if 'raw_pixels' in opts:
    img_ = sp.misc.imresize(img, opts['raw_pixels']['img_scale'])
    features = np.append(features, img_.flatten())

  if 'hog' in opts:
    h = hog(img, **opts['hog'])
    h = h.flatten()
    features = np.append(features, h.flatten())

  if 'daisy' in opts:
    pass

  if 'hog_bow' in opts:
    opts_ = opts['hog_bow']
    for woop in range(2):
      features_ = np.array([], dtype=float)
      descs = []
      if woop == 0:
        h = hog(img, **opts_['hog'])
      else:
        h = hog(img[4:,4:], **opts_['hog'])
      segmask_ = sp.misc.imresize(segmask, h.shape[:2])
      for i in range(h.shape[0]):
        for j in range(h.shape[1]):
          descs.append(h[i,j,:,:,:].flatten())


      kmeans = params['hog_bow_kmeans']
      clusters = kmeans.predict(descs)
      grid = opts_['grid']
      clusters = clusters.reshape((h.shape[0],(h.shape[1])))
      clusters = np.ma.array(clusters, mask = segmask_)
      cellHeight = clusters.shape[0]/grid[0]
      cellWidth = clusters.shape[1]/grid[1]
      for i in range(grid[0]):
        for j in range(grid[1]):
          t = cellHeight*i
          b = cellHeight*(i+1)
          l = cellWidth*j
          r = cellWidth*(j+1)
          hist = np.bincount(clusters[t:b, l:r].flatten(), minlength=opts_['num_clusters'])
          hist = hist.astype(float)
  #          hist /= np.max(hist)
          features_ = np.append(features_, hist)
      if woop == 0:
        features = np.append(features, features_)
      else:
        features += features_
#  print features.shape
#  sys.exit(0)
  return features

