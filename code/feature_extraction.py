import numpy as np
import scipy as sp
import dataset
import caching
import options
from misc import print_progress
from canonization import canonize

from skimage.measure import regionprops
from skimage.morphology import label
from daisy import daisy

from sklearn.cluster import KMeans

@caching.cache
def feature_training(opts, params):
  print '# Feature extraction training'
  if 'daisy_bow' in opts:
    opts_ = opts['daisy_bow']
    descs = []
    files = dataset.training_files(opts_['num_train_images'])
    for img_file, depth_file in print_progress(files):
      img, segmask = canonize(img_file, depth_file, opts['canonization'], params)
      h = daisy(img, **opts_['daisy'])
      segmask = sp.misc.imresize(segmask, h.shape[:2])
      for i in range(h.shape[0]):
        for j in range(h.shape[1]):
          if segmask[i,j] != 0:
            descs.append(h[i,j,:].flatten())
    descs = [descs[i] for i in range(0, len(descs), 3)]
    descs = np.vstack(tuple(descs))
    print '# K-means clustering of %i features of dimensionality %i'%(descs.shape[0], descs.shape[1])
    kmeans = KMeans(opts_['num_clusters'], n_jobs=options.num_threads)
    kmeans.fit(descs)
    params['daisy_bow_kmeans'] = kmeans
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

  if 'daisy_bow' in opts:
    opts_ = opts['daisy_bow']
    features_ = np.array([], dtype=float)
    d = daisy(img, **opts_['daisy'])
    descs = d.reshape(d.shape[0]*d.shape[1],d.shape[2])
    kmeans = params['daisy_bow_kmeans']
    clusters = kmeans.predict(descs)
    clusters = clusters.reshape(d.shape[:2])
    gauss_centers = grid_centers(clusters.shape,
        opts_['gauss_window_grid'])
    gauss_win = gauss_windows(clusters.shape, gauss_centers,
        opts_['gauss_window_sigma'])
    hists = np.empty((gauss_win.shape[0], opts_['num_clusters']))
    segmask_ = sp.misc.imresize(segmask, clusters.shape)
    segmask_[segmask_>0] = 1
    for i in range(hists.shape[0]):
      gauss_win[i,:,:] = np.multiply(gauss_win[i,:,:],segmask_)
      hists[i,:] = np.bincount(clusters.flatten(),
          weights=gauss_win[i,:,:].flatten(),
          minlength=opts_['num_clusters'])
    features = np.append(features, hists)

  return features

def gauss_windows(shape, centers, sigma):
  y = np.arange(shape[0])
  x = np.arange(shape[1])
  x, y = np.meshgrid(x,y)
  windows = np.empty((centers.shape[0],) + shape)
  for i in range(len(centers)):
    windows[i,:,:] = np.exp(-((x-centers[i,1])**2+(y-centers[i,0])**2 )/(2*sigma**2))
  return windows

def grid_centers(array_shape, grid_shape):
  array_shape = np.array(array_shape)
  grid_shape = np.array(grid_shape)
  cell_shape = array_shape / grid_shape.astype(float)
  centers = np.empty((np.prod(grid_shape),grid_shape.size))
  idx = 0
  for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
      centers[idx,0] = cell_shape[0]*(i+.5)
      centers[idx,1] = cell_shape[1]*(j+.5)
      idx += 1
  return centers

