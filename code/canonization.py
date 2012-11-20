import os
import os.path
import numpy as np
import scipy as sp
import scipy.misc
import misc
import math
import sys

from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.morphology import label


def canonize(img, segMask, opts):
  # Canonize image and its segmentation mask
  # The rotation performed does not consider that an image may be
  # upside down.
  segMask = sp.misc.imresize(segMask, img.shape, interp='nearest')
  img = rgb2gray(img)

  # Rotate image according to the major/minor axis
  props = regionprops(label(segMask), properties=['Orientation','BoundingBox'])
  angle = -props[0]['Orientation']*180/math.pi
  angle_ = props[0]['Orientation']
  bbox_ = props[0]['BoundingBox']
  shape_ = img.shape
  img = sp.misc.imrotate(img, angle)
  segMask = sp.misc.imrotate(segMask, angle, interp='nearest')

  # Crop image according to the bounding box padded by 1 pixel
  props = regionprops(label(segMask), properties=['BoundingBox'])
  bbox = props[0]['BoundingBox']
  t = bbox[0]-1
  l = bbox[1]-1
  b = bbox[2]+1
  r = bbox[3]+1
  img = img[t:b, l:r]
  segMask = segMask[t:b, l:r]

  # Resize image and segmentation mask
  img = sp.misc.imresize(img, opts['canonization']['imgShape'])
  segMask = sp.misc.imresize(segMask, opts['canonization']['imgShape'], interp='nearest')

  # Cut image according to segmentation
  img[segMask==0] = 0
  return img, segMask


def canonizationTraining(opts):
  # Generate average intensity image from a subset of the dataset.
  print '# Canonization, training'
  filepaths = misc.gatherFiles(os.path.join(opts['workingPath'],'preprocessing'), '*_kam*.png')
  filepaths = filepaths[:opts['canonization']['numTrainImages']]
  img_avg = np.zeros(opts['canonization']['imgShape'], dtype=int)
  fileNum = 0
  for inImg in filepaths:
    fileNum += 1
    misc.printProgress(fileNum,len(filepaths))
    inSegMask = inImg.replace('preprocessing','segmentation').replace('_kam','_segmask').replace('.png','.npy')
    img = sp.misc.imread(inImg)
    segMask = np.load(inSegMask)
    img, segMask = canonize(img, segMask, opts)
    # Orient correctly if image is upside down
    if isUpsideDown(inImg):
      img = np.fliplr(np.flipud(img))
    img_avg += img
  img_avg /= fileNum
  sp.misc.imsave(imgAvgPath(opts), img_avg)


def canonization(opts):
  print '# Canonization'
  filepaths = misc.gatherFiles(os.path.join(opts['workingPath'],'preprocessing'), '*_kam*.png')
  fileNum = 0
#  numFalse = 0
  img_avg = sp.misc.imread(imgAvgPath(opts)).astype(int)
  img_avg_upsidedown = np.fliplr(np.flipud(img_avg))
  for inImg in filepaths:
    fileNum += 1
    misc.printProgress(fileNum,len(filepaths))
    inSegMask = inImg.replace('preprocessing','segmentation').replace('_kam','_segmask').replace('.png','.npy')
    outSegMask = inSegMask.replace('segmentation', 'canonization')
    outImg = inImg.replace('preprocessing', 'canonization')
    misc.ensureDir(outImg)

    img = sp.misc.imread(inImg)
    segMask = np.load(inSegMask)
    img, segMask = canonize(img, segMask, opts)

    # Determine if image is upside down.
    diff = np.sum(np.abs(img - img_avg))
    diff_upsidedown = np.sum(np.abs(img - img_avg_upsidedown))
    upsidedown = diff > diff_upsidedown
    if upsidedown:
      img = np.fliplr(np.flipud(img))
      segMask = np.fliplr(np.flipud(segMask))

    sp.misc.imsave(outImg, img)
    np.save(outSegMask, segMask)

#    if isUpsideDown(inImg) != upsidedown:
#      print 'False prediction ' + str(upsidedown) + ': ' + canonizePath(inImg)
#      numFalse += 1
#  print 'Total number of predictions: ' + str(len(filepaths))
#  print 'Total number of false predictions: ' + str(numFalse)
#  print 'Accuracy: ' + str(1-float(numFalse)/len(filepaths))


upsidedown_images = [
'Dag 2/Normal/Normal_kam52',
'Dag 2/Normal/Normal_kam38',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam176',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam177',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam178',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam180',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam181',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam183',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam185',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam186',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam191',
'Dag 2/Ekstra 2/Ekstra billedserie 2_kam192',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam156',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam157',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam158',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam159',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam160',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam161',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam162',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam163',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam164',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam165',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam166',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam167',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam168',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam169',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam170',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam171',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam172',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam173',
'Dag 2/Ekstra 1/Ekstra billedserie 1_kam174',
]

def canonizePath(path):
  # normalize path string
  return  os.path.splitext('/'.join(path.rsplit('/', 3)[1:]))[0]

def isUpsideDown(path):
  path = canonizePath(path)
  return path in upsidedown_images

def imgAvgPath(opts):
  return os.path.join(opts['workingPath'],'kam_avg.png')

