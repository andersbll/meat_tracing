import numpy as np
import scipy as sp
import dataset
import caching
from misc import print_progress
from preprocessing import preprocess_image
from segmentation import segmentation

from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.exposure import equalize

def histeq(img, mask=None, nbr_bins=256):
  if mask != None:
    img_masked = np.ma.array(img, mask=mask)
    pixels = img_masked.compressed()
  else:
    pixels = img.flatten()

  #get image histogram
  imhist,bins = np.histogram(pixels, nbr_bins, normed=True)
  cdf = imhist.cumsum() #cumulative distribution function
  cdf = 255 * cdf / cdf[-1] #normalize

  #use linear interpolation of cdf to find new pixel values
  img2 = np.interp(img.flatten(), bins[:-1], cdf)

  return img2.reshape(img.shape).astype(np.uint8)

def _canonize(img, segmask, opts):
  # Canonize image and its segmentation mask
  # The rotation performed does not consider that an image may be
  # upside down.
  segmask = sp.misc.imresize(segmask, img.shape, interp='nearest')
  img = rgb2gray(img)

  # Rotate image according to the major/minor axis
  props = regionprops(label(segmask), properties=['Orientation'])
  angle = -props[0]['Orientation']*180/np.pi
  img = sp.misc.imrotate(img, angle)
  segmask = sp.misc.imrotate(segmask, angle, interp='nearest')

  # Crop image according to the bounding box padded by 1 pixel
  props = regionprops(label(segmask), properties=['BoundingBox'])
  bbox = props[0]['BoundingBox']
  t = bbox[0]-1
  l = bbox[1]-1
  b = bbox[2]+1
  r = bbox[3]+1
  img = img[t:b, l:r]
  segmask = segmask[t:b, l:r]

  # Resize image and segmentation mask
  img = sp.misc.imresize(img, opts['img_shape'])
  segmask = sp.misc.imresize(segmask, opts['img_shape'], interp='nearest')

  # Cut image according to segmentation
  img[segmask==0] = 0

  if opts['hist_eq'] == True:
    # Histogram equalization
    segmask[segmask!=0] = 1
    segmask_ = (segmask+1) % 2
    img_masked = np.ma.array(img, mask=segmask_)
    img = histeq(img, segmask_)

  return img, segmask


@caching.cache
def canonization_training(opts):
  print '## Canonization training'
  params = caching.nul_repr_dict()
  # Generate average intensity image from a subset of the dataset.
  img_avg = np.zeros(opts['img_shape'], dtype=int)
  files = dataset.training_files(opts['num_train_images'])
  for img_file, depth_file in print_progress(files):
    img = preprocess_image(img_file)
    segmask = segmentation(depth_file, opts['segmentation'])
    img, segmask = _canonize(img, segmask, opts)
    # Orient correctly if image is upside down
    if dataset.is_upside_down(img_file):
      img = np.fliplr(np.flipud(img))
    img_avg += img
  img_avg /= len(files)
  params['img_avg'] = img_avg
  params['img_avg_upsidedown'] = np.fliplr(np.flipud(img_avg))
  return params

@caching.cache
def canonize(img_file, depth_file, opts, params):
  img = preprocess_image(img_file)
  segmask = segmentation(depth_file, opts['segmentation'])

  img, segmask = _canonize(img, segmask, opts)

  # Determine if image is upside down.
  diff = np.sum(np.abs(img - params['img_avg']))
  diff_upsidedown = np.sum(np.abs(img - params['img_avg_upsidedown']))
  upsidedown = diff > diff_upsidedown
  if upsidedown:
    img = np.fliplr(np.flipud(img))
    segmask = np.fliplr(np.flipud(segmask))

  return img, segmask

