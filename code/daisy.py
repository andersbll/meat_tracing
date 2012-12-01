import numpy as np
import scipy as sp
from numpy.linalg import norm
from scipy import sqrt, pi, arctan2, cos, sin, floor
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import sys
import misc

def daisy(img, step=8, radius=15, rings=3, hists=8, bins=8,
    center_sigma=1.2, ring_sigmas=None, ring_radii=None):
  '''
R  : radius of the descriptor
RQ : number of rings
TQ : number of histograms on each ring
HQ : number of bins of the histograms

SI : spatial interpolation enable/disable
LI : layered interpolation enable/disable
NT : normalization type:
     0 = No normalization
     1 = Partial Normalization
     2 = Full Normalization
     3 = Sift like normalization
'''

  if img.ndim > 2:
    raise ValueError('Only grey-level images are supported.')

  if ring_sigmas != None and ring_radii != None \
      and len(ring_sigmas) != len(ring_radii):
    raise ValueError('len(ring_sigmas) != len(ring_radii)')
  if ring_radii != None:
    rings = len(ring_radii)
    radius = ring_radii[-1]
  if ring_sigmas != None:
    rings = len(ring_sigmas)

  if img.dtype.kind == 'u':
    img = img.astype('float')
    img = img/255.

  # Smooth image
  img_sigma = .5
  img = gaussian_filter(img, sigma=img_sigma)
#  sp.misc.imsave('img_smooth.png', img)

  # Compute derivatives
  hf = np.array([.5, 0, -.5]).reshape(1,3)
  vf = hf.transpose()
  dx = convolve2d(img, hf, mode='same')
  dy = convolve2d(img, vf, mode='same')
#  sp.misc.imsave('img_dx.png', dx)
#  sp.misc.imsave('img_dy.png', dy)
  mag = sqrt(dx**2 + dy**2)


  # Compute orientations and their contribution to each histogram bin
  ori = np.empty((bins,) + img.shape, dtype=float)
  for i in range(bins):
    o = 2*i*pi/bins
    ori[i,:,:] = np.maximum(np.cos(o)*dx + np.sin(o)*dy,0)
    ori[i,:,:] = np.multiply(ori[i,:,:], mag)
    # Smooth
    sigma = sqrt(center_sigma**2 - img_sigma**2);
    ori[i,:,:] = gaussian_filter(ori[i,:,:], sigma=sigma)
#    sp.misc.imsave('img_ori%i.png'%i, ori[i,:,:])

  # Compute smoothed orientations for each ring
  if ring_sigmas == None:
    ring_sigmas = [radius*(i+1)/float(2*rings) for i in range(rings)]
  ring_ori = np.empty((rings,) + ori.shape, dtype=float)
  for i in range(rings):
    if i==0:
      sigma = sqrt(ring_sigmas[i]**2 - center_sigma**2)
      ori_ = ori
    else:
      sigma = sqrt(ring_sigmas[i]**2 - ring_sigmas[i-1]**2)
      ori_ = ring_ori[i-1,:,:,:]
    for j in range(bins):
      ring_ori[i,j,:,:] = gaussian_filter(ori_[j,:,:], sigma=sigma)
#    sp.misc.imsave('img_ring_ori%i.png'%i, ring_ori[i,0,:,:])

  # Compute coordinate offsets for each histogram in every ring
  if ring_radii == None:
    ring_radii = [radius*(i+1)/float(rings) for i in range(rings)]
  theta = [2*pi*j/hists for j in range(hists)]
  coord_off = np.empty((rings, hists, 2))
  for i in range(rings):
    for j in range(hists):
      coord_off[i,j,0] = floor(ring_radii[i]*sin(theta[j]))
      coord_off[i,j,1] = floor(ring_radii[i]*cos(theta[j]))

  # Extract grid of descriptors
  padding = radius+ring_sigmas[-1]
  desc_grid_h = int((img.shape[0]-2*padding) / step)
  desc_grid_w = int((img.shape[1]-2*padding) / step)
  desc_dims = (rings*hists + 1)*bins
  desc_grid = np.empty((desc_grid_h, desc_grid_w, desc_dims))
  desc = np.empty(desc_dims)
  for i in range(desc_grid_h):
    y = padding + i*step
    for j in range(desc_grid_w):
      x = padding + j*step
      idx = 0
      desc[idx:idx+hists] = ori[:,y,x]
      idx += hists
      for k in range(rings):
        x_off = coord_off[k,:,0]
        y_off = coord_off[k,:,0]
        for l in range(hists):
          desc[idx:idx+hists] = ring_ori[k,:,y+y_off[l],x+x_off[l]]
          idx += hists
      desc_grid[i,j,:] = desc
  return desc_grid

def run():
  img = sp.misc.imread('img.png')
  desc_grid = daisy(img)
#  print desc_grid.shape

if __name__ == '__main__':
  run()
