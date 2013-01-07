import numpy as np
import scipy as sp
#TODO rfft ?
from scipy import sqrt, pi, arctan2, cos, sin, exp
from scipy.ndimage import gaussian_filter
from scipy.misc import factorial
from numpy.linalg import eigh
from skimage import data, img_as_float

np.set_printoptions(precision=4, suppress=True)
def fft2_deriv(img_f, sigma, dy, dx):
    h, w = img_f.shape
    y, x = np.mgrid[-.5 + .5 / h:.5:1. / h, -.5 + .5 / w:.5:1. / w]
    g = exp(- (x**2 + y**2) * (pi*2*sigma)**2 / 2.)
    g = np.fft.fftshift(g)
    if dy > 0 or dx > 0:
        y = np.array((range(0, h/2) + range(-h/2,0) ), dtype=float, ndmin=2) / h
        x = np.array((range(0, w/2) + range(-w/2,0) ), dtype=float, ndmin=2) / w
        dg = (y.T ** dy) * (x ** dx) * (1j * 2 * pi) ** (dy + dx)
        return np.multiply(np.multiply(img_f, g), dg)
    else:
        return np.multiply(img_f, g)

def jet_dimensionality(k):
    return int(factorial(2+k)/(2*factorial(k)))

def jet(img, k_min=1, k_max=3, sigma=5, step=4, rings=3, ring_samplings=8, normalization='l2', whitening=True):

    # Validate image format.
    if img.ndim > 2:
        raise ValueError('Only grey-level images are supported.')
    if img.dtype.kind != 'f':
        img = img_as_float(img)

    if k_min == 0:
        jet_dims = jet_dimensionality(k_max)
    else:
        jet_dims = jet_dimensionality(k_max) - jet_dimensionality(k_min-1)
    deriv_orders = np.empty((jet_dims,2), dtype=int)
    idx = 0
    for order in range(k_min, k_max+1):
        for i in range(order+1):
            deriv_orders[idx,0] = order-i
            deriv_orders[idx,1] = i
            idx += 1

    # Compute image derivatives.
    img_f = np.fft.fft2(img)
    jets = np.zeros((jet_dims,) + img_f.shape)
    for i in range(deriv_orders.shape[0]):
      dy = deriv_orders[i,0]
      dx = deriv_orders[i,1]
      derivatives = np.fft.ifft2(fft2_deriv(img_f, sigma, dy, dx)).real
      jets[i, :, :] = sigma ** order * derivatives

    # Jet whitening
#    covar = np.zeros((jet_dims, jet_dims))
#    for i in range(jet_dims):
#        for j in range(jet_dims):
#            m = deriv_orders[i][0] + deriv_orders[j][0]
#            n = deriv_orders[i][1] + deriv_orders[j][1]
#            if not (n&1 or m&1):
#                order = float(m+n)
#                covar[i,j] = (-1.0) ** (order / 2 + np.sum(deriv_orders[j,:])) \
#                            * factorial(n) * factorial(m) \
#                            / (2 * pi * 2 ** order * order * factorial(n/2) *
#                               factorial(m/2))
#    d, v = eigh(covar)
#    white = np.dot(np.diag(1.0 / sqrt(d)), v.T)
#    for i in range(jets.shape[1]):
#        for j in range(jets.shape[2]):
#            jets[:,i,j] = np.dot(jets[:,i,j], white)
#    jets = jets.swapaxes(0, 1).swapaxes(1, 2)
#    print jets.shape
#    print white.shape
#    print jets[:,50,50] * white
#    print white
#    print white.shape
#    jets = jets[np.newaxis, :,:,:] * white[:, :, np.newaxis, np.newaxis]
#    jets = np.sum(jets,0)
#    print jets.shape
#      desc = params.covariance * desc;

    # Assemble descriptor grid.
    ring_radii = [int(sigma*i) for i in range(1,rings+1)]
    theta = [2 * pi * j / ring_samplings for j in range(ring_samplings)]
    desc_dims = (rings * ring_samplings + 1) * jet_dims
    padding = ring_radii[-1]
    descs = np.empty((desc_dims, img.shape[0] - 2 * padding,
                      img.shape[1] - 2 * padding))
    descs[:jet_dims, :, :] = jets[:, padding:-padding, padding:-padding]
    idx = jet_dims
    for i in range(rings):
        for j in range(ring_samplings):
            if i%2 == 1:
              theta = 2 * pi * (j + 0.5) / ring_samplings
            else:
              theta = 2 * pi * j / ring_samplings
            y_min = padding + int(round(ring_radii[i] * sin(theta)))
            y_max = descs.shape[1] + y_min
            x_min = padding + int(round(ring_radii[i] * cos(theta)))
            x_max = descs.shape[2] + x_min
            descs[idx:idx + jet_dims, :, :] = jets[:, y_min:y_max, x_min:x_max]
            idx += jet_dims
    descs = descs[:, ::step, ::step]
    descs = descs.swapaxes(0, 1).swapaxes(1, 2)

    # Normalize descriptors.
    if normalization != 'off':
        descs += 1e-10
        if normalization == 'l1':
            descs /= np.sum(descs, axis=2)[:, :, np.newaxis]
        elif normalization == 'l2':
            descs /= sqrt(np.sum(descs ** 2, axis=2))[:, :, np.newaxis]

    return descs


def run():
#  img = np.ones((8,8))
  img = data.camera()
#  sp.misc.imsave('woop_orig.png', img)
#  img_f = np.fft.fft2(img)
#  print img.shape
#  print img_f.shape
#  sigma = 4
#  img_f = fft2_deriv(img_f, sigma, 3, 3)
#  img = np.fft.ifft2(img_f).real
#  print np.max(img), np.min(img)
#  sp.misc.imsave('woop.png', img)
#  img = data.camera()
#  img = gaussian_filter(img, sigma)
#  sp.misc.imsave('woop_gaussian_filter.png', img)
#  return
#  y,x = np.mgrid[-50:51,-50:51]
#  img = exp(- (x**2 + y**2)/(2.*1**2))
#  print img
#  print img_f
#  sp.misc.imsave('img_s.png', img)
#  sp.misc.imsave('img_f.png', img_f.real)
#  sp.misc.imsave('img_f_.png', img_f.imag)
#  img = sp.misc.imread('img.png',flatten=True)
  jets = jet(img)
#  print 1./np.sum(jets[10,6,8:16])
#  print grid.shape

if __name__ == '__main__':
  run()
