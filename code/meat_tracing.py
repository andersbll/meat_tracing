#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import scipy as sp
import matplotlibinit
import options
import caching
caching.set_cache_dir(options.working_dir)

from canonization import canonize, canonization_training
from feature_extraction import feature_extraction, feature_training
from matching import matching
from dataset import dataset
from misc import print_progress, save_img

@caching.cache
def training(canonize_opts, features_opts):
  print '# Training'
  params = canonization_training(canonize_opts)
  params.update(feature_training(features_opts, params))
  return params

@caching.cache
def extract_features(features_opts, dataset_opts, params):
  print '# Extracting image features'
  files1, files2 = dataset(dataset_opts)
  features = []
  for img_file, depth_file in print_progress(files1 + files2):
    features.append(feature_extraction(img_file, depth_file, features_opts, params))
  return files1, features[:len(features)/2], files2, features[len(features)/2:]


#def output(files1, files2, distances):

##    i += 1
##    fromPath1 = filepaths1[topmatches[idx]].replace('feature_extraction','canonization').replace('features','kam').replace('.npy','.png')
##    fromPath2 = filepaths2[topmatches[idx]].replace('feature_extraction','canonization').replace('features','kam').replace('.npy','.png')
##    toPath1 = os.path.join(opts['outputPath'], 'top_' + str(i) + '_dag1_' + os.path.basename(fromPath1))
##    toPath2 = os.path.join(opts['outputPath'], 'top_' + str(i) + '_dag2_' + os.path.basename(fromPath2))
##    shutil.copy2(fromPath1, toPath1)
##    shutil.copy2(fromPath2, toPath2)

###  i = 0
###  for idx in topmatches[-5:]:
###    i += 1
###    fromPath1 = filepaths1[topmatches[idx]].replace('feature_extraction','canonization').replace('features','kam').replace('.npy','.png')
###    fromPath2 = filepaths2[topmatches[idx]].replace('feature_extraction','canonization').replace('features','kam').replace('.npy','.png')
###    toPath1 = os.path.join(opts['outputPath'], 'bottom_' + str(i) + '_dag1_' + os.path.basename(fromPath1))
###    toPath2 = os.path.join(opts['outputPath'], 'bottom_' + str(i) + '_dag2_' + os.path.basename(fromPath2))
###    shutil.copy2(fromPath1, toPath1)
###    shutil.copy2(fromPath2, toPath2)


if __name__ == '__main__':
  params = training(options.canonization, options.feature_extraction)
  files1, features1, files2, features2 = extract_features(
      options.feature_extraction, options.dataset, params)
  distances = matching(features1, features2, options.matching)

  # Output results
  save_img(os.path.join(options.output_dir, 'confusion.png'), distances)

  num_false = 0
  false_matches = []
  for i in range(distances.shape[0]):
    if i != distances[i,:].argmin():
      num_false += 1
      false_matches.append(i)
  results = {}
  results['Descriptor dimensionality'] = features1[0].shape
  results['Total number of predictions'] = distances.shape[0]
  results['Number of mispredictions'] = num_false
  results['Accuracy'] = 1-float(num_false)/distances.shape[0]
  print results


  for i, idx in enumerate(false_matches):
    img_file1, depth_file1 = files1[idx]
    img_file2, depth_file2 = files2[idx]
    img1, _ = canonize(img_file1, depth_file1, options.canonization, params)
    img2, _ = canonize(img_file2, depth_file2, options.canonization, params)
    sp.misc.imsave(os.path.join(options.output_dir, 'false_match%i_dag1.png'%(i+1)), img1)
    sp.misc.imsave(os.path.join(options.output_dir, 'false_match%i_dag2.png'%(i+1)), img2)

  topmatches = np.diag(distances).argsort()
  for i, idx in enumerate(topmatches[:3]):
    img_file1, depth_file1 = files1[idx]
    img_file2, depth_file2 = files2[idx]
    img1, _ = canonize(img_file1, depth_file1, options.canonization, params)
    img2, _ = canonize(img_file2, depth_file2, options.canonization, params)
    sp.misc.imsave(os.path.join(options.output_dir, 'top%i_dag1.png'%(i+1)), img1)
    sp.misc.imsave(os.path.join(options.output_dir, 'top%i_dag2.png'%(i+1)), img2)
  for i, idx in enumerate(topmatches[-3:]):
    img_file1, depth_file1 = files1[idx]
    img_file2, depth_file2 = files2[idx]
    img1, _ = canonize(img_file1, depth_file1, options.canonization, params)
    img2, _ = canonize(img_file2, depth_file2, options.canonization, params)
    sp.misc.imsave(os.path.join(options.output_dir, 'bottom%i_dag1.png'%(5-i)), img1)
    sp.misc.imsave(os.path.join(options.output_dir, 'bottom%i_dag2.png'%(5-i)), img2)


