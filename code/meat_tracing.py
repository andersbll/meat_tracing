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
def training(features_opts):
  params = canonization_training(features_opts['canonization'])
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

def run():
  params = training(options.feature_extraction)
  files1, features1, files2, features2 = extract_features(
      options.feature_extraction, options.dataset, params)
  distances = matching(features1, features2, options.matching)

  # Delete previous results
  for f in os.listdir(options.output_dir):
    os.remove(os.path.join(options.output_dir, f))

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

  results['False matches'] = []
  for i, idx in enumerate(false_matches):
    results['False matches'].append(files1[idx][0])
    output_img(files1[idx], 'false_match%i_dag1.png'%(i+1), params)
    guessed_idx = distances[idx,:].argmin()
    output_img(files2[guessed_idx], 'false_match%i_dag2.png'%(i+1), params)
    output_img(files2[idx], 'false_match%i_dag2_correct.png'%(i+1), params)

  topmatches = np.diag(distances).argsort()
  for i, idx in enumerate(topmatches[:3]):
    output_img(files1[idx], 'top_match%i_dag1.png'%(i+1), params)
    output_img(files2[idx], 'top_match%i_dag2.png'%(i+1), params)
  for i, idx in enumerate(topmatches[-3:]):
    output_img(files1[idx], 'bottom_match%i_dag1.png'%(3-i), params)
    output_img(files2[idx], 'bottom_match%i_dag2.png'%(3-i), params)
  print results

def output_img(filename, output_filename, params):
  img_file, depth_file = filename
  img, _ = canonize(img_file, depth_file, options.canonization, params)
  sp.misc.imsave(os.path.join(options.output_dir, output_filename), img)

if __name__ == '__main__':
  run()
