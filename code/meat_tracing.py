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
from misc import print_progress, dicvariations
from output import output


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
  variations = [
#      ('daisy_bow_sigma', ['daisy_bow', 'gauss_window_sigma'], [8]),
#      ('daisy_bow_clusters', ['features', 'daisy_bow', 'num_clusters'],
#                             [1100]),
#      ('daisy_radius', ['features', 'daisy_bow', 'feature_opts', 'radius'],
#                             [12,13,14,15]),
#      ('daisy_norm', ['features', 'daisy_bow', 'feature_opts', 'normalization'],
#                             ['l1','l2']),
#      ('jet_bow_clusters', ['features', 'jet_bow', 'num_clusters'],
#                             [900, 1000, 1100]),
#      ('jet_sigma', ['features', 'jet_bow', 'feature_opts', 'sigma'],
#                             [4, 5, 6, 7]),
      ('experiment', ['dataset', 'experiment'],
                             ['afskaering', 'ekstra1', 'ekstra2',
                              'mishandling', 'normal', 'ophaengning']),
  ]

  summary = ''
  for opts, var_name in dicvariations(options.options, variations):
    features_opts = opts['features']
    dataset_opts = opts['dataset']
    matching_opts = opts['matching']
    run_name = var_name
    print '# Running '+run_name
    params = training(features_opts)

    files1, features1, files2, features2 = extract_features(features_opts,
        dataset_opts, params)

    distances = matching(features1, features2, matching_opts)

    results = output(run_name, files1, features1, files2, features2, distances,
        features_opts, dataset_opts, matching_opts, params)
    print results
    summary += run_name + ': ' + str(results['Mispredictions'])+'\n'

  print '# Summary\n'+summary
if __name__ == '__main__':
  run()
