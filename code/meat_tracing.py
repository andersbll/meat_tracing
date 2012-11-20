#!/usr/bin/env python
# coding: utf-8

import matplotlibinit

from preprocessing import preprocessing
from segmentation import segmentation
from canonization import canonization, canonizationTraining

options = {
  # Path to the input dataset
  'datasetPath': '/home/abll/meat_tracing/data/input',
  # Path to temporary files generated from the dataset
  'workingPath': '/home/abll/meat_tracing/data/working',
  # Path to results
  'outputPath': '/home/abll/meat_tracing/data/output',

  # Distance from the camera to the surface where the meat is placed
 'boardDepth' : 993.715,

  'segmentation': {
  },
  'canonization': {
    # Number of images to use for determining whether other images are
    # rotated upside down.
    'numTrainImages' : 20,
    # Dimensions of the canonized images.
    'imgShape': (180,600),
  },
  'featureExtraction': {
  },
  'featureClustering': {
  },
}


def run():
  preprocessing(options)
  segmentation(options)
  canonizationTraining(options)
  canonization(options)
#  featureExtraction(options)
#  featureClustering(options)


if __name__ == '__main__':
  run()

