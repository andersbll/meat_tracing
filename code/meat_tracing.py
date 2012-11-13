#!/usr/bin/env python
# coding: utf-8

def X_is_running():
	from subprocess import Popen, PIPE
	p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
	p.communicate()
	return p.returncode == 0

import matplotlib
if not X_is_running():
	# use non-gui backend if X not present
	matplotlib.use('Agg')


from preprocessing import preprocessing
from segmentation import segmentation

options = {
  # Path to the input dataset
  'datasetPath': '/home/abll/meat_tracing/data/input',
  # Path to temporary files generated from the dataset
  'workingPath': '/home/abll/meat_tracing/data/working',
  # Path to results
  'outputPath': '/home/abll/meat_tracing/data/output',
  
  'segmentation': {
    # distance from camera to the surface where the meat is placed
   'boardDepth' : 64541.285,
  },
  'canonization': {
  },
  'featureExtraction': {
  },
  'featureClustering': {
  },
  'classification': {
  }
}


def run():
#  preprocessing(options)
  segmentation(options)
#  canonization(options)
#  featureExtraction(options)
#  featureClustering(options)
#  featureBinning(options)
#  classification(options)



if __name__ == '__main__':
	run()

