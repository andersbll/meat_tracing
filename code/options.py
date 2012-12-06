# Path to the input dataset
dataset_dir = '/home/abll/meat_tracing/data/input'

# Path to temporary files generated from the dataset
working_dir = '/home/abll/meat_tracing/data/working'

# Path to results
output_dir = '/home/abll/meat_tracing/data/output'

# Number of threads to use for parallelization
num_threads = 8

# Distance from the camera to the surface where the meat is placed
board_depth = 993.715


dataset = {
#  'experiment' : 'afskaering',
#  'experiment' : 'ekstra1',
#  'experiment' : 'ekstra2',
#  'experiment' : 'mishandling',
#  'experiment' : 'normal',
  'experiment' : 'ophaengning',
}

preprocess_depth = {
  'board_depth' : board_depth,
}

segmentation = {
  'board_depth' : board_depth,
  'preprocess_depth' : preprocess_depth,
}

canonization = {
  # Dimensions of the canonized images.
  'img_shape': (180,600),
  'hist_eq': True,
  # Number of images to use for determining if an image is rotated
  # upside down.
  'num_train_images' : 20,
  'segmentation' : segmentation,
}

feature_extraction = {
  'canonization' : canonization,

  'daisy_bow': {
    'gauss_window_grid': (2,4),
    'gauss_window_sigma': 10,
    'num_train_images' : 20,
    'num_clusters': 500,
    'daisy': {
      'step': 4,
    },
  },
#  'daisy': {
#  },
#  'region_properties': {
#    'grid': (2,6),
#    'properties': [
#      'Area',
#      'WeightedCentroid',
#      'WeightedCentralMoments',
#    ],
#  },
#  'raw_pixels': {
#    'img_scale': .5,
#  },
}

matching = {
#  'bipartite_matching': True,
  'bipartite_matching': False,
  'metric' : 'euclidean',
}

