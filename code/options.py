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
  'experiment' : 'afskaering',
#  'experiment' : 'ekstra1',
#  'experiment' : 'ekstra2',
#  'experiment' : 'mishandling',
#  'experiment' : 'normal',
#  'experiment' : 'ophaengning',
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
#  'hog': {
#    'orientations': 8,
#    'pixels_per_cell': (22, 22),
#    'cells_per_block': (2, 2),
#    'normalise': False,
#  },

#  'hog_bow': {
#    'grid': (2,4),
#    'num_train_images' : 20,
#    'num_clusters': 401,
#    'hog': {
#      'orientations': 8,
#      'pixels_per_cell': (8, 8),
#      'cells_per_block': (4, 4),
#      'normalise': False,
#    },
#  },
  'daisy_bow': {
    'grid': (2,4),
    'num_train_images' : 12,
    'num_clusters': 500,
    'daisy': {
      'step': 4,
#      'radius': 18,
#      'rings': 3,
#      'hists': 8,
#      'bins':8,
#      'ring_sigmas': [1.7, 3., 5.]
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
  'metric' : 'manhattan',
}

