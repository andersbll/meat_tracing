import os
import misc
import options

def canonize_path(path):
  # normalize path string
  return  '/'.join(path.rsplit('/', 3)[1:])

def prune(filepaths1, filepaths2):
  for f in filepaths1:
    if '131.bmp' in f:
      filepaths1.remove(f)
  filepaths1 = sorted(filepaths1)
  filepaths2 = sorted(filepaths2)
  filepaths1_ = []
  for f in filepaths1:
    if f.replace('Dag 1', 'Dag 2') in filepaths2:
      filepaths1_.append(f)

  filepaths2_ = []
  for f in filepaths2:
    if f.replace('Dag 2', 'Dag 1') in filepaths1_:
      filepaths2_.append(f)

  for i in range(len(filepaths1_)):
    if filepaths1_[i] != filepaths2_[i].replace('Dag 2', 'Dag 1'):
      print 'filepaths do not correspond'
  return filepaths1_, filepaths2_

def depth_path(path):
  return path.replace('_kam', '_dybde')

def training_files(num):
  img_files = misc.gather_files(options.dataset_dir, '*_kam*.bmp')[:num]
  depth_files = map(depth_path, img_files)
  return zip(img_files, depth_files)

def dataset(opts):
  img_paths1 = misc.gather_files(os.path.join(
      options.dataset_dir, 'Dag 1'), 'Normal*_kam*.bmp')
  img_paths2 = misc.gather_files(os.path.join(
      options.dataset_dir, 'Dag 2'), 'Normal*_kam*.bmp')

  img_paths1, img_paths2 = prune(img_paths1, img_paths2)

  depth_paths1 = map(depth_path, img_paths1)
  depth_paths2 = map(depth_path, img_paths2)

  return zip(img_paths1, depth_paths1), zip(img_paths2, depth_paths2)


upsidedown_images = [
  'Dag 2/Normal/Normal_kam52',
  'Dag 2/Normal/Normal_kam38',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam176',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam177',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam178',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam180',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam181',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam183',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam185',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam186',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam191',
  'Dag 2/Ekstra 2/Ekstra billedserie 2_kam192',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam156',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam157',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam158',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam159',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam160',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam161',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam162',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam163',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam164',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam165',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam166',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam167',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam168',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam169',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam170',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam171',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam172',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam173',
  'Dag 2/Ekstra 1/Ekstra billedserie 1_kam174',
]

def is_upside_down(filename):
  file_id = os.path.splitext('/'.join(filename.rsplit('/', 3)[1:]))[0]
  return file_id in upsidedown_images

