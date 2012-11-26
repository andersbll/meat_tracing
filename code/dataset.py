import os
import misc
import options
import fnmatch
import re
import sys


def canonize_path(path):
  # normalize path string
  return  '/'.join(path.rsplit('/', 3)[1:])

def get_file_number(filepath):
  m = re.match(".*[a-z]+(\d+).*", filepath)
  return int(m.group(1))

def prune(filepaths1, filepaths2):

  # remove bad images
  for f in filepaths1:
    # arm occlusion
    if fnmatch.fnmatch(f, '*Normal*m84.bmp'):
      filepaths1.remove(f)
#    if fnmatch.fnmatch(f, '*Normal*m84_1.bmp'):
#      filepaths1.remove(f)
  for f in filepaths2:
    # overexposed
    if fnmatch.fnmatch(f, '*Normal*m44.bmp'):
      filepaths2.remove(f)
    # meat not there
    if fnmatch.fnmatch(f, '*Normal*m131.bmp'):
      filepaths2.remove(f)
    # just a duplicate
    if fnmatch.fnmatch(f, '*Afskaering*m190.bmp'):
      filepaths2.remove(f)

  # Remove reoccuring images, e.g. if both a normal version and an
  # 'afskaering' version occurs.
  file_numbers2 = set()
  filepaths2_ = []
  for f in filepaths2:
    num = get_file_number(f)
    if num not in file_numbers2:
      file_numbers2.add(num)
      filepaths2_.append(f)
  filepaths2 = filepaths2_

  file_numbers1 = set()
  file_numbers2 = set()
  for f in filepaths1:
    file_numbers1.add(get_file_number(f))
  for f in filepaths2:
    file_numbers2.add(get_file_number(f))

  # Remove images that do not occur both days
  filepaths1 = [f for f in filepaths1 if get_file_number(f) in file_numbers2]
  filepaths2 = [f for f in filepaths2 if get_file_number(f) in file_numbers1]

  # Sort lists to make files match
  filepaths1 = sorted(filepaths1, cmp=lambda x,y: cmp(get_file_number(x), get_file_number(y)))
  filepaths2 = sorted(filepaths2, cmp=lambda x,y: cmp(get_file_number(x), get_file_number(y)))

  # Check that files match
  if len(filepaths1) != len(filepaths2):
    print 'filepaths do not match!'
    sys.exit(0)
  for f1, f2 in zip(filepaths1, filepaths2):
    if get_file_number(f1) != get_file_number(f2):
      print 'filepaths do not match!'
      sys.exit(0)
  return filepaths1, filepaths2

def depth_path(path):
  return path.replace('_kam', '_dybde')

def training_files(num):
  img_files = misc.gather_files(options.dataset_dir, '*_kam*.bmp')[:num]
  depth_files = map(depth_path, img_files)
  return zip(img_files, depth_files)

def dataset(opts):
  img_paths1 = []
  img_paths2 = []
  if opts['experiment'] == 'afskaering':
    img_paths2 = misc.gather_files(os.path.join(
        options.dataset_dir, 'Dag 2'), 'Afskaering*_kam*.bmp')
  elif opts['experiment'] == 'ekstra1':
    img_paths2 = misc.gather_files(os.path.join(
        options.dataset_dir, 'Dag 2'), 'Ekstra billedserie 1*_kam*.bmp')
  elif opts['experiment'] == 'ekstra2':
    img_paths2 = misc.gather_files(os.path.join(
        options.dataset_dir, 'Dag 2'), 'Ekstra billedserie 2*_kam*.bmp')
  elif opts['experiment'] == 'mishandling':
    img_paths2 = misc.gather_files(os.path.join(
        options.dataset_dir, 'Dag 2'), 'Mishandling*_kam*.bmp')
  elif opts['experiment'] == 'ophaengning':
    img_paths2 = misc.gather_files(os.path.join(
        options.dataset_dir, 'Dag 2'), 'Ophaengning*_kam*.bmp')

  img_paths1.extend(misc.gather_files(os.path.join(
      options.dataset_dir, 'Dag 1'), 'Normal*_kam*.bmp'))
  img_paths2.extend(misc.gather_files(os.path.join(
      options.dataset_dir, 'Dag 2'), 'Normal*_kam*.bmp'))

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

