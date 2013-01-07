import os
import shutil
import inspect
import numpy as np
import scipy as sp
import matplotlibinit
import options
import json
from canonization import canonize
from misc import save_img
from dataset import canonize_path


def output(run_name, files1, features1, files2, features2, distances, features_opts,
      dataset_opts, matching_opts, params):

  outdir = os.path.join(options.output_dir, run_name)

  # Delete previous results
  if os.path.exists(outdir):
    shutil.rmtree(outdir)
  os.mkdir(outdir)
#  for f in os.listdir(options.output_dir):
#    os.remove(os.path.join(outdir, f))

  # Output options
  options_outfile = os.path.join(outdir, 'options.py')
  with open(options_outfile, "w") as f:
    f.write(inspect.getsource(options))

  # Output results
  save_img(os.path.join(outdir, 'confusion.png'), distances)

  num_false = 0
  false_matches = []
  for i in range(distances.shape[0]):
    if i != distances[i,:].argmin():
      num_false += 1
      false_matches.append(i)
  results = {}
  results['Descriptor dim.'] = features1[0].shape
  results['Predictions'] = distances.shape[0]
  results['Mispredictions'] = num_false
  results['Accuracy'] = 1-float(num_false)/distances.shape[0]

#  results['False matches'] = []
  for i, idx in enumerate(false_matches):
#    results['False matches'].append(canonize_path(files1[idx][0]))
    output_img(files1[idx], outdir, 'false_match%i_dag1.png'%(i+1), params)
    guessed_idx = distances[idx,:].argmin()
    output_img(files2[guessed_idx], outdir, 'false_match%i_dag2.png'%(i+1), params)
    output_img(files2[idx], outdir, 'false_match%i_dag2_correct.png'%(i+1), params)

  topmatches = np.diag(distances).argsort()
  for i, idx in enumerate(topmatches[:3]):
    output_img(files1[idx], outdir, 'top_match%i_dag1.png'%(i+1), params)
    output_img(files2[idx], outdir, 'top_match%i_dag2.png'%(i+1), params)
  for i, idx in enumerate(topmatches[-3:]):
#    print files2[idx][0], (3-i)
    output_img(files1[idx], outdir, 'bottom_match%i_dag1.png'%(3-i), params)
    output_img(files2[idx], outdir, 'bottom_match%i_dag2.png'%(3-i), params)

  results_file = os.path.join(outdir, 'results.json')
  with open(results_file, "w") as f:
    json.dump(results, f)

  return results

def output_img(filename, outdir, output_filename, params):
  img_file, depth_file = filename
  img, _ = canonize(img_file, depth_file, options.canonization, params)
  sp.misc.imsave(os.path.join(outdir, output_filename), img)

