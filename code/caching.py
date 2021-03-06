import os
import sys
import hashlib
import json
import pickle
import options
import inspect

_cache_dir = os.path.join(os.path.abspath(
    os.path.dirname(sys.argv[0])), "cache")

def set_cache_dir(cache_dir):
  '''Set the root location of the cached files. Must be called before
     the cache decorator class is initialized.'''
  global _cache_dir
  _cache_dir = cache_dir

class nul_repr_dict:
  '''Simple dictionary to store objects that should be ignored by the
  cache decorator class when given as function arguments. This means
  that the contents of this dictionary, when passed as argument to a
  cached function, will not be part of the argument hashing.'''
  def __init__(self):
    self.dict = {}
  def __repr__(self):
    return '<nul_repr_dict()>'
  def __str__(self):
    return '<nul_repr_dict()>'
  def __getitem__(self, key):
    return self.dict[key]
  def __setitem__(self, key, value):
    self.dict[key] = value
  def update(self, other):
    self.dict.update(other.dict)
    return self

class _JSONEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, nul_repr_dict):
      return ''
    return json.JSONEncoder.default(self, obj)

class cache(object):
  '''Decorator class that caches the return value of its function on
  the file system.

  It works similar to the percache package [1], except for:
    - The shelve module is not used for caching return values. Instead,
      return values are pickled to individual files.
    - The function source code is also included in the hash sum such
      that code changes trigger a new function evaluation.
    - It allows for passing arguments that should not be included in
      the hash sum. See the nul_repr_dict class.

  Warning: Don't use this decorator headlessly. It is possible to
  unintentionally cheat the checksum mechanism in several ways. E.g.:
    - If the function relies on code from another part of the program
      and this part is changed.
    - If the function is shadowed by a cache decorator higher in the
      function tree.

  [1]: http://pypi.python.org/pypi/percache
  '''
  def __init__(self, func):
    self.func = func
    self.cache_dir = os.path.join(_cache_dir, func.__module__,  func.__name__)
    self.func_source = inspect.getsource(func)
    if not os.path.exists(self.cache_dir):
      os.makedirs(self.cache_dir)

  def __call__(self, *args):
    checksum = hashlib.sha1(json.dumps(args, sort_keys=True, 
        cls=_JSONEncoder)+self.func_source).hexdigest()
    cache_file = os.path.join(self.cache_dir, checksum)
    if os.path.exists(cache_file):
      with open(cache_file,'rb') as f:
        return pickle.load(f)
    else:
      result = self.func(*args)
      with open(cache_file,'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
      return result

