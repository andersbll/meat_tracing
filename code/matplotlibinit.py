import matplotlib
import os

def isXRunning():
  ''' Detect if an X session is available.'''
  from subprocess import Popen, PIPE
  p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
  p.communicate()
  return p.returncode == 0

if os.name == 'posix' and not isXRunning():
  # Use non-gui rendering backend if no X session is available.
  matplotlib.use('Agg')

